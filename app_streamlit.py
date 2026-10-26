import streamlit as st
from pathlib import Path
import sys
import os
import json
import pandas as pd
from src.pipeline import Pipeline, max_config
from src.questions_processing import QuestionsProcessor

# 把项目根目录（本文件所在目录）加入模块搜索路径，保证从任何目录启动都能导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 你可以让 root_path 固定，也可以让用户输入（基于脚本位置定位，不依赖启动时的工作目录）
root_path = Path(__file__).resolve().parent / "data" / "stock_data"
pipeline = Pipeline(root_path, run_config=max_config)

st.set_page_config(page_title="企业知识库", layout="wide")

# 页面标题
st.markdown("""
<div style='background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%); padding: 20px 0; border-radius: 12px; text-align: center;'>
    <h2 style='color: white; margin: 0;'>企业知识库</h2>
    <div style='color: #fff; font-size: 16px;'>MinerU 财报深度解析 | 文本分块向量化入库（FAISS） | 按公司路由检索 | 父文档回溯 | LLM 重排序 | 大模型生成带引用定位的结构化答案</div>
</div>
""", unsafe_allow_html=True)


# 左侧输入区
with st.sidebar:
    st.header("查询设置")
    # 仅单问题输入
    user_question = st.text_area("输入问题", "中芯国际的主要业务是什么", height=80)

    st.divider()
    # 重排设置：选择检索后是否用 LLM 对候选重排序
    st.subheader("重排设置")
    rerank_mode = st.radio(
        "重排模式",
        ["不用重排（纯向量）", "LLM重排（qwen-flash）"],
        index=1,  # 默认 LLM 重排，与 max_config 一致
        help="LLM重排：向量先召回一批候选，再由大模型按相关性精排出最终结果；不用重排：直接取向量检索Top N",
    )
    use_rerank = rerank_mode.startswith("LLM")

    st.divider()
    # 检索策略三选一
    st.subheader("检索策略")
    retrieval_mode_map = {
        "向量检索（语义匹配）": "vector",
        "BM25检索（关键词精确匹配）": "bm25",
        "混合检索（向量+BM25融合）": "hybrid",
    }
    retrieval_label = st.radio(
        "检索模式",
        list(retrieval_mode_map.keys()),
        index=0,
        help="向量：按语义相似度检索；BM25：按关键词精确匹配（适合数字、型号、专有名词）；混合：双路召回合并去重",
    )
    retrieval_mode = retrieval_mode_map[retrieval_label]

    st.divider()
    # 父文档回溯开关：命中小块后回溯整段原文喂给大模型
    parent_doc = st.checkbox(
        "父文档回溯",
        value=True,
        help="开启：检索命中小块后回溯整段原文喂给大模型，上下文更完整；关闭：直接用小块，省token更快",
    )

    st.divider()
    # 召回参数：控制向量召回数量和最终送入大模型的文档数
    st.subheader("召回参数")
    sample_size = st.slider(
        "向量召回候选数", min_value=5, max_value=50,
        value=max_config.llm_reranking_sample_size, step=5,
        help="第一轮向量检索召回的候选块数量（LLM重排和混合检索模式下生效）",
    )
    top_n = st.slider(
        "最终返回文档数", min_value=1, max_value=20,
        value=max_config.top_n_retrieval, step=1,
        help="检索（或重排）后最终送入大模型的文档块数量",
    )

    st.divider()
    submit_btn = st.button("生成答案", use_container_width=True)

# 右侧主内容区
st.markdown("<h3 style='margin-top: 24px;'>检索结果</h3>", unsafe_allow_html=True)


def load_subset_mapping():
    # 读取 subset.csv，建立 sha1 -> 文件名 的映射，用于引用溯源定位源 PDF
    csv_path = root_path / "subset.csv"
    mapping = {}
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="gbk")
        for _, row in df.iterrows():
            mapping[str(row["sha1"])] = str(row["file_name"])
    return mapping


def open_source_pdf(sha1):
    # 用系统默认程序打开引用对应的源 PDF（仅本机 Windows 可用，云端部署无本地文件系统）
    if sys.platform != "win32":
        st.info("云端部署环境不支持打开本地 PDF，请在本机运行时使用")
        return
    mapping = load_subset_mapping()
    file_name = mapping.get(str(sha1), "")
    if not file_name:
        st.warning("subset.csv 中未找到该引用对应的文件")
        return
    pdf_path = root_path / "pdf_reports" / f"{file_name}.pdf"
    if pdf_path.exists():
        os.startfile(str(pdf_path))
    else:
        st.warning(f"未找到源 PDF: {pdf_path}")


if submit_btn and user_question.strip():
    with st.spinner("正在生成答案，请稍候..."):
        try:
            answer, retrieval = pipeline.answer_single_question(
                user_question, kind="string",
                use_rerank=use_rerank, sample_size=sample_size, top_n=top_n,
                parent_doc=parent_doc, retrieval_mode=retrieval_mode,
            )
            # 兼容 answer 可能为 str 或 dict
            if isinstance(answer, str):
                try:
                    answer_dict = json.loads(answer)
                except Exception:
                    st.error("返回内容无法解析为结构化答案：" + str(answer))
                    answer_dict = {}
            else:
                answer_dict = answer or {}
            # 兼容两种返回结构：外层带 content 包装 或 直接就是答案字段
            content = answer_dict.get("content", answer_dict)
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    content = {}
            # 各字段直接从 content 取；final_answer 是纯文本，不需要再解析 JSON
            step_by_step = content.get("step_by_step_analysis", "-")
            reasoning_summary = content.get("reasoning_summary", "-")
            relevant_pages = content.get("relevant_pages", [])
            final_answer = content.get("final_answer", "-")
            st.markdown("**分步推理：**")
            st.info(step_by_step)
            st.markdown("**推理摘要：**")
            st.success(reasoning_summary)
            # 相关页面显示为可读文本（markdown 报告中 page 为行区块编号）
            if relevant_pages:
                pages_text = "、".join(f"第 {p} 段" for p in relevant_pages)
                st.markdown(f"**相关页面：** {pages_text}")
            else:
                st.markdown("**相关页面：** 无")
            st.markdown("**最终答案：**")
            # 深色主题下自定义背景必须同时指定深色文字，否则白底白字看不见
            st.markdown(
                f"<div style='background:#e8f0fe;color:#1a1a2e;padding:16px;border-radius:8px;font-size:17px;line-height:1.7;'>{final_answer}</div>",
                unsafe_allow_html=True
            )

            # 引用溯源：展示本次检索命中的文档块（含得分），可展开查看原文并打开源 PDF
            st.divider()
            st.markdown("**引用溯源（点击展开查看命中原文）**")
            results = retrieval.get("results", [])
            if results:
                for i, item in enumerate(results, 1):
                    score = item.get("distance", 0)
                    page = item.get("page", "-")
                    text = (item.get("text", "") or "").strip()
                    preview = text[:80].replace("\n", " ") + ("..." if len(text) > 80 else "")
                    with st.expander(f"[{i}] 第 {page} 段 （得分: {score}）  {preview}"):
                        st.markdown(text)
                # 提供打开源 PDF 的入口
                if st.button("打开源 PDF 文档"):
                    open_source_pdf(answer_dict.get("references", [{}])[0].get("pdf_sha1", ""))
            else:
                st.info("本次检索无溯源结果")
        except Exception as e:
            st.error(f"生成答案时出错: {e}")
else:
    st.info("请在左侧输入问题并点击【生成答案】")

# 知识库管理：PDF 上传入库 + 重建知识库
st.divider()
with st.expander("知识库管理（上传 PDF 入库 / 重建索引）"):
    st.markdown("上传财报 PDF，自动完成：登记 → MinerU 解析 → 分块 → 向量化入库 → BM25 索引")
    uploaded_pdf = st.file_uploader("选择 PDF 文件", type=["pdf"])
    new_company = st.text_input("公司名称（问题路由匹配用）", value="中芯国际", key="ingest_company")
    ingest_btn = st.button("开始入库（MinerU 解析较慢，请耐心等待）", use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        rebuild_btn = st.button("重建索引（分块+向量库+BM25）", use_container_width=True)
    with col2:
        rebuild_bm25_btn = st.button("仅重建 BM25 索引", use_container_width=True)

    if ingest_btn:
        if uploaded_pdf is None:
            st.warning("请先选择 PDF 文件")
        elif not new_company.strip():
            st.warning("请填写公司名称")
        else:
            # 保存到 pdf_reports 目录
            pdf_dir = root_path / "pdf_reports"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = pdf_dir / uploaded_pdf.name
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            try:
                with st.spinner("入库中：MinerU 解析 → 分块 → 向量化 → BM25，可能需要几分钟..."):
                    sha1 = pipeline.ingest_pdf(str(pdf_path), new_company.strip())
                st.success(f"入库完成！sha1={sha1}，现在可以直接提问该公司相关内容了")
            except Exception as e:
                st.error(f"入库失败: {e}")

    if rebuild_btn:
        try:
            with st.spinner("重建中：分块 → 向量化 → BM25 ..."):
                pipeline.chunk_reports()
                pipeline.create_vector_dbs()
                pipeline.create_bm25_db()
            st.success("索引重建完成")
        except Exception as e:
            st.error(f"重建失败: {e}")

    if rebuild_bm25_btn:
        try:
            with st.spinner("重建 BM25 索引中 ..."):
                pipeline.create_bm25_db()
            st.success("BM25 索引重建完成（仅本地计算，不调用 API）")
        except Exception as e:
            st.error(f"重建失败: {e}")
