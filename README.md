# 企业知识库（财报 RAG 问答系统）

基于深度 RAG（检索增强生成）的财报知识库问答系统：将券商研报、年报 PDF 解析入库，支持按公司路由检索、多种检索策略、LLM 重排序，最终由通义千问生成带引用定位的结构化答案。

## 功能特性

- **PDF 智能入库**：调用 MinerU 云端 API 解析 PDF（OCR + 版面分析），自动按 200 页限制拆分、解析后合并；表格转为 HTML 保留结构，图片抽取为独立文件
- **文本分块**：按行滑窗切块（默认 30 行/块、5 行重叠），块级元数据记录行号定位
- **双索引**：DashScope `text-embedding-v1` 向量化 + FAISS 向量库；BM25 中文分词关键词索引
- **问题路由**：从问题中抽取公司名，自动路由到对应报告的索引，支持多公司比较问答
- **三种检索策略**：向量语义检索 / BM25 关键词检索 / 混合检索（双路召回合并去重）
- **父文档回溯**：命中小块后回溯整段原文喂给大模型，上下文更完整
- **LLM 重排序**：向量先召回候选，qwen-flash 按相关性精排
- **引用溯源**：答案附带相关段落定位与检索得分，可展开查看命中原文、打开源 PDF
- **结构化输出**：Pydantic Schema 约束 LLM 输出 JSON（分步推理/摘要/相关页面/最终答案），支持 string/number/boolean/names/comparative 多种答案类型

## 项目结构

```
RAG-cy/
├── app_streamlit.py          # Web 问答界面（推荐入口）
├── main.py                   # CLI 命令行入口（分步执行/多配置评测）
├── requirements.txt
├── data/stock_data/          # 数据目录
│   ├── pdf_reports/          # 原始 PDF
│   ├── subset.csv            # 文档登记表（sha1 / file_name / company_name）
│   ├── questions.json        # 批量评测问题集
│   ├── debug_data/
│   │   └── 03_reports_markdown/  # MinerU 解析出的 markdown
│   └── databases/
│       ├── chunked_reports/  # 分块结果（含元数据）
│       ├── vector_dbs/       # FAISS 向量库（每份报告一个）
│       └── bm25_dbs/         # BM25 索引（每份报告一个）
├── src/
│   ├── pipeline.py           # 核心调度器：所有流水线方法
│   ├── pdf_mineru.py         # MinerU 解析客户端（拆分/上传/轮询/下载）
│   ├── text_splitter.py      # 分块（含中文分词）
│   ├── ingestion.py          # 向量库 / BM25 索引构建
│   ├── retrieval.py          # 向量 / BM25 / 混合检索器
│   ├── reranking.py          # LLM 重排序
│   ├── questions_processing.py  # 问题处理与路由
│   ├── api_requests.py       # DashScope/OpenAI/Gemini 调用封装
│   └── prompts.py            # 各答案类型的 prompt 模板
```

## 环境准备

1. Python 3.11+，安装依赖：

```powershell
pip install -r requirements.txt
```

2. 配置阿里云 DashScope API Key（系统环境变量或项目根目录 `.env` 文件）：

```
DASHSCOPE_API_KEY=你的key
```

3. 确认 `data/stock_data/subset.csv` 中已登记文档（三列：`sha1, file_name, company_name`），sha1 格式为 `stock_10001` 递增编号。

## 使用方式

### 方式一：Web 界面（推荐）

```powershell
cd RAG-cy
python -m streamlit run app_streamlit.py
```

浏览器打开 http://localhost:8501：

- **左侧查询设置**：输入问题、选择重排模式（LLM 重排/不用重排）、检索策略（向量/BM25/混合）、父文档回溯开关、召回参数（候选数/返回数）
- **右侧检索结果**：分步推理、推理摘要、相关页面、最终答案、引用溯源（展开看命中原文，可打开源 PDF）
- **知识库管理**：上传 PDF + 填公司名 → 一键入库（登记 → MinerU 解析 → 分块 → 向量库 → BM25），也可单独重建索引

注意：修改 `src/` 下后端代码后需重启 Streamlit 服务才生效（模块有进程级缓存）。

### 方式二：命令行批量处理

```powershell
cd RAG-cy

# 分块 + 建库（向量）
python src\pipeline.py
```

直接运行 `pipeline.py` 会依次执行：分块 → 建向量库 → 批量回答 `questions.json`，答案输出到 `data/stock_data/answers_*.json`。需要运行其他步骤（如解析新 PDF），编辑 `pipeline.py` 底部 `__main__`，取消对应方法注释即可。

## 新增一份财报的完整流程

1. 把 PDF 放入 `data/stock_data/pdf_reports/`（或直接在页面"知识库管理"里上传）
2. 在 `subset.csv` 登记一行（或在页面填公司名自动登记），sha1 续号
3. 调用 `pipeline.export_reports_to_markdown('文件名.pdf')` 解析（超过 200 页自动拆分合并）
4. 依次执行 `chunk_reports()`、`create_vector_dbs()`、`create_bm25_db()`
5. 之后提问该公司的内容即可被自动路由检索

（页面"知识库管理"的入库按钮会把 2~5 步自动串起来。）

## 技术栈

| 类别 | 组件 |
|---|---|
| 界面 | Streamlit |
| PDF 解析 | MinerU 云端 API（OCR，200 页/文件上限自动拆分） |
| 分块 | LangChain RecursiveCharacterTextSplitter（按行滑窗） |
| 向量库 | FAISS（IndexFlatIP 余弦内积） |
| 关键词检索 | rank-bm25（中文按字分词） |
| 模型 | 通义千问 qwen-flash（回答/重排）、text-embedding-v1（向量化） |
| 其他 | pandas、pydantic、tiktoken、click |

## 已知限制

- MinerU 单文件上限 200 页，超长 PDF 自动拆成多份解析后合并 markdown
- FAISS 在 Windows 下无法直接读写含中文的路径，代码已通过临时目录中转处理
- PDF 中的图片当前不参与问答（MinerU 抽出的图片文件未接入检索），图片信息依赖周边文字描述
- 表格以 HTML 形式保存在分块文本中，超大表格可能被切分到相邻块
- qwen-flash 等 API 有速率限制，批量答题时 `parallel_requests` 不宜过大
