# Qwen-Turbo API的基础限流设置为每分钟不超过500次API调用（QPM）。同时，Token消耗限流为每分钟不超过500,000 Tokens
from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
import time
import sys

# 把项目根目录（src 的上一级）加入模块搜索路径，
# 保证用 `python src/pipeline.py` 直接运行时也能导入 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pdf_mineru
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.tables_serialization import TableSerializer

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        # self.parsed_reports_dirname = "01_parsed_reports"
        # self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        # self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        #self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        #self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        #self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

@dataclass
class RunConfig:
    # 运行流程参数配置
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    # 检索模式：vector（向量）/ bm25（关键词）/ hybrid（向量+BM25融合）
    retrieval_mode: str = "vector"
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 1 # 并行的数量，需要限制，否则qwen-turbo会超出阈值
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "dashscope" #openai
    answering_model: str = "qwen-flash" # 千问 flash（原 qwen-turbo）
    config_suffix: str = ""

class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        # 初始化主流程，加载路径和配置
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """根据配置初始化所有路径"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV。
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    @staticmethod
    def download_docling_models(): 
        # 下载Docling所需模型，避免首次运行时自动下载
        # PDFParser 依赖较重的 docling，改为方法内延迟导入，避免影响主流程
        from src.pdf_parsing import PDFParser
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析PDF报告，提升处理效率
        参数：
            chunk_size: 每个worker处理的PDF数
            num_workers: 并发worker数
        """
        # PDFParser 依赖较重的 docling，改为方法内延迟导入，避免影响主流程
        from src.pdf_parsing import PDFParser
        logging.basicConfig(level=logging.DEBUG)
        
        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
        
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def export_reports_to_markdown(self, file_name):
        """
        使用 pdf_mineru.py（MinerU 批量上传 API），将指定 PDF 文件转换为 markdown，
        并放到 reports_markdown_path 目录下。
        超过 200 页的 PDF 会被自动拆分解析，最后把各部分的 full.md 合并为一个文件。
        :param file_name: PDF 文件名（如 '【财报】中芯国际：中芯国际2024年年度报告.pdf'）
        """
        print(f"开始处理: {file_name}")
        local_pdf = self.paths.pdf_reports_dir / file_name
        if not local_pdf.exists():
            print(f"未找到 PDF 文件: {local_pdf}")
            return

        # 解析结果（zip 和解压目录）统一放在 debug_data 下的 mineru_download 目录
        download_dir = self.paths.debug_data_path / "mineru_download"
        md_paths = pdf_mineru.extract_pdf(str(local_pdf), output_dir=str(download_dir))
        if not md_paths:
            print(f"解析失败，未生成 markdown: {file_name}")
            return

        # 目标目录和目标文件（文件名与原 PDF 同名，扩展名改为 .md）
        os.makedirs(self.paths.reports_markdown_path, exist_ok=True)
        base_name = os.path.splitext(file_name)[0]
        target_path = os.path.join(self.paths.reports_markdown_path, f"{base_name}.md")
        # 按顺序把各部分 markdown 合并成一个文件（md_paths 已按 part1、part2... 排序）
        with open(target_path, 'w', encoding='utf-8') as out:
            for i, md_path in enumerate(md_paths):
                with open(md_path, 'r', encoding='utf-8') as f:
                    out.write(f.read())
                # 多个分卷之间用空行分隔
                if i < len(md_paths) - 1:
                    out.write('\n\n')
        print(f"已合并 {len(md_paths)} 个部分，保存到: {target_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将规整后 markdown 报告分块，便于后续向量化和检索
        """
        text_splitter = TextSplitter()
        # 只处理 markdown 文件，输入目录为 reports_markdown_path，输出目录为 documents_dir
        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的 markdown 文件...")
        # 自动传入 subset.csv 路径，便于补充 company_name 字段
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.reports_markdown_path,
            output_dir=self.paths.documents_dir,
            subset_csv=self.paths.subset_path
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        """从分块报告创建BM25数据库"""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path
        
        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        # 解析PDF报告，支持并行处理
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库
        """
        print("开始处理报告流程...")
        
        print("步骤1：报告分块...")
        self.chunk_reports()
        
        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()
        
        print("报告处理流程已成功完成！")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀。
        例如：若answers.json已存在，则返回answers_01.json等。
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        # 处理所有问题，生成答案文件
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def answer_single_question(self, question: str, kind: str = "string",
                               use_rerank: bool = None, sample_size: int = None, top_n: int = None,
                               parent_doc: bool = None, retrieval_mode: str = None):
        """
        单条问题即时推理，返回 (结构化答案 dict, 检索结果 dict)。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        use_rerank/sample_size/top_n: 前端可覆盖的重排开关与召回参数，缺省用 run_config
        parent_doc: 前端可覆盖的父文档回溯开关，缺省用 run_config
        retrieval_mode: 检索模式 vector / bm25 / hybrid，缺省用 run_config
        """
        t0 = time.time()
        print("[计时] 开始初始化 QuestionsProcessor ...")
        # 前端未指定时使用 run_config 的默认值
        llm_reranking = self.run_config.llm_reranking if use_rerank is None else use_rerank
        rerank_sample = self.run_config.llm_reranking_sample_size if sample_size is None else sample_size
        top_n_val = self.run_config.top_n_retrieval if top_n is None else top_n
        parent_doc_val = self.run_config.parent_document_retrieval if parent_doc is None else parent_doc
        mode = retrieval_mode or getattr(self.run_config, "retrieval_mode", "vector")
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,  # 单问无需文件
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=parent_doc_val,
            llm_reranking=llm_reranking,
            llm_reranking_sample_size=rerank_sample,
            top_n_retrieval=top_n_val,
            parallel_requests=1,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context,
            retrieval_mode=mode,
            bm25_db_dir=self.paths.bm25_db_path
        )
        t1 = time.time()
        print(f"[计时] QuestionsProcessor 初始化耗时: {t1-t0:.2f} 秒")
        print("[计时] 开始调用 process_single_question ...")
        answer = processor.process_single_question(question, kind=kind)
        t2 = time.time()
        print(f"[计时] process_single_question 推理耗时: {t2-t1:.2f} 秒")
        print(f"[计时] answer_single_question 总耗时: {t2-t0:.2f} 秒")
        # 一并返回本次检索结果（含得分），供前端引用溯源展示
        retrieval = getattr(processor, "last_retrieval", {"company_name": "", "results": []})
        return answer, retrieval

    def _register_in_subset(self, file_name: str, company_name: str) -> str:
        """
        向 subset.csv 登记新文档（file_name 去扩展名），sha1 自动续号（stock_1000x）。
        已登记过同名文件则直接返回现有 sha1。返回登记的 sha1。
        """
        file_no_ext = Path(file_name).stem
        subset_path = self.paths.subset_path
        # 与项目其他地方一致：优先 utf-8，失败用 gbk
        encoding = 'utf-8'
        try:
            df = pd.read_csv(subset_path, encoding='utf-8')
        except UnicodeDecodeError:
            encoding = 'gbk'
            df = pd.read_csv(subset_path, encoding='gbk')
        # 同名文件已登记，直接返回
        if 'file_name' in df.columns and file_no_ext in df['file_name'].astype(str).values:
            return df[df['file_name'].astype(str) == file_no_ext].iloc[0]['sha1']
        # 生成下一个 sha1 编号
        max_num = 0
        for sha in df['sha1'].astype(str):
            if sha.startswith('stock_'):
                try:
                    max_num = max(max_num, int(sha.split('_')[1]))
                except (IndexError, ValueError):
                    continue
        new_sha1 = f"stock_{max_num + 1:05d}"
        new_row = pd.DataFrame([{'sha1': new_sha1, 'file_name': file_no_ext, 'company_name': company_name}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(subset_path, index=False, encoding=encoding)
        print(f"已登记到 subset.csv: {new_sha1} / {file_no_ext} / {company_name}")
        return new_sha1

    def ingest_pdf(self, pdf_path, company_name: str):
        """
        一键入库：登记 subset.csv -> MinerU 解析 -> 分块 -> 建向量库 -> 建 BM25 库。
        :param pdf_path: PDF 文件路径（应位于 pdf_reports 目录下）
        :param company_name: 公司名（问题路由时匹配用）
        :return: 新登记的 sha1
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
        # 1. 登记 subset.csv（幂等：同名文件不重复登记）
        sha1 = self._register_in_subset(pdf_path.name, company_name)
        # 2. MinerU 解析成 markdown（超过 200 页自动拆分并合并）
        print(f"步骤1：MinerU 解析 {pdf_path.name} ...")
        self.export_reports_to_markdown(pdf_path.name)
        # 3. 分块
        print("步骤2：重新分块全部报告 ...")
        self.chunk_reports()
        # 4. 重建向量库
        print("步骤3：重建向量数据库 ...")
        self.create_vector_dbs()
        # 5. 重建 BM25 库
        print("步骤4：重建 BM25 索引 ...")
        self.create_bm25_db()
        print(f"入库完成: {pdf_path.name}（sha1={sha1}）")
        return sha1

preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}

base_config = RunConfig(
    parallel_requests=10,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_base"
)

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=20,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_pdr"
)

## 这里
max_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=4,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = qwen-flash",
    answering_model="qwen-flash",
    config_suffix="_qwen_turbo"
)


configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config}


# 你可以直接在本文件中运行任意方法：
# python .\src\pipeline.py
# 只需取消你想运行的方法的注释即可
# 你也可以修改 run_config 以尝试不同的配置
if __name__ == "__main__":
    # 设置数据集根目录：基于脚本位置定位项目根，不依赖运行时的工作目录
    project_root = Path(__file__).resolve().parent.parent
    root_path = project_root / "data" / "stock_data"
    print('root_path:', root_path)
    #print(type(root_path))
    # 初始化主流程，使用推荐的最佳配置
    pipeline = Pipeline(root_path, run_config=max_config)
    
    print('4. 将pdf转化为纯markdown文本')
    #pipeline.export_reports_to_markdown('【财报】中芯国际：中芯国际2024年年度报告.pdf') 

    # 5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    print('5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    pipeline.chunk_reports() 
    
    # 6. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    print('6. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    pipeline.create_vector_dbs()     
    
    # 7. 处理问题并生成答案，具体逻辑取决于 run_config
    # 默认questions.json
    print('7. 处理问题并生成答案，具体逻辑取决于 run_config')
    pipeline.process_questions() 
    
    print('完成')
