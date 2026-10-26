import os
import time
import zipfile
import requests
from pypdf import PdfReader, PdfWriter

# MinerU API Key 从环境变量读取，避免硬编码泄露
api_key = os.getenv('MINERU_API_KEY', '')

# MinerU 限制单个文件最多 200 页
MAX_PAGES = 200


def split_pdf(local_pdf):
    """
    按页数上限拆分 PDF，返回拆分后的文件路径列表。
    不超过上限时直接返回原文件。
    """
    reader = PdfReader(local_pdf)
    total = len(reader.pages)
    if total <= MAX_PAGES:
        return [local_pdf]

    base = os.path.splitext(local_pdf)[0]
    part_paths = []
    part_index = 0
    start = 0
    while start < total:
        end = min(start + MAX_PAGES, total)
        part_index += 1
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        part_path = f"{base}_part{part_index}.pdf"
        with open(part_path, 'wb') as f:
            writer.write(f)
        print(f"已拆分: {part_path} (第 {start + 1}-{end} 页, 共 {end - start} 页)")
        part_paths.append(part_path)
        start = end
    return part_paths


def create_batch_task(file_names):
    # 向 MinerU 申请批量上传链接，一次批量提交多个拆分后的文件
    url = 'https://mineru.net/api/v4/file-urls/batch'
    header = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    data = {
        'enable_formula': False,
        'language': 'ch',
        'files': [{'name': os.path.basename(name), 'is_ocr': True} for name in file_names],
    }
    res = requests.post(url, headers=header, json=data)
    print(res.status_code)
    print(res.json())
    body = res.json()
    if body.get('code') != 0:
        raise RuntimeError(f"申请上传链接失败: {body.get('msg')}")
    batch_id = body['data']['batch_id']
    upload_urls = body['data']['file_urls']
    return batch_id, upload_urls


def upload_file(upload_url, local_path):
    # 用 PUT 方式把本地文件上传到 MinerU 返回的预签名链接（该请求不需要 Authorization）
    with open(local_path, 'rb') as f:
        res = requests.put(upload_url, data=f)
    print(f"上传 {os.path.basename(local_path)} 状态码: {res.status_code}")
    res.raise_for_status()


def get_result(batch_id):
    """
    轮询批量任务直到全部完成，下载并解压每个文件的解析结果。
    返回 [(file_name, full_md 路径), ...] 列表；解析失败的文件不在列表中。
    下载的 zip 和解压目录都保存在当前工作目录下。
    """
    url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'
    header = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    md_results = []
    while True:
        res = requests.get(url, headers=header)
        body = res.json()
        if body.get('code') != 0:
            print(f"查询失败: {body.get('msg')}")
            return md_results
        items = body['data'].get('extract_result', [])
        states = [item.get('state') for item in items]
        print(f"当前状态: {states}")
        # 只要还有任务在处理中，就等待后重试
        if any(s in ['waiting-file', 'pending', 'running'] for s in states):
            print("任务未完成，等待10秒后重试...")
            time.sleep(10)
            continue
        # 逐个处理已完成的任务
        for item in items:
            file_name = item.get('file_name', '')
            state = item.get('state')
            err_msg = item.get('err_msg', '')
            if state == 'failed' or err_msg:
                print(f"{file_name} 任务出错: {err_msg}")
                continue
            if state == 'done':
                full_zip_url = item.get('full_zip_url')
                if full_zip_url:
                    local_filename = f"{os.path.splitext(file_name)[0]}.zip"
                    print(f"开始下载: {full_zip_url}")
                    r = requests.get(full_zip_url, stream=True)
                    with open(local_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(f"下载完成，已保存到: {local_filename}")
                    # 下载完成后自动解压（返回绝对路径，避免调用方工作目录不同导致找不到文件）
                    extract_dir = unzip_file(local_filename)
                    md_path = os.path.abspath(os.path.join(extract_dir, 'full.md'))
                    if os.path.exists(md_path):
                        md_results.append((file_name, md_path))
                    else:
                        print(f"{file_name} 解压结果中未找到 full.md")
                else:
                    print(f"{file_name} 未找到 full_zip_url，无法下载。")
        return md_results


# 解压zip文件的函数
def unzip_file(zip_path, extract_dir=None):
    """
    解压指定的zip文件到目标文件夹，返回解压目录。
    :param zip_path: zip文件路径
    :param extract_dir: 解压目标文件夹，默认为zip同名目录
    """
    if extract_dir is None:
        # 用 removesuffix 去掉扩展名，避免 rstrip 按字符集误删文件名末尾字符
        extract_dir = zip_path.removesuffix('.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")
    return extract_dir


def extract_pdf(local_pdf, output_dir=None):
    """
    一站式入口：拆分 → 上传 → 云端解析 → 下载解压。
    :param local_pdf: 本地 PDF 文件路径
    :param output_dir: 下载和解压的输出目录，默认为当前工作目录
    :return: 按拆分顺序排列的 full.md 路径列表
    """
    if output_dir is None:
        output_dir = os.getcwd()
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 切换到输出目录，保证 zip 和解压结果统一落在这里
    old_cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        # 按页数上限拆分（超过 200 页时拆成多个文件）
        part_paths = split_pdf(local_pdf)
        # 申请批量上传链接并上传所有拆分文件
        batch_id, upload_urls = create_batch_task(part_paths)
        print('batch_id:', batch_id)
        for upload_url, part_path in zip(upload_urls, part_paths):
            upload_file(upload_url, part_path)
        # 轮询解析结果，完成后下载并解压
        md_results = get_result(batch_id)
    finally:
        os.chdir(old_cwd)

    # 按文件名排序，保证 part1 在 part2 之前，顺序与原 PDF 页码一致
    md_results.sort(key=lambda x: x[0])
    return [md_path for _, md_path in md_results]


if __name__ == "__main__":
    # 项目根目录（src 的上一级），无论从哪个目录运行脚本都能正确定位文件
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 本地 PDF 文件路径（用绝对路径，避免相对路径受运行目录影响）
    local_pdf = os.path.join(project_root, 'data', 'stock_data', 'pdf_reports', '【财报】中芯国际：中芯国际2024年年度报告.pdf')
    # 完整流程：解析结果输出到项目根目录
    md_paths = extract_pdf(local_pdf, output_dir=project_root)
    print('解析出的 markdown 文件:', md_paths)
