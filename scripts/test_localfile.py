import json
import requests
import os

def read_json(file_path):
    """读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json(data, file_path):
    """写入JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def call_deduplication_api(input_data):
    """调用查重API"""
    url = "http://gfda7a93.natappfree.cc/api/v2/analyze"
    # url = "http://localhost:8000/api/v2/analyze"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=input_data, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API调用失败: {response.status_code}, {response.text}")

def main():
    input_file = "../data/body.txt"
    output_file = "../data/output.json"

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"输入文件 {input_file} 不存在！请提供一个有效的JSON文件。")
        return

    # 读取输入数据
    print(f"读取输入文件: {input_file}")
    input_data = read_json(input_file)

    # 调用查重API
    print("调用查重API...")
    try:
        result = call_deduplication_api(input_data)
        print("查重完成！")
    except Exception as e:
        print(f"查重失败: {e}")
        return

    # 保存结果到输出文件
    print(f"保存结果到: {output_file}")
    write_json(result, output_file)
    print("处理完成！")

if __name__ == "__main__":
    main()
