import uuid
from pymongo import MongoClient
import numpy as np
from datetime import datetime, timezone
from PIL import Image
import os
import psutil  # 메모리 측정을 위한 라이브러리

# MongoDB 연결
client = MongoClient("mongodb://localhost:27017/")
db = client["test_database"]
collection = db["test_collection"]

# 데이터 가져오기: 첫 번째 문서부터 최대 12,000개
query_start = datetime.now()  # 쿼리 시작 시간
cursor = collection.find().limit(8000)  # 처음부터 최대 12000개만 가져옴
data_list = list(cursor)  # 커서를 리스트로 변환
query_end = datetime.now()  # 쿼리 종료 시간

tick_count = len(data_list)  # 검색된 문서 개수

# 메모리 측정 함수
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"현재 메모리 사용량: {mem_info.rss / (1024 * 1024):.2f} MB")  # RSS 메모리 사용량

# 이미지 로드 및 numpy 변환
def load_and_convert_image(image_url):
    start = datetime.now()  # 이미지 로드 시작
    try:
        image_path = image_url  # 로컬 이미지 경로
        image = Image.open(image_path).convert("RGB")  # 이미지를 RGB로 변환
        image_array = np.array(image)  # numpy 배열로 변환
        duration = datetime.now() - start  # 이미지 로드 시간 측정
        return image_array, duration
    except Exception as e:
        duration = datetime.now() - start
        print(f"이미지 로드 실패: {image_url}, 오류: {e}, 소요 시간: {duration}")
        return None, duration

# 데이터 병합 로직
def merge_robot_data(data_list):
    robot_merged = {}
    camera_merged = {}
    image_load_time = datetime.now() - datetime.now()  # 초기화 (0초)

    merge_start = datetime.now()  # 병합 시작 시간
    for idx, data in enumerate(data_list):
        # 메모리 상태 출력
        if idx % 100 == 0:  # 100개 처리할 때마다 메모리 사용량 출력
            print_memory_usage()

        # 로봇 데이터 병합
        for robot in data.get("robot_data", []):
            robot_id = robot["robot_id"]
            if robot_id not in robot_merged:
                robot_merged[robot_id] = {}

            for key, value in robot.items():
                if key in ["robot_id", "timestamp"]:
                    continue
                if isinstance(value, list):
                    if key not in robot_merged[robot_id]:
                        robot_merged[robot_id][key] = []
                    robot_merged[robot_id][key].append(value)

        # 카메라 데이터 병합 (이미지 변환 포함)
        for camera in data.get("camera_data", []):
            camera_name = camera["camera_name"]
            image_url = camera["image_url"]

            if camera_name not in camera_merged:
                camera_merged[camera_name] = []

            # 이미지 불러와 numpy로 변환
            image_array, load_time = load_and_convert_image(image_url)
            image_load_time += load_time  # 누적 이미지 로드 시간
            if image_array is not None:
                camera_merged[camera_name].append(image_array)

    merge_end = datetime.now()  # 병합 종료 시간
    merge_duration = merge_end - merge_start  # 병합에 걸린 시간

    # 병합된 데이터를 numpy 배열로 변환 (로봇 데이터)
    for robot_id, robot_data in robot_merged.items():
        for key, values in robot_data.items():
            robot_merged[robot_id][key] = np.array(values).tolist()

    return robot_merged, camera_merged, merge_duration, image_load_time

# 전체 병합 시작 시간
total_start = datetime.now()

# 병합 수행
robot_data_merged, camera_data_merged, merge_duration, image_load_time = merge_robot_data(data_list)

total_end = datetime.now()
total_duration = total_end - total_start  # 전체 소요 시간

# 최종 메모리 사용량 출력
print_memory_usage()

# 결과 출력
print(f"쿼리 소요 시간: {query_end - query_start}")
print(f"이미지 로드 시간: {image_load_time}")
print(f"병합 소요 시간: {merge_duration}")
print(f"총 소요 시간: {total_duration}")
print(f"검색된 문서 개수: {tick_count}")