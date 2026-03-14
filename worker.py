import random
import sys
import json
import io
import base64
import os
from aiohttp import web, ClientResponse
from PIL import Image

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# ComyUI model configuration
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18288
MODEL_LOG_FILE             = '/var/log/portal/comfyui.log'
MODEL_HEALTHCHECK_ENDPOINT = "/health"  # 헬스체크 유지

# ComfyUI가 이미지를 저장하는 아웃풋 폴더 (실제 환경에 맞게 수정해주세요)
COMFY_OUTPUT_DIR = "/workspace/ComfyUI/output"

def extract_filename(response_data: dict):
    """ComfyUI 응답 JSON에서 생성된 파일명 추출"""
    try:
        comfy_resp = response_data.get("comfyui_response", {})
        for data in comfy_resp.values():
            if isinstance(data, dict) and "outputs" in data:
                for node_output in data["outputs"].values():
                    if "images" in node_output and node_output["images"]:
                        return node_output["images"][0].get("filename")
    except Exception:
        pass
    return None

def convert_to_webp_base64(filepath: str, quality: int = 80) -> str:
    """로컬 이미지를 읽어 WebP로 압축한 뒤 Base64 문자열로 반환"""
    img = Image.open(filepath)
    buffer = io.BytesIO()
    img.save(buffer, format="WEBP", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# --- 커스텀 응답 생성기 ---
async def custom_response_generator(client_request: web.Request, model_response: ClientResponse):
    body = await model_response.read()
    
    try:
        resp_data = json.loads(body.decode('utf-8'))
        filename = extract_filename(resp_data)
        
        if filename:
            filepath = os.path.join(COMFY_OUTPUT_DIR, filename)
            
            # 디스크에서 이미지를 읽어 WebP Base64로 압축 후 기존 응답 데이터에 삽입
            if os.path.exists(filepath):
                webp_b64 = convert_to_webp_base64(filepath, quality=80)
                resp_data['images'] = webp_b64
                
                # 삽입한 데이터로 HTTP Body 덮어쓰기
                body = json.dumps(resp_data).encode('utf-8')
            else:
                print(f"[Warning] Image file not found on disk: {filepath}")
                
    except Exception as e:
        print(f"[Error] Failed to process image: {e}")

    # Body의 크기(용량)가 변했으므로 기존 Content-Length 헤더는 지워줍니다 (프레임워크가 자동 재계산)
    headers = model_response.headers.copy()
    headers.pop("Content-Length", None)
    headers.pop("Content-Type", None)
    
    return web.Response(
        body=body,
        status=model_response.status,
        content_type="application/json",
        headers=headers,
    )


# ComyUI-specific log messages
MODEL_LOAD_LOG_MSG = ["To see the GUI go to: "]
MODEL_ERROR_LOG_MSGS = ["MetadataIncompleteBuffer", "Value not in list: ", "[ERROR] Provisioning Script failed"]
MODEL_INFO_LOG_MSGS = ['"message":"Downloading']

benchmark_prompts = [
    "Cartoon hoodie hero; orc, anime cat, bunny; black goo; buff; vector on white.",
    "Cozy farming-game scene with fine details.",
    # ... (기존 프롬프트 생략) ...
]

benchmark_dataset = [
    {
        "input": {
            "request_id": f"test-{random.randint(1000, 99999)}",
            "modifier": "Text2Image",
            "modifications": {
                "prompt": prompt,
                "width": 512,
                "height": 512,
                "steps": 20,
                "seed": random.randint(0, sys.maxsize)
            }
        }
    } for prompt in benchmark_prompts
]

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=100.0,
            response_generator=custom_response_generator,  # <-- 콜백 연결
            benchmark_config=BenchmarkConfig(
                dataset=benchmark_dataset
            )
        )
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

Worker(worker_config).run()