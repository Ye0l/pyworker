import random
import sys
import json
import io
import base64
import os
import aiohttp
from aiohttp import web, ClientResponse
from PIL import Image

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# ComyUI model configuration
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18288
MODEL_LOG_FILE             = '/var/log/portal/comfyui.log'
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# 결과물을 전송받을 웹훅(n8n, 커스텀 API 등)의 POST URL을 입력하세요.
TARGET_WEBHOOK_URL = "https://n8n.kstr.dev/webhook/vast/complete"

# ComfyUI가 이미지를 저장하는 컨테이너 내부 디스크 경로
COMFY_OUTPUT_DIR = "/workspace/ComfyUI/output"


# ComyUI-specific log messages
MODEL_LOAD_LOG_MSG = [
    "To see the GUI go to: "
]

MODEL_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",
    "Value not in list: ",
    "[ERROR] Provisioning Script failed"
]

MODEL_INFO_LOG_MSGS = [
    '"message":"Downloading'
]

benchmark_prompts = [
    "Cartoon hoodie hero; orc, anime cat, bunny; black goo; buff; vector on white.",
    "Cozy farming-game scene with fine details.",
    "2D vector child with soccer ball; airbrush chrome; swagger; antique copper.",
    "Realistic futuristic downtown of low buildings at sunset.",
    "Perfect wave front view; sunny seascape; ultra-detailed water; artful feel.",
    "Clear cup with ice, fruit, mint; creamy swirls; fluid-sim CGI; warm glow.",
    "Male biker with backpack on motorcycle; oilpunk; award-worthy magazine cover.",
    "Collage for textile; surreal cartoon cat in cap/jeans before poster; crisp.",
    "Medieval village inside glass sphere; volumetric light; macro focus.",
    "Iron Man with glowing axe; mecha sci-fi; jungle scene; dynamic light.",
    "Pope Francis DJ in leather jacket, mixing on giant console; dramatic.",
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


def extract_filename(response_data: dict):
    """ComfyUI 응답 JSON에서 생성된 최종 파일명을 추출합니다."""
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
    """로컬 이미지를 읽어 WebP로 압축한 뒤 Base64 문자열로 반환합니다."""
    img = Image.open(filepath)
    buffer = io.BytesIO()
    img.save(buffer, format="WEBP", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# --- HTTP POST 전송을 위한 커스텀 응답 생성기 ---
async def custom_response_generator(client_request: web.Request, model_response: ClientResponse):
    body = await model_response.read()
    
    try:
        resp_data = json.loads(body.decode('utf-8'))
        filename = extract_filename(resp_data)
        
        if filename:
            filepath = os.path.join(COMFY_OUTPUT_DIR, filename)
            
            # 로컬 디스크에서 이미지를 바로 읽어 WebP로 압축
            if os.path.exists(filepath):
                webp_b64 = convert_to_webp_base64(filepath, quality=80)
                
                # POST로 전송할 JSON 페이로드 구성
                payload = {
                    "event": "image_generation_complete",
                    "filename": filename.replace(".png", ".webp"),
                    "image_webp_base64": webp_b64
                }

                # aiohttp를 사용하여 대상 URL로 POST 요청 전송
                async with aiohttp.ClientSession() as session:
                    async with session.post(TARGET_WEBHOOK_URL, json=payload) as webhook_resp:
                        print(f"[Webhook] Sent WebP image ({len(webp_b64)} bytes) to {TARGET_WEBHOOK_URL}. Status: {webhook_resp.status}")
            else:
                print(f"[Webhook] Image file not found on disk: {filepath}")
                
    except Exception as e:
        print(f"[Webhook] Error processing/sending data: {e}")

    # 원래 클라이언트(API 호출자)에게 보낼 HTTP 응답을 정상적으로 조립해서 리턴
    headers = model_response.headers.copy()
    headers.pop("Content-Type", None)
    return web.Response(
        body=body,
        status=model_response.status,
        content_type=model_response.content_type or None,
        headers=headers,
    )


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT, # 헬스체크 설정 유지
    handlers=[
        HandlerConfig(
            route="/generate/sync",
            allow_parallel_requests=False,
            max_queue_time=10.0,
            response_generator=custom_response_generator, # 커스텀 콜백 연결
            benchmark_config=BenchmarkConfig(
                dataset=benchmark_dataset,
                concurrency=1
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