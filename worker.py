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

MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18288
MODEL_LOG_FILE             = '/var/log/portal/comfyui.log'

TARGET_WS_URL = "ws://여러분의_웹소켓_서버_주소:포트"

# ComfyUI가 이미지를 저장하는 실제 로컬 디스크 경로로 맞춰주세요. (일반적인 기본 경로 예시)
COMFY_OUTPUT_DIR = "/workspace/ComfyUI/output"


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
    
    # 투명도(Alpha) 채널이 있는 경우 RGB로 변환하여 압축 효율을 높이거나, 그대로 WEBP로 저장합니다.
    img.save(buffer, format="WEBP", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


async def custom_response_generator(client_request: web.Request, model_response: ClientResponse):
    body = await model_response.read()
    
    try:
        resp_data = json.loads(body.decode('utf-8'))
        
        # 1. 응답 데이터에서 생성된 파일명 찾기
        filename = extract_filename(resp_data)
        
        if filename:
            filepath = os.path.join(COMFY_OUTPUT_DIR, filename)
            
            # 2. 로컬 디스크에서 이미지를 바로 읽어 WebP로 압축
            if os.path.exists(filepath):
                webp_b64 = convert_to_webp_base64(filepath, quality=80)
                
                # 3. 경량화된 데이터를 웹소켓 페이로드에 담아 전송
                ws_payload = {
                    "event": "image_generation_complete",
                    "filename": filename.replace(".png", ".webp"),
                    "image_webp_base64": webp_b64
                }

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(TARGET_WS_URL) as ws:
                        await ws.send_json(ws_payload)
                        print(f"[Websocket] Successfully sent WebP image ({len(webp_b64)} bytes) to {TARGET_WS_URL}")
            else:
                print(f"[Websocket] Image file not found on disk: {filepath}")
                
    except Exception as e:
        print(f"[Websocket] Error processing/sending data: {e}")

    # 원래 API 요청자에게 보낼 HTTP 응답은 그대로 조립해서 반환
    headers = model_response.headers.copy()
    headers.pop("Content-Type", None)
    return web.Response(
        body=body,
        status=model_response.status,
        content_type=model_response.content_type or None,
        headers=headers,
    )


# ... (이하 기존 WorkerConfig 및 BenchmarkConfig 등 로직은 동일하게 유지) ...

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
            response_generator=custom_response_generator,
            workload_calculator=lambda payload: 250.0,
            benchmark_config=BenchmarkConfig(
                dataset=benchmark_dataset,
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
