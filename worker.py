import random
import sys
import os
import json
import base64
import aiohttp
from PIL import Image

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig
from aiohttp import web, ClientResponse
from typing import Union

# 기본 Callback URL (클라이언트 요청에 callback_url이 없을 경우 사용하는 fallback)
DEFAULT_CALLBACK_URL = 'https://n8n.kstr.dev/webhook/vast/complete' 

async def custom_response_generator(
    client_request: web.Request,
    model_response: ClientResponse,
) -> Union[web.Response, web.StreamResponse]:
    
    # 1. 클라이언트 요청 바디에서 동적으로 callback_url 추출
    target_callback_url = DEFAULT_CALLBACK_URL
    try:
        req_data = await client_request.json()
        # Body 최상단에 있거나 'input' 객체 내부에 있는 경우 모두 확인
        target_callback_url = req_data.get('callback_url') or req_data.get('input', {}).get('callback_url') or DEFAULT_CALLBACK_URL
        print(f"[DEBUG] Target Callback URL resolved to: {target_callback_url}")
    except Exception as e:
        print(f"[DEBUG] Failed to parse client request for callback_url: {e}")

    # 2. 모델 응답 데이터 처리
    data_bytes = await model_response.read()
    print(f"[DEBUG] Received data length: {len(data_bytes)}")
    
    try:
        data = json.loads(data_bytes)
        print(f"[DEBUG] JSON parsed successfully. Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        # 'output' 리스트를 순회하며 각 항목의 local_path 처리
        if isinstance(data, dict) and 'output' in data and isinstance(data['output'], list):
            modified = False
            for item in data['output']:
                if isinstance(item, dict) and 'local_path' in item:
                    original_path = item['local_path']
                    webp_path = os.path.splitext(original_path)[0] + '.webp'
                    print(f"[DEBUG] Processing local_path: {original_path}")
                    
                    if os.path.exists(original_path):
                        with Image.open(original_path) as img:
                            img.save(webp_path, format='WEBP', quality=87)
                        print(f"[DEBUG] Image saved to webp (quality=87): {webp_path}")
                        
                        with open(webp_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        item['image'] = webp_path
                        item['image_base64'] = encoded_string
                        modified = True
                    else:
                        print(f"[DEBUG] ERROR: File not found at {original_path}")
            
            if modified:
                # 3. 추출된 대상 URL로 원본 데이터(base64 포함) 전송
                if target_callback_url:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(target_callback_url, json=data) as resp:
                                print(f"[DEBUG] Callback sent to {target_callback_url}. Status: {resp.status}")
                    except Exception as e:
                        print(f"[DEBUG] Failed to send callback: {str(e)}")
                
                # 4. 클라이언트 응답용 데이터에서 base64 제거 (용량 최적화)
                for item in data['output']:
                    if isinstance(item, dict) and 'image_base64' in item:
                        del item['image_base64']
                
                # 5. 최종 가공된 데이터를 bytes로 변환
                data_bytes = json.dumps(data).encode('utf-8')
                print(f"[DEBUG] Client response prepared (base64 removed). Final length: {len(data_bytes)}")
        else:
            print(f"[DEBUG] 'output' list not found or not a list.")
            
    except json.JSONDecodeError:
        print(f"[DEBUG] JSONDecodeError: Failed to parse data as JSON.")
    except Exception as e:
        print(f"[DEBUG] Exception occurred: {str(e)}")

    return web.Response(
        body=data_bytes,
        status=model_response.status,
        content_type=model_response.content_type,
        headers={"X-Worker": "my-custom-pyworker"},
    )

# ComyUI model configuration
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18288
MODEL_LOG_FILE             = '/var/log/portal/comfyui.log'
MODEL_HEALTHCHECK_ENDPOINT = "/health"

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
            "workflow_json": {
                # (중략 - 기존 벤치마크 워크플로우 내용과 동일하게 유지하시면 됩니다)
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
            max_queue_time=900.0,
            response_generator=custom_response_generator,
            benchmark_config=BenchmarkConfig(
                dataset=benchmark_dataset,
                runs=3
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
