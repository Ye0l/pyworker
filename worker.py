import random
import sys
import os
import json
import base64
import aiohttp
import urllib.request
from PIL import Image

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig
from aiohttp import web, ClientResponse
from typing import Union

# 기본 Callback URL (클라이언트 요청에 callback_url이 없을 경우 사용하는 fallback)
CALLBACK_POST_URL = os.environ.get('CALLBACK_POST_URL', 'https://n8n.kstr.dev/webhook/vast/complete')
CF_AUTH_KEY = os.environ.get('CF_AUTH_KEY')

async def custom_response_generator(
    client_request: web.Request,
    model_response: ClientResponse,
) -> Union[web.Response, web.StreamResponse]:
    
    data_bytes = await model_response.read()
    
    try:
        data = json.loads(data_bytes)
        
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
            
            if modified and CALLBACK_POST_URL:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(CALLBACK_POST_URL, json=data) as resp:
                            print(f"[DEBUG] Callback sent to {CALLBACK_POST_URL}. Status: {resp.status}")
                except Exception as e:
                    print(f"[DEBUG] Failed to send callback: {str(e)}")

                for item in data['output']:
                    if isinstance(item, dict) and 'image_base64' in item:
                        del item['image_base64']
                
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

# --- 벤치마크 워크플로우 동적 로드 로직 ---
BENCHMARK_WORKFLOW_URL = os.environ.get('BENCHMARK_WORKFLOW_URL')

def inject_dynamic_values(obj, prompt_text):
    """외부에서 받아온 워크플로우의 값 중 예약어를 실제 프롬프트와 시드로 치환합니다."""
    if isinstance(obj, dict):
        return {k: inject_dynamic_values(v, prompt_text) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [inject_dynamic_values(item, prompt_text) for item in obj]
    elif isinstance(obj, str):
        if obj == "__PROMPT__":
            return prompt_text
        elif obj == "__SEED__":
            return random.randint(0, sys.maxsize)
        return obj
    return obj

benchmark_dataset = []

if BENCHMARK_WORKFLOW_URL:
    try:
        print(f"[INFO] Fetching benchmark workflow from: {BENCHMARK_WORKFLOW_URL}")
        req = urllib.request.Request(BENCHMARK_WORKFLOW_URL, headers={'Accept': 'application/json', 'x-kstr-passport': CF_AUTH_KEY})
        with urllib.request.urlopen(req) as response:
            resp_data = json.loads(response.read().decode('utf-8'))
        
        # body.workflow에서 추출 (없을 경우를 대비한 폴백 처리 포함)
        base_workflow = resp_data.get('body', {}).get('workflow')
        if not base_workflow:
            base_workflow = resp_data.get('workflow', resp_data)

        # 받아온 워크플로우 템플릿을 기반으로 benchmark_prompts 수만큼 데이터셋 생성
        for prompt in benchmark_prompts:
            workflow_json = inject_dynamic_values(base_workflow, prompt)
            
            benchmark_dataset.append({
                "input": {
                    "request_id": f"benchmark-{random.randint(1000, 99999)}",
                    "workflow_json": workflow_json
                }
            })
        print(f"[INFO] Successfully loaded {len(benchmark_dataset)} benchmark items from URL.")
    except Exception as e:
        print(f"[ERROR] Failed to load benchmark workflow from URL: {e}")
        # 필요 시 sys.exit(1) 등을 호출하여 워커 실행을 중단시킬 수 있습니다.
else:
    print("[WARN] BENCHMARK_WORKFLOW_URL is not set. Benchmark dataset will be empty.")

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
                runs=1
            ),
            workload_calculator= lambda _ : 200.0
        )
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

Worker(worker_config).run()
