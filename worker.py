import random
import sys
import os
import json
import base64
from PIL import Image

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig
from aiohttp import web, ClientResponse
from typing import Union

async def custom_response_generator(
    client_request: web.Request,
    model_response: ClientResponse,
) -> Union[web.Response, web.StreamResponse]:
    data_bytes = await model_response.read()
    
    try:
        data = json.loads(data_bytes)
        if isinstance(data, dict) and 'local_path' in data:
            original_path = data['local_path']
            webp_path = os.path.splitext(original_path)[0] + '.webp'
            
            with Image.open(original_path) as img:
                img.save(webp_path, format='WEBP')
            
            # WebP 이미지를 base64로 인코딩
            with open(webp_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            data['image'] = webp_path
            data['image_base64'] = encoded_string
            data_bytes = json.dumps(data).encode('utf-8')
    except (json.JSONDecodeError, KeyError, Exception):
        # JSON이 아니거나 에러가 발생하면 처리하지 않고 원본 데이터 전송
        pass

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
            max_queue_time=900.0,
            response_generator=custom_response_generator,
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