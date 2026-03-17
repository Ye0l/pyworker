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
            "workflow_json":         "10": {
            "_meta": {
                "title": "체크포인트 로드"
            },
            "inputs": {
                "ckpt_name": "animayume_v01.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "12": {
            "_meta": {
                "title": "CLIP 텍스트 인코딩 (프롬프트)"
            },
            "inputs": {
                "clip": [
                    "145",
                    1
                ],
                "text": prompt
            },
            "class_type": "CLIPTextEncode"
        },
        "13": {
            "_meta": {
                "title": "VAE 로드"
            },
            "inputs": {
                "vae_name": "qwen_image_vae.safetensors"
            },
            "class_type": "VAELoader"
        },
        "14": {
            "_meta": {
                "title": "빈 잠재 이미지"
            },
            "inputs": {
                "width": 896,
                "height": 1152,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "15": {
            "_meta": {
                "title": "CLIP 텍스트 인코딩 (프롬프트)"
            },
            "inputs": {
                "clip": [
                    "145",
                    1
                ],
                "text": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, crop image, bad anatomy, furry, futanari, futa"
            },
            "class_type": "CLIPTextEncode"
        },
        "16": {
            "_meta": {
                "title": "VAE 디코드"
            },
            "inputs": {
                "vae": [
                    "13",
                    0
                ],
                "samples": [
                    "71",
                    0
                ]
            },
            "class_type": "VAEDecode"
        },
        "18": {
            "_meta": {
                "title": "CLIP 로드"
            },
            "inputs": {
                "type": "stable_diffusion",
                "device": "default",
                "clip_name": "qwen_3_06b_base.safetensors"
            },
            "class_type": "CLIPLoader"
        },
        "69": {
            "_meta": {
                "title": "VAE 인코드"
            },
            "inputs": {
                "vae": [
                    "127",
                    4
                ],
                "pixels": [
                    "16",
                    0
                ]
            },
            "class_type": "VAEEncode"
        },
        "70": {
            "_meta": {
                "title": "KSampler (nai)"
            },
            "inputs": {
                "cfg": 4.5,
                "model": [
                    "127",
                    0
                ],
                "steps": 30,
                "negative": [
                    "127",
                    2
                ],
                "positive": [
                    "127",
                    1
                ],
                "add_noise": "enable",
                "scheduler": "sgm_uniform",
                "noise_seed": random.randint(0, sys.maxsize),
                "end_at_step": 10000,
                "latent_image": [
                    "69",
                    0
                ],
                "sampler_name": "euler",
                "start_at_step": 12,
                "return_with_leftover_noise": "disable"
            },
            "class_type": "KSamplerAdvanced"
        },
        "71": {
            "_meta": {
                "title": "KSampler (Anima)"
            },
            "inputs": {
                "cfg": 5.5,
                "model": [
                    "145",
                    0
                ],
                "steps": 30,
                "negative": [
                    "15",
                    0
                ],
                "positive": [
                    "12",
                    0
                ],
                "add_noise": "enable",
                "scheduler": "sgm_uniform",
                "noise_seed": random.randint(0, sys.maxsize),
                "end_at_step": 12,
                "latent_image": [
                    "14",
                    0
                ],
                "sampler_name": "euler_ancestral",
                "start_at_step": 0,
                "return_with_leftover_noise": "disable"
            },
            "class_type": "KSamplerAdvanced"
        },
        "76": {
            "_meta": {
                "title": "업스케일 모델 로드"
            },
            "inputs": {
                "model_name": "2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
            },
            "class_type": "UpscaleModelLoader"
        },
        "79": {
            "_meta": {
                "title": "UltralyticsDetectorProvider"
            },
            "inputs": {
                "model_name": "bbox/face_yolov8m.pt"
            },
            "class_type": "UltralyticsDetectorProvider"
        },
        "80": {
            "_meta": {
                "title": "얼굴 디테일러"
            },
            "inputs": {
                "cfg": 4.5,
                "vae": [
                    "127",
                    4
                ],
                "clip": [
                    "127",
                    5
                ],
                "seed": random.randint(0, sys.maxsize),
                "cycle": 1,
                "image": [
                    "196",
                    1
                ],
                "model": [
                    "127",
                    0
                ],
                "steps": 8,
                "denoise": 0.3,
                "feather": 5,
                "max_size": 1536,
                "negative": [
                    "127",
                    2
                ],
                "positive": [
                    "127",
                    1
                ],
                "wildcard": "",
                "drop_size": 10,
                "scheduler": "sgm_uniform",
                "guide_size": 512,
                "noise_mask": True,
                "sam_dilation": 30,
                "sampler_name": "euler",
                "tiled_decode": False,
                "tiled_encode": False,
                "bbox_detector": [
                    "79",
                    0
                ],
                "bbox_dilation": 30,
                "force_inpaint": True,
                "inpaint_model": False,
                "sam_model_opt": [
                    "83",
                    0
                ],
                "sam_threshold": 0.93,
                "bbox_threshold": 0.2,
                "guide_size_for": True,
                "bbox_crop_factor": 3,
                "noise_mask_feather": 20,
                "sam_bbox_expansion": 0,
                "sam_detection_hint": "center-1",
                "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False"
            },
            "class_type": "FaceDetailer"
        },
        "83": {
            "_meta": {
                "title": "SAMLoader (Impact)"
            },
            "inputs": {
                "model_name": "sam_vit_b_01ec64.pth",
                "device_mode": "AUTO"
            },
            "class_type": "SAMLoader"
        },
        "127": {
            "_meta": {
                "title": "Efficient Loader"
            },
            "inputs": {
                "negative": "chinese text, english text, black hair, worst quality, blurry, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands, ambiguous form, colored skin, unfinished, monochrome, sketch, anthro, furry, detailed background, food, bad anatomy, futa, futanari, dickgirl, girl on dick, embedding:lazy/lazyloli, embedding:lazy/lazyneg",
                "positive": prompt,
                "vae_name": "Baked VAE",
                "ckpt_name": "naiXLVpred102d_colorized.safetensors",
                "clip_skip": -2,
                "lora_name": "None",
                "batch_size": 1,
                "lora_stack": [
                    "128",
                    0
                ],
                "empty_latent_width": 512,
                "lora_clip_strength": 1,
                "empty_latent_height": 512,
                "lora_model_strength": 1,
                "token_normalization": "none",
                "weight_interpretation": "comfy"
            },
            "class_type": "Efficient Loader"
        },
        "128": {
            "_meta": {
                "title": "LoRA Stacker"
            },
            "inputs": {
                "lora_wt_1": 1,
                "lora_wt_2": 0.1,
                "lora_wt_3": 0.4,
                "lora_wt_4": 0.2,
                "lora_wt_5": 0.3,
                "lora_wt_6": 0.2,
                "lora_wt_7": 0.2,
                "lora_wt_8": 0.8,
                "lora_wt_9": 1,
                "clip_str_1": 1,
                "clip_str_2": 1,
                "clip_str_3": 1,
                "clip_str_4": 1,
                "clip_str_5": 1,
                "clip_str_6": 1,
                "clip_str_7": 1,
                "clip_str_8": 1,
                "clip_str_9": 1,
                "input_mode": "simple",
                "lora_count": 8,
                "lora_wt_10": 1,
                "lora_wt_11": 1,
                "lora_wt_12": 1,
                "lora_wt_13": 1,
                "lora_wt_14": 1,
                "lora_wt_15": 1,
                "lora_wt_16": 1,
                "lora_wt_17": 1,
                "lora_wt_18": 1,
                "lora_wt_19": 1,
                "lora_wt_20": 1,
                "lora_wt_21": 1,
                "lora_wt_22": 1,
                "lora_wt_23": 1,
                "lora_wt_24": 1,
                "lora_wt_25": 1,
                "lora_wt_26": 1,
                "lora_wt_27": 1,
                "lora_wt_28": 1,
                "lora_wt_29": 1,
                "lora_wt_30": 1,
                "lora_wt_31": 1,
                "lora_wt_32": 1,
                "lora_wt_33": 1,
                "lora_wt_34": 1,
                "lora_wt_35": 1,
                "lora_wt_36": 1,
                "lora_wt_37": 1,
                "lora_wt_38": 1,
                "lora_wt_39": 1,
                "lora_wt_40": 1,
                "lora_wt_41": 1,
                "lora_wt_42": 1,
                "lora_wt_43": 1,
                "lora_wt_44": 1,
                "lora_wt_45": 1,
                "lora_wt_46": 1,
                "lora_wt_47": 1,
                "lora_wt_48": 1,
                "lora_wt_49": 1,
                "lora_wt_50": 1,
                "clip_str_10": 1,
                "clip_str_11": 1,
                "clip_str_12": 1,
                "clip_str_13": 1,
                "clip_str_14": 1,
                "clip_str_15": 1,
                "clip_str_16": 1,
                "clip_str_17": 1,
                "clip_str_18": 1,
                "clip_str_19": 1,
                "clip_str_20": 1,
                "clip_str_21": 1,
                "clip_str_22": 1,
                "clip_str_23": 1,
                "clip_str_24": 1,
                "clip_str_25": 1,
                "clip_str_26": 1,
                "clip_str_27": 1,
                "clip_str_28": 1,
                "clip_str_29": 1,
                "clip_str_30": 1,
                "clip_str_31": 1,
                "clip_str_32": 1,
                "clip_str_33": 1,
                "clip_str_34": 1,
                "clip_str_35": 1,
                "clip_str_36": 1,
                "clip_str_37": 1,
                "clip_str_38": 1,
                "clip_str_39": 1,
                "clip_str_40": 1,
                "clip_str_41": 1,
                "clip_str_42": 1,
                "clip_str_43": 1,
                "clip_str_44": 1,
                "clip_str_45": 1,
                "clip_str_46": 1,
                "clip_str_47": 1,
                "clip_str_48": 1,
                "clip_str_49": 1,
                "clip_str_50": 1,
                "lora_name_1": "spo_sdxl_10ep_4k-data_lora_webui.safetensors",
                "lora_name_2": "noobaiXLNAIXL_epsilonPred11Version-lora.safetensors",
                "lora_name_3": "il_contrast_slider_d1.safetensors",
                "lora_name_4": "guilty_gear_strive_style_trigger_gu1lty.safetensors",
                "lora_name_5": "benismanXL_il_lokr_V6311P.safetensors",
                "lora_name_6": "rimixO.safetensors",
                "lora_name_7": "realistic_cum_v1.safetensors",
                "lora_name_8": "Lee Ha-neul prefectPonyxl.safetensors",
                "lora_name_9": "None",
                "model_str_1": 1,
                "model_str_2": 1,
                "model_str_3": 1,
                "model_str_4": 1,
                "model_str_5": 1,
                "model_str_6": 1,
                "model_str_7": 1,
                "model_str_8": 1,
                "model_str_9": 1,
                "lora_name_10": "None",
                "lora_name_11": "None",
                "lora_name_12": "None",
                "lora_name_13": "None",
                "lora_name_14": "None",
                "lora_name_15": "None",
                "lora_name_16": "None",
                "lora_name_17": "None",
                "lora_name_18": "None",
                "lora_name_19": "None",
                "lora_name_20": "None",
                "lora_name_21": "None",
                "lora_name_22": "None",
                "lora_name_23": "None",
                "lora_name_24": "None",
                "lora_name_25": "None",
                "lora_name_26": "None",
                "lora_name_27": "None",
                "lora_name_28": "None",
                "lora_name_29": "None",
                "lora_name_30": "None",
                "lora_name_31": "None",
                "lora_name_32": "None",
                "lora_name_33": "None",
                "lora_name_34": "None",
                "lora_name_35": "None",
                "lora_name_36": "None",
                "lora_name_37": "None",
                "lora_name_38": "None",
                "lora_name_39": "None",
                "lora_name_40": "None",
                "lora_name_41": "None",
                "lora_name_42": "None",
                "lora_name_43": "None",
                "lora_name_44": "None",
                "lora_name_45": "None",
                "lora_name_46": "None",
                "lora_name_47": "None",
                "lora_name_48": "None",
                "lora_name_49": "None",
                "lora_name_50": "None",
                "model_str_10": 1,
                "model_str_11": 1,
                "model_str_12": 1,
                "model_str_13": 1,
                "model_str_14": 1,
                "model_str_15": 1,
                "model_str_16": 1,
                "model_str_17": 1,
                "model_str_18": 1,
                "model_str_19": 1,
                "model_str_20": 1,
                "model_str_21": 1,
                "model_str_22": 1,
                "model_str_23": 1,
                "model_str_24": 1,
                "model_str_25": 1,
                "model_str_26": 1,
                "model_str_27": 1,
                "model_str_28": 1,
                "model_str_29": 1,
                "model_str_30": 1,
                "model_str_31": 1,
                "model_str_32": 1,
                "model_str_33": 1,
                "model_str_34": 1,
                "model_str_35": 1,
                "model_str_36": 1,
                "model_str_37": 1,
                "model_str_38": 1,
                "model_str_39": 1,
                "model_str_40": 1,
                "model_str_41": 1,
                "model_str_42": 1,
                "model_str_43": 1,
                "model_str_44": 1,
                "model_str_45": 1,
                "model_str_46": 1,
                "model_str_47": 1,
                "model_str_48": 1,
                "model_str_49": 1,
                "model_str_50": 1
            },
            "class_type": "LoRA Stacker"
        },
        "145": {
            "_meta": {
                "title": "LoRA 로드"
            },
            "inputs": {
                "clip": [
                    "18",
                    0
                ],
                "model": [
                    "10",
                    0
                ],
                "lora_name": "anima-masterpieces-nlmix2-e41.safetensors",
                "strength_clip": 1,
                "strength_model": 0.9
            },
            "class_type": "LoraLoader"
        },
        "174": {
            "_meta": {
                "title": "이미지 저장"
            },
            "inputs": {
                "images": [
                    "199",
                    0
                ],
                "filename_prefix": "ServerlessImage"
            },
            "class_type": "SaveImage"
        },
        "175": {
            "_meta": {
                "title": "이미지 미리보기"
            },
            "inputs": {
                "images": [
                    "16",
                    0
                ]
            },
            "class_type": "PreviewImage"
        },
        "196": {
            "_meta": {
                "title": "Latent Scale (on Pixel Space)"
            },
            "inputs": {
                "vae": [
                    "127",
                    4
                ],
                "samples": [
                    "70",
                    0
                ],
                "scale_factor": 2,
                "scale_method": "lanczos",
                "use_tiled_vae": False,
                "upscale_model_opt": [
                    "76",
                    0
                ]
            },
            "class_type": "LatentPixelScale"
        },
        "197": {
            "_meta": {
                "title": "이미지 미리보기"
            },
            "inputs": {
                "images": [
                    "80",
                    0
                ]
            },
            "class_type": "PreviewImage"
        },
        "199": {
        "_meta": {
            "title": "Clear Cache All"
        },
        "inputs": {
            "anything": [
                "80",
                0
            ]
        },
        "class_type": "easy clearCacheAll"
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
