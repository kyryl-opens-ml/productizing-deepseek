import modal


vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==v0.7.2", "fastapi[standard]==0.115.4"
)

try:
    volume = modal.Volume.from_name("productizing-deepseek", create_if_missing=False).hydrate()
except modal.exception.NotFoundError:
    raise Exception("Download models first")


APP_NAME = "distill-llama-8b"
N_GPU = 1
TOKEN = "token"
GPU_NAME = "A100"
MODEL_DIR = "/productizing-deepseek"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

app = modal.App(APP_NAME)



MINUTES = 60
HOURS = 60 * MINUTES


@app.function(
    image=vllm_image,
    gpu=f"{GPU_NAME}:{N_GPU}",
    timeout=24 * HOURS,
    container_idle_timeout=5 * MINUTES,
    allow_concurrent_inputs=1000,
    volumes={MODEL_DIR: volume},
)
@modal.asgi_app()
def serve():
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import (
        OpenAIServingCompletion,
    )
    from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
    from vllm.usage.usage_lib import UsageContext

    volume.reload()

    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com 🚀",
        version="0.0.1",
        docs_url="/docs",
    )

    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="See code for authentication details.",
    )
    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != TOKEN:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])

    router.include_router(api_server.router)
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODEL_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        enforce_eager=False,
        trust_remote_code=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
    model_config = get_model_config(engine)
    
    async def init_state():
        request_logger = RequestLogger(max_log_len=2048)
        base_model_paths = [
            BaseModelPath(name=MODEL_NAME.split("/")[1], model_path=MODEL_NAME)
        ]

        web_app.state.engine_client = engine
        web_app.state.log_stats = True
        web_app.state.openai_serving_models = OpenAIServingModels(
            base_model_paths=base_model_paths,
            engine_client=engine,
            model_config=model_config,
            lora_modules=[],
            prompt_adapters=[],
        )
        web_app.state.openai_serving_chat = OpenAIServingChat(
            engine,
            model_config=model_config,
            models=web_app.state.openai_serving_models,
            chat_template=None,
            chat_template_content_format="string",
            response_role="assistant",
            request_logger=request_logger,
        )
        web_app.state.openai_serving_completion = OpenAIServingCompletion(
            engine,
            models=web_app.state.openai_serving_models,
            model_config=model_config,
            request_logger=request_logger,
        )

    @web_app.on_event("startup")
    async def startup_event():
        await init_state()

    return web_app



def get_model_config(engine):
    import asyncio
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        model_config = asyncio.run(engine.get_model_config())
    return model_config