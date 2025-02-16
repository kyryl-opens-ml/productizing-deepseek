import typer
from openai import OpenAI
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import UserMessage


load_dotenv()

app = typer.Typer()

def call_openai(
    token: str,
    base_url: str,
    prompt: str,
    model_name: str
) -> str:
    client = OpenAI(api_key=token, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    
    return response.choices[0].message.content

@app.command()
def deepseek(prompt: str):
    """Call DeepSeek API"""
    result = call_openai(
        token=os.getenv("DEEPSEEK_API_TOKEN"),
        base_url="https://api.deepseek.com",
        prompt=prompt,
        model_name="deepseek-reasoner"
    )
    print(result)

@app.command()
def groq(prompt: str):
    """Call Groq API"""
    result = call_openai(
        token=os.getenv("GROQ_API_TOKEN"),
        base_url="https://api.groq.com/openai/v1",
        prompt=prompt,
        model_name="deepseek-r1-distill-llama-70b"
    )
    print(result)

@app.command()
def together(prompt: str):
    """Call Together API"""
    result = call_openai(
        token=os.getenv("TOGETHER_API_TOKEN"),
        base_url="https://api.together.xyz/v1",
        prompt=prompt,
        model_name="deepseek-ai/DeepSeek-R1"
    )
    print(result)


@app.command()
def azure(prompt: str):
    """Call Azure API"""
    client = ChatCompletionsClient(
        endpoint=os.getenv("AZURE_INFERENCE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_CREDENTIAL")),
    )    
    print(client)
    r = client.complete(
        messages=[
            UserMessage(content=prompt),
        ],
        model="DeepSeek-R1"
    )
    print(r)

@app.command()
def modal(prompt: str, model_name: str, base_url: str):
    """Call Modal API"""
    result = call_openai(
        token=os.getenv("MODAL_API_TOKEN"),
        base_url=base_url,
        prompt=prompt,
        model_name=model_name
    )
    print(result)

if __name__ == "__main__":
    app()