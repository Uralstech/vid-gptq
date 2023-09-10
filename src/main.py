from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI
from uvicorn import run
from typing import Any
import torch

from classes import ChatCompletionRequest, ChatCompletionResponse
from middleware import UMiddleware

# CONSTANTS

# Set USE_FIREBASE_ADMIN_AUTH to `False` if you do not want Firebase Admin SDK authentication.
USE_FIREBASE_ADMIN_AUTH: bool = True
# Set the name or path to your model here.
MODEL_NAME_OR_PATH = "TheBloke/Llama-2-13B-chat-GPTQ"

APP_VERSION: str = "1.0.0"
APP_NAME: str = "Vid GPTQ LLaMA-2 13b" # You can change the name to suite your needs and/or model name.

# APP INITIALIZATION

firebase_app: Any = None
if USE_FIREBASE_ADMIN_AUTH:
    from firebase_admin import initialize_app as initialize_firebase_app
    firebase_app = initialize_firebase_app()

app: FastAPI = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(UMiddleware, use_firebase_admin_auth=USE_FIREBASE_ADMIN_AUTH, firebase_app=firebase_app)

# To use a different branch, change the 'revision' argument.
# Example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH,
                                             torch_dtype=torch.float16,
                                             device_map="auto",
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)

# APP FUNCTIONS

@app.post("/api/chat", response_model=ChatCompletionResponse)
async def chat(request: ChatCompletionRequest) -> ChatCompletionResponse:
    print("Chat-completion request received!")

    prompt = f'''[INST] <<SYS>>
    {request.system or 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don not know the answer to a question, please don not share false information.'}
    <</SYS>>
    {request.user}[/INST]

    '''

    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=request.temperature, max_new_tokens=512)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    result = pipe(prompt)
    output = result[0]["generated_text"]

    print(f"RESULT: {result}")

    print("Sending completion!")
    return ChatCompletionResponse(response=output)

# APP RUN

if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8080)