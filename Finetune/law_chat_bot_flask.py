import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
from pydantic import BaseModel


app = FastAPI()

# 解决422问题
class LegalAdviceRequest(BaseModel):
    user_question: str

# 默认的 Prompt 模板
DEFAULT_LEGAL_PROMPT_TEMPLATE = """
你是一个专业的法律助手，专门帮助用户理解和导航复杂的法律问题。请根据以下用户的问题，提供清晰、准确的法律建议，提供法律条款和案例的简要解释，最后引导⽤⼾获取法律帮助或咨询专业律师。

用户问题：
{user_question}

法律助手回答：
"""

def generate_legal_prompt(user_question, prompt_template):
    return prompt_template.format(user_question=user_question)

# 加载模型和分词器
model_path = "/root/Qwen/lora_merge_new"

try:
    # 信任远程代码
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # 将模型移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")

def generate_response(prompt, max_new_tokens=512):
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):]  # 去掉原始 prompt 部分
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response."

@app.post("/get_legal_advice/")
def get_legal_advice_endpoint(user_question: LegalAdviceRequest):
    try:
        
        prompt = generate_legal_prompt(user_question, DEFAULT_LEGAL_PROMPT_TEMPLATE)
        print(prompt)
        response = generate_response(prompt)
        return {"advice": response}
    except Exception as e:
        print(f"Error in endpoint: {e}")
        return {"advice": "Error processing request."}

# Gradio 界面
def legal_assistant_interface(user_input):
    response = get_legal_advice_endpoint(user_input)
    # 保存用户的输入和大模型回答，用于后期的调优
    with open("record.txt", "a") as f:
        f.write("ask:"+user_input+ "\n")
        f.write("response:"+response + "\n")
        f.write("-------------")
    return response["advice"]+"如需更严谨的建议，请线下获取专业法律帮助或咨询律师。"
    
iface = gr.Interface(
    fn=legal_assistant_interface,
    inputs="text",
    outputs="text",
    title="法律问题解答助手",
    description="这是一个qwen使用法律数据微调后的大模型，请输入有关法律问题，以获取专业参考建议。",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    #iface.launch()