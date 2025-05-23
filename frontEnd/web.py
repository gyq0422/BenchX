import gradio as gr
import ollama
from backEnd.bert.train import evaluate_model
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments
import requests

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-bxsbnzfgbaclqmzeowhyfyvhxokaismcyzpwawtnernvetev"

def chat_completion(messages, model, history):
    """调用API: https://siliconflow.cn/zh-cn/models"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": messages,
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "<string>",
                    "name": "<string>",
                    "parameters": {},
                    "strict": False
                }
            }
        ]
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()
        print(response.json()['choices'][0]['message']['content'])
        history.append([messages, response.json()['choices'][0]['message']['content']])
        return history
    except Exception as e:
        error_message = f"发生错误: {str(e)}"
        print(error_message)  # 打印错误信息以便调试
        history.append([messages, error_message])
        return history


# 定义与 Ollama 模型的聊天接口
# def chat_with_ollama(user_input, model_name, history):
#     history = history or []
#     prompt = []
#     for human, assistant in history:
#         # 确保跳过 None 值，并且内容一定是字符串类型
#         if human is not None:
#             prompt.append({"role": "user", "content": str(human)})
#
#         if assistant is not None:
#             prompt.append({"role": "assistant", "content": str(assistant)})
#
#     prompt.append({"role": "user", "content": str(user_input)})
#
#     try:
#         # 调用 Ollama API，使用用户选择的模型
#         response = ollama.chat(
#             model=model_name, messages=prompt  # 发送完整的对话历史
#         )
#         print(response)
#
#         # 获取模型回应
#         if 'message' in response and 'content' in response['message']:
#             model_reply = response['message']['content']
#         else:
#             model_reply = "模型未能回应。"
#
#         # 将模型回应添加到对话历史
#         history.append([user_input, model_reply])
#         return history  # 返回更新后的历史记录
#
#     except Exception as e:
#         error_message = f"发生错误: {str(e)}"
#         print(error_message)  # 打印错误信息以便调试
#         history.append([user_input, error_message])
#         return history


# 创建模型选择下拉框



# 聊天界面组件
def chat_page():
    model_options = ["deepseek-ai/DeepSeek-R1","Pro/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B","deepseek-ai/DeepSeek-V3"]
    gr.Markdown("# 智能助手")

    chatbot = gr.Chatbot(
        height=600,
        show_copy_button=True,
    )
    model_dropdown = gr.Dropdown(model_options, label="选择模型", value=model_options[0])

    msg = gr.Textbox(
        placeholder="请输入您的问题...",
        container=False
    )
    clear = gr.Button("清空对话")

    def user(user_message, history):
        print(user_message)
        return "", history + [[user_message, None]]

    def clear_history():
        # 返回一个空的聊天记录列表和一个空的消息框
        return None, []

    msg.submit(  # msg.submit 用于处理文本框提交事件
        user,  # 第一步调用 user 函数
        [msg, chatbot],  # 输入参数:消息框和聊天记录
        [msg, chatbot],  # 输出参数:更新后的消息框和聊天记录
        queue=False  # 不进入队列,立即执行
    ).then(  # 链式调用,在 user 函数执行完后
        chat_completion,  # 调用 AI 对话函数
        [msg, model_dropdown, chatbot],  # 输入参数:消息和聊天记录
        chatbot  # 输出参数:    更新聊天记录
    )

    clear.click(  # clear.click 处理按钮点击事件
        clear_history,  # 调用清空历史记录函数
        None,  # 不需要输入参数
        [msg, chatbot],  # 输出参数：更新消息框和聊天记录
        queue=False  # 不进入队列,立即执行
    )

    return gr.Column([model_dropdown, msg, chatbot, gr.Row(clear)])


# def evaluate_model_with_progress(model_name, dataset_name):
#     """
#     Evaluate a pre-trained BERT model for a given dataset and update the progress bar in real-time.
#     """
#     if model_name == "BERT":
#        evaluate_model(dataset_name)
#        return
#     elif model_name == "GPT-2":
#         exit()



def other_page():
    # 模型选择下拉框
    model_options = ['BERT', 'GPT-2']
    # 数据集选择下拉框
    datasets_options = ['ag_news']

    gr.Markdown("# 大语言模型评测系统")

    # 模型和数据集选择框
    model_dropdown = gr.Dropdown(model_options, label="选择模型", value=model_options[0])
    dataset_dropdown = gr.Dropdown(datasets_options, label="选择数据集", value=datasets_options[0])

    # 结果显示框
    result_output = gr.Textbox(label="评测结果", interactive=False, placeholder="评测结果将在此显示", lines=10)

    # 提交按钮
    submit_btn = gr.Button("提交")

    # 按钮点击后触发评测过程
    submit_btn.click(
        fn=evaluate_model,
        inputs=[dataset_dropdown],  # 传递选定的模型和数据集
        outputs=result_output,  # 输出到结果框
    )

    # 返回构建的界面布局
    return gr.Column([model_dropdown, dataset_dropdown, submit_btn, result_output])


# 创建多个页面界面
with gr.Blocks() as demo:
    with gr.Tab("与模型对话"):
        chat_page()

    with gr.Tab("其他功能"):
        other_page()

# 启动 Gradio 界面
demo.launch(debug=True)
