# coding=utf-8
from src.logger import LoggerFactory
from src.prompt_concat import GetManualTestSamples, CreateTestDataset
from src.utils import decode_csv_to_json, load_json, save_to_json
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)
from queue import Queue
from llama_cpp import Llama
from typing import List

import gradio as gr
import logging
import os
import shutil
import torch
import warnings

logger = LoggerFactory.create_logger(name="test", level=logging.INFO)
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

config_data = load_json("config/config.json")
gguf_model_path = config_data["gguf_model_local_path"]
character_path = "./character"
llm = Llama(model_path = gguf_model_path, n_gpu_layers=-1, verbose=True, n_ctx=0)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,trust_remote_code=True)


def generate_with_question(question, role_name, max_seq_len=2048):
    question_in = "\n".join(["\n".join(pair) for pair in question])

    g = GetManualTestSamples(
        role_name=role_name,
        role_data_path=f"./character/{role_name}.json",
        save_samples_dir="./character",
        save_samples_path=role_name + "_rag.json",
        prompt_path="./prompt/dataset_character.txt",
        max_seq_len=max_seq_len
    )
    g.get_qa_samples_by_query(
        questions_query=question_in,
        keep_retrieve_results_flag=True
    )


def create_datasets(role_name):
    testset = []
    role_samples_path = os.path.join("./character", role_name + "_rag.json")

    c = CreateTestDataset(role_name=role_name,
                          role_samples_path=role_samples_path,
                          role_data_path=role_samples_path,
                          prompt_path="./prompt/dataset_character.txt"
                          )
    res = c.load_samples()
    testset.extend(res)
    save_to_json(testset, f"./character/{role_name}_测试问题.json")


def hf_gen(dialog: List, role_name, top_k, top_p, temperature, repetition_penalty, max_dec_len, max_seq_len):
    generate_with_question(dialog, role_name, max_seq_len)
    create_datasets(role_name)

    json_data = load_json(f"{character_path}/{role_name}_测试问题.json")[0]
    text = json_data["input_text"]
    messages = [{"role": "user", "content": text}]

    generation_kwargs = dict(
        messages = messages,
        top_k=int(top_k),
        top_p=float(top_p),
        temperature=float(temperature),
        repeat_penalty=float(repetition_penalty),
        max_tokens=int(max_dec_len),
        stream = True
    )

    # 创建一个Queue来传递生成的文本
    queue = Queue()

    # 定义一个函数来运行生成任务，并将输出放入Queue
    def run_generation():
        for new_text in llm.create_chat_completion(**generation_kwargs):
            if 'content' in new_text['choices'][0]['delta']:
                if new_text['choices'][0]['delta']['content'] == '':
                    continue
                queue.put(new_text['choices'][0]['delta']['content'])

        queue.put(None)  # 使用 None 作为结束信号

    # 启动线程运行生成函数
    thread = Thread(target=run_generation)
    thread.start()

    # 从Queue中读取生成的文本并逐步返回
    answer = ""
    while True:
        chunk = queue.get()
        if chunk is None:
            break
        answer += chunk
        yield answer  # 返回新生成的文本


def generate(chat_history: List, query, role_name, user_name, top_k, top_p, temperature, repetition_penalty, max_dec_len, max_seq_len):
    """generate after hitting "submit" button

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        query (str): query of current round
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n], [q_n+1, a_n+1]]. chat_history + QA of current round.
    """
    assert query != "", "Input must not be empty!!!"
    # apply chat template
    chat_history.append([f"{user_name}:{query}", ""])
    for answer in hf_gen(chat_history, role_name, top_k, top_p, temperature, repetition_penalty, max_dec_len, max_seq_len):
        chat_history[-1][1] = role_name + ":" + answer
        yield gr.update(value=""), chat_history


def regenerate(chat_history: List,role_name,role_desc, top_k, top_p, temperature, repetition_penalty, max_dec_len, max_seq_len):
    """re-generate the answer of last round's query

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        top_p (float): only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature (float): strictly positive float value used to modulate the logits distribution.
        max_dec_len (int): The maximum numbers of tokens to generate.

    Yields:
        List: [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. chat_history
    """
    assert len(chat_history) >= 1, "History is empty. Nothing to regenerate!!"
    if len(chat_history[-1]) > 1:
        chat_history[-1][1] = ""
    # apply chat template
    for answer in hf_gen(chat_history, role_name, top_k, top_p, temperature, repetition_penalty, max_dec_len, max_seq_len):
        chat_history[-1][1] = role_name + ":" + answer
        yield gr.update(value=""), chat_history


def clear_history():
    """clear all chat history

    Returns:
        List: empty chat history
    """
    torch.cuda.empty_cache()
    return []

def save_history(chat_history: List, role_name):
    """save chat history to local file

    Args:
        chat_history (List): [[q_1, a_1], [q_2, a_2], ..., [q_n, a_n]]. list that stores all QA records
        role_name (str): role name

    Returns:
        List: chat history
    """
    role_data_path = os.path.join("./character", role_name + ".json")
    role_data = load_json(role_data_path)

    for chat in chat_history:
        role_data.append(role_data[0].copy())
        role_data[-1]["dialog"] = chat
    save_to_json(role_data, role_data_path)
    gr.Info(f"{role_name}对话保存成功")
    

def generate_file(file_obj, role_info):
    role_name = os.path.basename(file_obj).split(".")[0]
    file_name = f"{role_name}.csv"
    new_path = os.path.join(character_path, file_name)
    shutil.copy(file_obj, new_path)
    decode_csv_to_json(os.path.join(character_path, role_name + ".csv"), role_name, role_info,
                       os.path.join(character_path, role_name + ".json"))
    gr.Info(f"{role_name}生成成功")


def main():
    # launch gradio demo
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("""# Index-1.9B 角色扮演演示""")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""## 👼 角色生成""")
                file_input = gr.File(label="上传角色对话语料(角色名称.csv)")
                role_description = gr.Textbox(label="角色描述", placeholder="请输入角色描述。", lines=2)
                upload_button = gr.Button("生成角色!")
                upload_button.click(generate_file, inputs=[file_input, role_description])

                gr.Markdown("""## 🔧 模型参数""")
                top_k = gr.Slider(0, 10, value=5, step=1, label="单步采用Token的数量")
                top_p = gr.Slider(0.0, 1.0, value=0.8, step=0.1, label="单步累计采用阈值")
                temperature = gr.Slider(0.1, 2.0, value=0.85, step=0.1, label="生成文本随机性（温度）")
                repetition_penalty = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="重复惩罚系数")
                max_dec_len = gr.Slider(1, 4096, value=512, step=1, label="最大输出长度")
                max_seq_len = gr.Slider(1, 4096, value=2048, step=1, label="最大输入长度")

            with gr.Column(scale=10):
                gr.Markdown("""## 😀 角色扮演""")
                chatbot = gr.Chatbot(bubble_full_width=False, height=600, label='Index-1.9B')
                with gr.Row():
                    role_name = gr.Textbox(label="聊天角色", placeholder="输入聊天角色名称。", lines=2, value="三三")
                    user_name = gr.Textbox(label="你的名字", placeholder="输入你的名字。", lines=2, value="主人")
                user_input = gr.Textbox(label="用户", placeholder="输入你的聊天。", lines=2)
                with gr.Row():
                    submit = gr.Button("🚀 提交对话")
                    clear = gr.Button("🧹 清空对话")
                    regen = gr.Button("🔄 重新生成")
                    save = gr.Button("💾 保存对话")

        submit.click(generate, inputs=[chatbot, user_input, role_name, user_name, top_k, top_p, temperature,
                                       repetition_penalty, max_dec_len, max_seq_len],
                     outputs=[user_input, chatbot])
        regen.click(regenerate,
                    inputs=[chatbot, role_name, user_name, top_k, top_p, temperature, repetition_penalty,
                            max_dec_len, max_seq_len],
                    outputs=[user_input, chatbot])
        save.click(save_history, inputs=[chatbot, role_name], outputs=[])
        clear.click(clear_history, inputs=[], outputs=[chatbot])

    demo.queue()
    demo.launch(server_name="0.0.0.0",
                server_port=8002,
                show_error=True,
                share=False)


if __name__ == "__main__":
    main()
