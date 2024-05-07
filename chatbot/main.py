import gradio as gr
from dotenv import load_dotenv

import os
import time

load_dotenv()

from chatbot.api import services


def add_message(history, message):
    # for x in message["files"]:
    #     history.append(((x,), None))
    # if message["text"] is not None:
    history.append((message, None))
    return history, gr.Textbox(value=None, interactive=False)


def bot(history):
    print(f"{history = }")
    user_message = history[-1][0]
    response = services.send_message(user_message)
    
    history[-1][1] = response
    return history


with gr.Blocks() as demo:
    with gr.Column(scale=0.60):
        chatbot = gr.Chatbot(value=[], elem_id="chatbot", bubble_full_width=False)

        chat_input = gr.Textbox(
            interactive=True,
            placeholder="Enter message",
            show_label=False,
        )

        chat_msg = chat_input.submit(
            add_message, [chatbot, chat_input], [chatbot, chat_input]
        )
        bot_msg = chat_msg.then(bot, chatbot, chatbot, api_name="bot_response")
        bot_msg.then(lambda: gr.Textbox(interactive=True), None, [chat_input])


    with gr.Column(scale=0.20):
        upload_pdf = gr.File(
            label="Upload a pdf",
            file_count="multiple",
            file_types=[".pdf"],
            type="filepath",
        )

        upload_pdf.upload(fn=services.upload_pdf, inputs=[upload_pdf])

if __name__ == '__main__':
    demo.queue()
    demo.launch()
