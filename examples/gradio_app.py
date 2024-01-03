import gradio as gr
import os


def process_resume(resume_text):
    # Replace this with your actual resume processing logic
    processed_text = f"Received resume:\n\n{resume_text}"
    return "oh yesss"


with gr.Blocks() as demo:

    txt = gr.Textbox(label="Input")
    txt_3 = gr.Textbox(value="", label="Output")
    btn = gr.Button(value="Submit")
    btn.click(process_resume, inputs=[txt], outputs=[txt_3])


if __name__ == "__main__":
    demo.launch(share=True)









