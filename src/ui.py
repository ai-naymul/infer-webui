import gradio as gr
from backend import LoadAndUseModel
from download_model import DownloadModel
import logging
import os




def generate_response(model_path, questions_csv_path, output_csv_path, log_filepath, max_new_tokens, temperature, top_p, repetition_penalty, top_k, retry_count, mode, question):
    updated_model_path = f"./models/{model_path}"
    model_handler = LoadAndUseModel(updated_model_path, log_filepath)
    if mode == "singlequestion":
        response = model_handler.generate_response(
            prompt="Your prompt template here",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            question=question
        )
        return response
    elif mode == "usecsv":
        response = model_handler.use_csv_to_generate_responses(
            csv_path=questions_csv_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            retry_count=retry_count,
            question=question
        )
        model_handler.save_results_to_csv(response, output_csv_path)
        return "Responses saved to " + output_csv_path
    else:
        return "Batch processing mode is not implemented yet."

def download_model(model_name, model_path, hugging_face_token):
    download_model = DownloadModel(model_name, model_path, hugging_face_token)
    if download_model:
        logging.info("Model Download Successful")
    return download_model.download_model(destination_dir=f"{model_path}/{model_name}")



homepage = gr.Interface(title="Download Model",
                    fn=download_model,
                    inputs=[
                        gr.Textbox(placeholder="Enter the model name...", label="Model Name"),
                        gr.Textbox(placeholder="Enter the path to save the model...", label="Model Path", value="./models"),
                        gr.Textbox(placeholder="Enter the Hugging Face token...", label="Hugging Face Token")
                    ],
                    outputs=gr.Textbox(label="Logs", interactive=True))


generate_interface = gr.Interface(title="Infer WebUI",
                    fn=generate_response,
                    inputs=[
                        gr.Dropdown(label="Load model", interactive=True, choices=[i for i in os.listdir("./models")]),
                        gr.Textbox(placeholder="Enter the input CSV file path...", label="Input CSV File Path", interactive=True),
                        gr.Textbox(placeholder="Enter the output CSV file path...", label="Output CSV File Path", interactive=True),
                        gr.Textbox(placeholder="Enter the log file path...", label="Log File Path", interactive=True),
                        gr.Number(value=100, label="Enter the max new tokens...", interactive=True),
                        gr.Number(value=0.7, label="Enter the temperature...", interactive=True),
                        gr.Number(value=0.95, label="Enter the top p...", interactive=True),
                        gr.Number(value=1.15, label="Enter the repetition penalty..."),
                        gr.Number(value=50, label="Enter the top k..."),
                        gr.Number(value=3, label="Enter the retry count..."),
                        gr.Dropdown(choices=["usecsv", "singlequestion"], label="Select the mode"),
                        gr.TextArea(placeholder="Enter the question/instruction here...", label="Question/Instruction")
                    ],
                    outputs="text",
                    description="""This project aims to develop a user-friendly web interface using Gradio for performing dynamic inference with large language models (LLMs) hosted on Hugging Face. The interface will allow users to load any open-source model from Hugging Face, input prompts or instructions, and receive generated responses. Key features include the ability to authenticate with a Hugging Face token, process batch questions from CSV files, and adjust model parameters such as token count and temperature. The project will also support logging, response history, and multi-language capabilities, making it a versatile tool for both individual and collaborative use cases.""",
                    theme='earneleh/paris')


tabbed_interface = gr.TabbedInterface([homepage, generate_interface], ["Home", "Generate"])

if __name__ == "__main__":
    tabbed_interface.launch()
