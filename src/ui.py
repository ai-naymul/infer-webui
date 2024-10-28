import gradio as gr
from backend import LoadAndUseModel

def generate_response(model_path, questions_csv_path, output_csv_path, log_filepath, max_new_tokens, temperature, top_p, repetition_penalty, top_k, retry_count, mode, question):
    model_handler = LoadAndUseModel(model_path, log_filepath)
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

home = gr.Interface(title="Infer WebUI",
                    fn=generate_response,
                    inputs=[
                        gr.Textbox(placeholder="Enter the model name...", label="Model Name"),
                        gr.Textbox(placeholder="Enter the input CSV file path...", label="Input CSV File Path"),
                        gr.Textbox(placeholder="Enter the output CSV file path...", label="Output CSV File Path"),
                        gr.Textbox(placeholder="Enter the log file path...", label="Log File Path"),
                        gr.Number(value=100, label="Enter the max new tokens..."),
                        gr.Number(value=0.7, label="Enter the temperature..."),
                        gr.Number(value=0.95, label="Enter the top p..."),
                        gr.Number(value=1.15, label="Enter the repetition penalty..."),
                        gr.Number(value=50, label="Enter the top k..."),
                        gr.Number(value=3, label="Enter the retry count..."),
                        gr.Dropdown(choices=["usecsv", "singlequestion"], label="Select the mode"),
                        gr.TextArea(placeholder="Enter the question/instruction here...", label="Question/Instruction")
                    ],
                    outputs="text",
                    description="""This project aims to develop a user-friendly web interface using Gradio for performing dynamic inference with large language models (LLMs) hosted on Hugging Face. The interface will allow users to load any open-source model from Hugging Face, input prompts or instructions, and receive generated responses. Key features include the ability to authenticate with a Hugging Face token, process batch questions from CSV files, and adjust model parameters such as token count and temperature. The project will also support logging, response history, and multi-language capabilities, making it a versatile tool for both individual and collaborative use cases.""",
                    theme='earneleh/paris')

if __name__ == "__main__":
    home.launch()
