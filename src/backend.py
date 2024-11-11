import os
import torch
import huggingface_hub
import pandas as pd
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

class LoadAndUseModel:
    def __init__(self, model_path, log_filepath):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_filepath,
            filemode='w'
        )
        logging.info("Loading model and tokenizer from %s", model_path)
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info("Model loaded successfully")
    
    def generate_response(self, prompt, max_new_tokens, temperature, top_p, repetition_penalty, top_k, question):
        ap = prompt.format(question, "")
        inputs = self.tokenizer([ap], return_tensors='pt').to('cuda')
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, top_k=top_k)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    def process_batch_questions(self, questions, max_new_tokens, temperature, top_p, repetition_penalty, top_k, batch_size):
        pass

    def use_csv_to_generate_responses(self, csv_path, max_new_tokens, temperature, top_p, repetition_penalty, top_k, retry_count, question):
        results = []
        questions_df = pd.read_csv(csv_path)
        for index, row in questions_df.iterrows():
            for ri in range(retry_count):
                question = row['Instruction']
                
                start_time = datetime.now()
                response_bangla = self.generate_response(question, max_new_tokens, temperature, top_p, repetition_penalty, top_k)

                end_time = datetime.now()
                gen_duration = end_time - start_time
                
                q_id = f"{index}.{ri}"
                result = {
                    'id': q_id,
                    'question': question,
                    self.model_name: response_bangla,
                    'gen_duration': self.format_timedelta(gen_duration),
                }
                results.append(result)

                logging.info(f"Generating response..")
                logging.info("\n\n\n----------------------------------")
                logging.info(result)
                logging.info("----------------------------------\n\n\n")
        
    
    def format_timedelta(self, td):
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    def save_results_to_csv(self, results, output_path):
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        logging.info(f"Responses saved to {output_path}")

