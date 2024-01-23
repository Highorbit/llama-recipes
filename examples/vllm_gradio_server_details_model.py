from vllm import LLM, SamplingParams
import numpy as np 
import pandas as pd 
import json 
import gradio as gr
from user_info import get_user_info
import ast, html, re


sampling_params = SamplingParams(temperature=0, max_tokens=4096)
llm = LLM(model="lakshay/work-details")

eval_df = pd.read_csv('custom_data/iimjobs_eval_df.csv')

def convert_to_json(input_string):
    # Replace single quotes at the start and end of keys and values with double quotes
    # This regex specifically targets the start of keys/values and the end of keys/values
    corrected_string = re.sub(r"(\{|\,)\s*\'", r'\1 "', input_string)  # Start of key/value
    corrected_string = re.sub(r"\'\s*(\,|\})", r'" \1', corrected_string)  # End of key/value
    corrected_string = re.sub(r"\'\s*:", r'":', corrected_string)  # Key end
    corrected_string = re.sub(r":\s*\'", r': "', corrected_string)  # Value start

    try:
        # Convert the string to a valid JSON
        valid_json = json.loads(corrected_string)
        print(f'it worked!')
        return valid_json
    except json.JSONDecodeError as e:
        print(f'Error thrown in JSON converter is {e}')
        return input_string

def make_eval_prompt(raw_text):
    
    
    work_prompt = f'''
    You are an accurate agent working for a job platform. You will be given the raw 
    unstructured text of a user's resume, and the task is to extract only the following details about the 
    work experience of the user from the resume in a JSON style format: company name, designation, start date and the end date.
    Please provide the data in a concise and parseable JSON format. Ensure the JSON syntax is correct
    with proper use of double quotes, commas, and braces. Dates should be in "mm/yyyy" format

    This is the resume text:\n{{resume_text}}\n
    This is the output in the required_format:\n
    '''
    
    
    eval_prompt = eval_prompt.format(resume_text=raw_text)

    return eval_prompt

def parse_user_work_ex(info_json):
    work_ex = []
    try:
        for x in info_json['professional_info'][0]:
            
            user_dict = {}
            exp_dict = x
        
            keys = ['id','designation','fromExpMonth','fromExpYear','toExpMonth','toExpYear'] 
        
            for k in keys:
                user_dict[k] = exp_dict[k]
                
            user_dict['company'] = exp_dict['organization']['name']
            work_ex.append(user_dict)
    except:
        return work_ex
            
    return work_ex



def get_response_from_model(user_id):

    es_output = get_user_info(user_id)
    resume_text = es_output['resume'][0]
    work_ex = parse_user_work_ex(es_output)
    if resume_text:
        eval_prompt = make_eval_prompt(resume_text)
    
    outputs = llm.generate(eval_prompt, sampling_params)
    out_text = outputs[0].outputs[0].text
    ot = html.unescape(out_text)
    generated_text = ot
    
    try:
        out_json = ast.literal_eval(generated_text)
        return json.dumps(out_json,indent=4), json.dumps(work_ex,indent=4)

    except Exception as e:
        print(f'couldnt JSONify this {e}')
        gt = convert_to_json(generated_text)
        return json.dumps(gt, indent=4), json.dumps(work_ex,indent=4)


demo = gr.Interface(
    fn=get_response_from_model,
    inputs=[gr.Textbox(label="UserID")],
    outputs=[gr.Textbox(label="LLM Output"),gr.Textbox(label="User Info from ElasticSearch")],
    allow_flagging="manual"
)

demo.launch(share=True)