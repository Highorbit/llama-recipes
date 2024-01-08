#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vllm import LLM, SamplingParams
import numpy as np 
import pandas as pd 
import json 
import ast

sampling_params = SamplingParams(temperature=0, max_tokens=4096)
llm = LLM(model="lakshay/work-model")

eval_df = pd.read_csv('custom_data/iimjobs_eval_df.csv')


def make_eval_prompt(raw_text):
    
    work_format = '''{
        'work_experience': [{'company': 'company Name 1',
                             'role': 'job designation 1',
                             'start_date': 'mm/yyyy',
                             'end_date': 'mm/yyyy',
                             'description': 'complete Job description taken from resume'},
                            {'company': 'company name 2',
                             'role': 'job designation 2',
                             'start_date': mm/yyyy',
                             'end_date': 'mm/yyyy',
                             'description': 'complete Job description taken from resume'}]
    }'''
    
    eval_prompt = f'''
    You are a helpful language model working for a job platform. You will be given the raw 
     unstructured text of a user's resume, and the task is to extract the work experience of the 
     user from the raw text in the following format: \n{{work_format}}\n
    
     This is the resume text:\n{{resume_text}}\n
     This is the output in the required format:\n
    '''
    
    
    eval_prompt = eval_prompt.format(work_format=work_format,
                                     resume_text=raw_text)

    return eval_prompt

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
        return valid_json
    except json.JSONDecodeError as e:
        return input_string


def get_response_from_model(user_id):

    es_output = get_user_info(user_id)
    resume_text = es_output['resume'][0]
    if resume_text:
        eval_prompt = make_eval_prompt(resume_text)
    
    outputs = llm.generate(eval_prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    try:
        out_json = ast.literal_eval(generated_text)
        return json.dumps(out_json,indent=4)

    except Exception as e:
        print(f'couldnt make a dataframe {e}')
        return generated_text


demo = gr.Interface(
    fn=get_response_from_model,
    inputs=["text"],
    outputs=["text"],
)

demo.launch('0.0.0.0',share=True)




