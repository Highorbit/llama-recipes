import gradio as gr
import html
import ast
from user_info import get_user_info
import requests

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

def http_bot(prompt):
    headers = {"User-Agent": "vLLM Client"}
    pload = {
        "prompt": prompt,
        "stream": False,
        "max_tokens": 2048,
    }
    response = requests.post('http://localhost:5000/generate',
                             headers=headers,
                             json=pload,
                             stream=False)
    
    out_text = response.json()['text']

    
    generated_string = out_text[0].split('This is the output in the required format:')[1]
    go = html.unescape(generated_string)

    generated_output = html.unescape(go)
    generated_output = generated_output.replace('\n','')
    generated_output = generated_output.strip()
    
    try:
        out_json = ast.literal_eval(generated_output)
        return out_json,2
    except:
        print('couldnt JSONify generated Text (ast.literal_eval)')
        return generated_output, 1
 
    cs = []
    # for chunk in response.iter_lines(chunk_size=8192,decode_unicode=False,delimiter=b"\0"):
    #     if chunk:
    #         data = json.loads(chunk.decode("utf-8"))
    #         cs.append(data['text'][0].replace(eval_prompt,''))

    # json_str = ''.join(cs)
    # return json_str


def get_response_from_model(user_id):

    es_output = get_user_info(user_id)
    resume_text = es_output['resume'][0]
    if resume_text:
        eval_prompt = make_eval_prompt(resume_text)
    
    response, error_code = http_bot(eval_prompt)
    try:
        # return json.dumps(response,indent=4)
        return pd.DataFrame(response)
    except:
        print('couldnt make a dataframe')
        return response

demo = gr.Interface(
    fn=get_response_from_model,
    inputs=["text"],
    outputs=["dataframe"],
)

demo.launch(share=True)