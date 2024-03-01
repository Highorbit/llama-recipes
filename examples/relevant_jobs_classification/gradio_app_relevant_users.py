import os
import requests
import pandas as pd
import numpy as np
print('kjsbf;ab')
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto
import torch
import time
import pandas as pd, html
from user_info import get_user_info
import requests
from tqdm.notebook import tqdm
from user_info import get_user_data_search_embed
print('import')
from job_info import fetch_jobs
from tqdm import tqdm
from vllm import SamplingParams, LLM
print('imports done')
sampling_params = SamplingParams(temperature=0, max_tokens=100)
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

dataset = pd.read_csv('/home/ubuntu/infoedge/llama-recipes/examples/relevant_jobs_classification/data/analysis_dataset.csv')
import gradio as gr


def fetch_job_title(job_id):
    
    '''
    Returns Job Data[Hirst, Source : ElasticSearch] for the last_k days, and with flexible page size (<10,000)
    
    Input : Last-K Days, and Page result size
    
    Output : Pandas Dataframe with jobs from the last k days 
    '''
    
    # columns = ['id', 'title','companyStatus', 'min', 'max', 'premium', 'applyCount', 'locations','tags','brandJobFlag']     #Added column name for companyStatus
    # new_cols = ['jobid', 'title','companyStatus', 'min', 'max', 'premium', 'applyCount', 'locations','tags', 'brandJobFlag']       #Added column name for companyStatus
    
    url = "http://internal-java-job-searcher-email-backend-1607736061.ap-south-1.elb.amazonaws.com/v1/user/-777/job/jobcode?query=<JOBID>"
    url = url.replace('<JOBID>',job_id)
    payload={}
    headers = {}
    
    response = requests.request("GET", url, headers=headers, data=payload)
    payload={}
    headers = {}
    
    response = requests.request("GET", url, headers=headers, data=payload)
    job_data = response.json()

    title = job_data[0]['title']
    jd = job_data[0]['introText']
    # jd = html.unescape(jd)

    return title + '\n' + jd

def get_answer_from_vllm(resume_text, job_title, llm):

    #define the input prompt here
    input_prompt = f'''<s>[INST]You are an accurate AI agent working for a job platform. You will be given the raw 
                            unstructured text of a user's resume, and the description of a job that they have applied to. Your task 
                            is to check if the resume contains prior work experience relevant to the job title being provided.
                            Respond with a JUST A SINGLE WORD answer "yes" or "no". If there's any ambiguity, answer with "no"
                            This is the job description {{job_title}}\n
                            This is the resume text:\n{{resume_text}}\n
                            The answer is: [\INST]
                            '''

    input_prompt = input_prompt.format(resume_text=resume_text,
                   job_title=job_title)
    
    outputs = llm.generate(input_prompt, sampling_params)
    generated_text = outputs[0].outputs[0].text

    return generated_text



def make_clickable(link):
    return f'<a href="{link}" target="_blank">{link}</a>'

def split_yes_no_extra_info(text):
    text_lower = text.lower()
    if 'yes' in text_lower:
        return 'yes', text.replace('yes', '').strip()
    elif 'no' in text_lower:
        return 'no', text.replace('no', '').strip()
    else:
        return None, text.strip()

def run_predictions_for_job(job_id):
    job_id = int(job_id)
    responses = {}
    ds = dataset.loc[dataset['job_id']==job_id]
    ex = ds[['job_id','user_id']].to_numpy().tolist()
    # print('ye bhi',ex)
    for jobid, userid in tqdm(ex):
        print(jobid)

        user_id_list = dataset.loc[dataset['job_id']==jobid]['user_id'].tolist()
        try:
            jt = fetch_job_title(str(jobid))
            # print(jt)
        except Exception as e:
            print(f'Error while getting job data for job id {jobid} is {e}\n')
            continue

        if jobid in responses:
            pass
        else:
            responses[jobid] = {}

        try:
            es_output = get_user_info(userid)
            rt = es_output['resume'][0]
            # rt = html.unescape(rt)
        except Exception as e:
            print(f'Error while getting resume text for user_id {userid} is {e}\n')
            continue
        
        full_doc = get_answer_from_vllm(rt,jt, llm)
        # print('ye kra',full_doc)
        responses[jobid][userid] = full_doc

    # return responses

    # print(responses)

        # df = get_user_data_search_embed(ds['user_id'].tolist())
    new_df = pd.DataFrame(responses[int(job_id)],index=[0]).T.reset_index()
    new_df.columns=['id','info']
    df = get_user_data_search_embed(new_df['id'].tolist())
        
    final_usr=pd.DataFrame()

    for x in df['id']:
        hy = "https://search.iimjobs.com/profile/userid"
        mini = df.loc[df['id'] == x].copy()
        mini['user_profile'] = [hy.replace("userid", str(x))]
        final_usr = pd.concat([final_usr, mini], ignore_index=True)

    final_usr.reset_index(drop=True, inplace=True)
    final_usr['user_profile'] = final_usr['user_profile'].apply(lambda x: make_clickable(x))

    final_usr = final_usr[['id','current_designation','user_experience','user_profile']]
    final_usr['id']=final_usr['id'].astype(int)

    fd = pd.merge(final_usr,new_df,on='id')

    fd['real_values'], fd['reason'] = zip(*fd['info'].apply(split_yes_no_extra_info))

    fd['reason'] = fd['reason'].replace('\n', ' ', regex=True)
    fd['reason'] = fd['reason'].replace('\n', ' ', regex=True).str.strip()

    # Remove extra spaces
    fd['reason'] = fd['reason'].replace('\s+', ' ', regex=True)

    fd = fd[['id','current_designation','user_experience','real_values','reason','user_profile']]

    return fd


# iface = gr.Interface(
#     fn=run_predictions_for_job,
#     inputs=[gr.Textbox(label="JOB ID")],
#     outputs=[gr.HTML(label="RESULTS")]
# )
with gr.Blocks() as demo:

    gr.Markdown(

   )

    inp = gr.Number(label="JOB ID")

    out = gr.DataFrame(label='output',wrap=True,
            datatype=["number", "markdown","markdown","markdown", "html","markdown"],
            interactive=True,)

    # inp.change(run_predictions_for_job, inp, out)
    btn = gr.Button("Run")
    btn.click(fn=run_predictions_for_job, inputs=inp, outputs=out)
 

demo.launch('0.0.0.0', share=True)

# iface.launch('0.0.0.0', share=True)











