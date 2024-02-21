import requests
import pandas as pd

def fetch_jobs(job_id):
    
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
    # print(job_data)
    job_df = pd.DataFrame(job_data)

    return job_df