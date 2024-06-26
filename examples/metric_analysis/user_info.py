import pandas as pd
import numpy as np
import requests
import re
from tqdm import tqdm

def clean_space(text):
    return " ".join(re.split("\s+", text, flags=re.UNICODE))

def remove_extra_spaces(input_string):
    # Use regular expression to remove extra spaces (only spaces)
    cleaned_string = re.sub(r' +', ' ', input_string)
    return cleaned_string

def get_user_info(user_id):
    
    '''
    This function combines all the functions for fetching and processing 
    user data from elastic search
    '''
    uidl= [user_id]

    sub_json = fetch_data_es(uidl)
    raw_data = construct_user_data_search_embed(sub_json)
    
    return raw_data

def get_user_data_search_embed(user_id_list):
    
    '''
    This function combines all the functions for fetching and processing 
    user data from elastic search
    '''


    sub_json = fetch_data_es(user_id_list)
    raw_data = construct_user_data_search_embed(sub_json)

    res_df = pd.DataFrame.from_dict(raw_data, orient='index').T
    result_df = res_df.astype(str).replace("None", " ")
    
    result_df['resume'] = result_df['resume'].apply(remove_extra_spaces)

    return result_df

def fetch_data_es(user_id_list):
    
    """
    Input : List of Valid Hirist User ID 
    Output : List of User Data JSON stored in Elastic Search
    """
    
    
    response_json = []
    replacements = {'null':'None', 'false':'False'}

    url = "http://10.208.230.226:8080/v1/recruiter/-123/applicant/search/searchById"

    res = []

    for uid in tqdm(user_id_list):


        payload = {
            'usersToSearchFrom' : [str(uid)]
        }

        headers = {
        'Content-Type': 'application/json'
        }

        try:
            response = requests.request("POST", url, headers=headers, data = str(payload))
            response_json = response.json()
    
            res.append(response_json['docs']) 

        except Exception as e:
            continue
        
    return res



def construct_user_data_search_embed(user_json):
    '''
    This is a function for fetching user ID, Keywords (in the form of tags) and 
    user designation from elastic search. This is specifically made for creating 
    user embeddings for our Recruiter search model/POC
    '''
    
    user_dict = {
        'id' : [],
        'resume' : [],
        'keywords': [],
       'current_designation' : [],
        'user_experience':[],
        'professional_info':[],
        'education_info':[]    
    
    }

    for s in tqdm(user_json):

        try:
            source = s[0]

            user_dict['id'].append(source['id'])

            skill_list = [str(skill['name']).replace(" ","_") for skill in source['skillSet']]
            user_dict['keywords'].append(" ".join(skill_list))
            
            user_dict['user_experience'].append(source['expYear'])
            
            user_dict['resume'].append(source['resumeText'])
            user_dict['current_designation'].append(source['currentDesignation'])
            user_dict['professional_info'].append(source['professionalInfo'])
            user_dict['education_info'].append(source['educationInfo'])  

        except Exception as e:
            continue

    return user_dict