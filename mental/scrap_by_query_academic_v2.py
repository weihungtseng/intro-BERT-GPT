### Topic: Web Crawler by Twitter API Academic V2 - Scrap by Query
### Tag: [Twitter] [Crawler] [API]
### Author: Wei-Hung, Tseng
### CreateDate: 2022/12/15
### Describe: 
'''
    we will download target tweets with the 3 main fields: "user_id, date, tweet_content". 
    These 3 main fields will be processed into 9 fields: "user_id, period_1, period_2, period_3, 
    M, R, S, HE, HA", where period_i (i=1, 2, 3) is from date, and M, R, S, HE, HA is our 
    mental status categories, predicted from tweet_content using some deep learning methods. 
    With these 9 fields, we will then run SPSS to see the significance of the status transition 
    among those 3 periods.
'''

### Install: none

### Execute:
'''
    nohup python scrap_by_query_academic_v2.py > stdout.txt 2> stderr.txt &
'''

### Record:
'''
   
'''

### Refernce:
'''
    Access levels and versions
        https://developer.twitter.com/en/docs/twitter-api/getting-started/about-twitter-api#v2-access-level

    Academic Research search all parameter:
        https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all
    
    Academic Research Full-Archive Search Code Example:
        https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Full-Archive-Search/full-archive-search.py

    Search all cURL example
        https://developer.twitter.com/apitools/api?endpoint=%2F2%2Ftweets%2Fsearch%2Fall&method=get

    About 'next' token and pagination
        https://developer.twitter.com/en/docs/twitter-api/premium/search-api/api-reference/premium-search#Pagination

    User look up parameter:
        https://developer.twitter.com/en/docs/twitter-api/users/lookup/api-reference/get-users

    User look up code example:
        https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/User-Lookup/get_users_with_bearer_token.py

    Academic Research counts all syntax:
        https://developer.twitter.com/en/docs/twitter-api/tweets/counts/api-reference/get-tweets-counts-all

    Academic Research Full-Archive Counts Code Example:
        https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Full-Archive-Tweet-Counts/full_archive_tweet_counts.py

    Twitter-API-v2-sample-code/Full-Archive-Search/full-archive-search.py
        https://github.com/twitterdev/Twitter-API-v2-sample-code/blob/main/Full-Archive-Search/full-archive-search.py

    Premium search APIs - {Data request parameters}
        https://developer.twitter.com/en/docs/twitter-api/premium/search-api/api-reference/premium-search

    Search Tweet: GET /2/tweets/search/all: Academic Research access. - {Query parameters} {Response fields}
        https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all

    Rate limits - {Search Tweets}
        https://developer.twitter.com/en/docs/twitter-api/rate-limits
'''

import pandas
import os, sys, time, re
from datetime import datetime
import requests
from pprint import pprint
import json
from tqdm import tqdm

### Global variable
totalTime = time.time()
queryDict = {
    "M12": '("stress out" OR stress OR pressure ' + 
           'OR "under pressure" OR depressed OR depression OR anxious ' + 
           'OR anxiety OR pressureOR feeling OR emotional OR upset ' + 
           'OR mad OR angry OR sad OR failed OR frustrated ' + 
           'OR concentrate)  ("feel better" OR "reward myself" OR relax ' + 
           'OR "I know how" OR "focus on" OR "how to focus" ' + 
           'OR " take a break")',
    "M13": '("stress out" OR stress OR pressure OR "under pressure" ' + 
           'OR depressed OR depression OR anxious OR anxiety OR pressure ' + 
           'OR feeling OR emotional OR upset OR mad OR angry ' + 
           'OR sad OR failed OR frustrated OR concentrate) ("kill myself" ' + 
           'OR "self harm" OR "can\'t handle" OR "white part" OR barcodes ' + 
           'OR die OR cry OR "mess up" OR overwhelm OR "hard to focus" ' + 
           'OR "can\'t control" OR "out of control" OR "escape from")',
    "R12": '("excessive worry" OR "emotionally drained" OR depressed ' + 
           'OR "low mood" OR anxious OR "Drug addiction" OR hallucinations ' + 
           'OR delusions OR "talking to oneself" OR "lack of motivation") ' + 
           '("mental illness" OR disorder OR disease OR symptoms OR sick ' + 
           'OR schizophrenia OR diagnosis OR anxiety OR GAD ' + 
           'OR "generalized anxiety disorder" OR "Anxiety disorder" ' + 
           'OR depression OR "mental health" OR "mental disease" OR psychosis)',
    "R123": '("excessive worry" OR "emotionally drained" OR depressed ' + 
            'OR "low mood" OR anxious OR "Drug addiction" OR hallucinations ' + 
            'OR delusions OR "talking to oneself" OR "lack of motivation") ' + 
            '("mental illness" OR disorder OR disease OR symptoms OR sick ' + 
            'OR schizophrenia OR diagnosis OR anxiety OR GAD ' + 
            'OR "generalized anxiety disorder" OR "Anxiety disorder" ' + 
            'OR depression OR "mental health" OR "mental disease" OR psychosis) ' + 
            '(not)',
    "S12": '("mental illness" OR schizophrenia OR "eating disorder" OR depression ' + 
           'OR  "panic attacks" OR anxiety OR psychosis OR psychopath ' + 
           'OR "mentally ill") (support OR help OR understand OR embrace)',
    "S13": '("mental illness" OR schizophrenia OR "eating disorder" OR depression ' + 
           'OR  "panic attacks" OR anxiety OR psychosis OR psychopath ' + 
           'OR "mentally ill") (dangerous OR shameful OR "ashamed of" OR scary ' + 
           'OR "self harm" OR unpredictable OR frightening OR stereoptypes)',
    "HE12": '("mental illness" OR schizophrenia OR "eating disorder" OR depression ' + 
            'OR  "panic attacks" OR anxiety OR psychosis OR psychopath ' + 
            'OR "mentally ill" OR "mental health") ("guidance counselor" ' + 
            'OR therapist OR "promote mental health" OR "mental hospital" ' + 
            'OR  therapy OR psychiatry OR psychiatric OR "psychiatric service")',
    "HE13": '("mental illness" OR schizophrenia OR "eating disorder" OR depression ' + 
            'OR  "panic attacks" OR anxiety OR psychosis OR psychopath ' + 
            'OR "mentally ill" OR "mental health") ("get help" OR "deal with" ' + 
            'OR information)',
    "HA12": '(depress OR anxiety OR panic OR "can\'t sleep" OR "mental problem" ' + 
            'OR "mental health" OR "mental illness" OR upset OR unhappy OR sad ' + 
            'OR anxious OR "self harm") ("in therapy" OR "self-help" OR "get help" ' + 
            'OR "mental hospital" OR psychiatris OR therapist "seek help")',
    "HA13": '(depress OR anxiety OR panic OR "can\'t sleep" OR "mental problem" ' + 
            'OR "mental health" OR "mental illness" OR upset OR unhappy OR sad ' + 
            'OR anxious OR "self harm") ("no therapy" OR "no meds" ' + 
            'OR "Instead of therapy")'
}
bearer_token = ""

## data20221227
# start_date = '2022-11-01T00:00:00Z'
# end_date = '2022-12-11T00:00:00Z'

## data20230104_1: 12/11 00:00 - 12/12 21:37
# start_date = '2022-12-11T00:00:00Z'
# end_date   = '2022-12-12T21:37:00Z'

## data20230104_2: 2022/12/22 09:22 - 2022/12/31 23:59
# start_date = '2022-12-22T09:22:00Z'
# end_date   = '2022-12-31T23:59:00Z'

## data20230104
data_date = 'data20230104'
start_date = '2022-11-01T00:00:00Z'
end_date   = '2023-01-01T00:00:00Z'

num_per_req = 500
# end_date = str(datetime.now()).split()[0]
# print(end_date); exit()
## Academic research search all endpoint URL
search_url = 'https://api.twitter.com/2/tweets/search/all'
request_cnt = 0

## Query example
'''
    (happy OR happiness OR excited OR elated) 
    lang:en 
    -birthday 
    -is:retweet 
    -holidays 

    from:twitterdev OR from:twitterapi 
    -from:twitter
'''

def search_bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r

def search_connect_to_endpoint(url, query, next_token=None):
    if url == search_url:
        if next_token != None:
            ## username,create_at(account),totaltweets,text,date,tweetid
            data = {
                "query": f"{query}",
                "start_time": f"{start_date}",
                "end_time": f"{end_date}",
                "max_results": num_per_req,
                # "sort_order": "relevancy", ## Default: recency(end -> start)
                "tweet.fields": "author_id,created_at",
                "next_token": f"{next_token}"
            }
        else:
            data = {
                "query": f"{query}",
                "start_time": f"{start_date}",
                "end_time": f"{end_date}",
                "max_results": num_per_req,
                # "sort_order": "relevancy", ## Default: recency(end -> start)
                "tweet.fields": "author_id,created_at",
                # "next_token": f"{next_token}"
            }

    response = requests.request("GET", url, auth=search_bearer_oauth, params=data)
    if response.status_code != 200: raise Exception(response.status_code, response.text)
    return response.json()

def create_url(TwitterNameOrID):
    # Specify the usernames that you want to lookup below
    # You can enter up to 100 comma-separated values.
    ids = f"ids={TwitterNameOrID}"
    user_fields = "user.fields=created_at,public_metrics"
    # User fields are adjustable, options include:
    # created_at, description, entities, id, location, name,
    # pinned_tweet_id, profile_image_url, protected,
    # public_metrics, url, username, verified, and withheld
    url = "https://api.twitter.com/2/users?{}&{}".format(ids, user_fields)
    return url

def user_bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2UserLookupPython"
    return r

def user_connect_to_endpoint(url):
    response = requests.request("GET", url, auth=user_bearer_oauth,)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

for key in queryDict:
    ## Init var
    search_next_token = None
    all_tweets = []
    recent_total_tweet = 0
    data_dict = {}
    cnt100 = 0
    TwitterNameOrID100_list = []
    TwitterNameOrID_str = ''
    lst = []

    print(f'{key} start')
    ## Dynamic var
    loopTime = time.time()
    query = queryDict[key]
    search_query = \
        f'{query} '+\
        'lang:en '+\
        '-is:retweet'
    # print(query); exit()

    ### Search
    while True:
        ## Academic research: 100 requests per 15 minutes
        if request_cnt == 100: print('sleeping'); time.sleep(900); request_cnt = 0

        search_res_json = search_connect_to_endpoint(search_url, search_query, search_next_token)
        ## <class 'dict'>
        request_cnt += 1

        # search_res_str = json.dumps(search_res_json, indent = 4)
        ## <class 'str'> = json.dumps(<class 'dict'>)
        # with open('academic_search.json', 'a') as f: f.write(search_res_str)
        # exit()
        
        ## meta: This object contains information about the number of users returned in the current request and pagination details.
        result_count = search_res_json['meta']['result_count']
        if result_count == 0: print('break by result_count = 0'); break
        recent_total_tweet += result_count
        print(f'Recent recent_total_tweet = {recent_total_tweet}')

        ## Store data
        all_tweets.extend(search_res_json['data'])

        ## Get next_token
        if 'next_token' in search_res_json['meta']:
            search_next_token = search_res_json['meta']['next_token']
        else:
            break
    print(f'{key} total_tweet = {recent_total_tweet}')

    ## Tweet to csv
    ## username,create_at(account),totaltweets,text,date,tweetid
    outtweets = [[
        tweet['author_id'],
        '',
        '',
        tweet['text'],
        tweet['created_at'],
        tweet['id']
    ] for idx, tweet in enumerate(all_tweets)]
    df = pandas.DataFrame(outtweets, columns=["author_id", "account_created_at", "account_total_tweet", "text", "tweet_created_at", "tweet_id"])
    # df = df.sort_values(by=['author_id'], ascending=True)
    df.to_csv(f'./{data_date}/{key}_incomplete.csv', index = False)
    outtweets = []
    
    ### User look up
    ## Add username, account_created_at, account_total_tweet
    df = pandas.read_csv(f"./{data_date}/{key}_incomplete.csv", lineterminator='\n')
    
    ## Make query string
    progress = tqdm(total=len(df))
    for ind, row in df.iterrows():
        author_id_str = str(row['author_id'])
        if author_id_str not in data_dict:
            data_dict[author_id_str] = {'username': '', 'account_created_at': '', 'account_total_tweet': ''}
            TwitterNameOrID_str += author_id_str
            cnt100 += 1
            if cnt100 != 100:
                TwitterNameOrID_str += ','
            else:
                if TwitterNameOrID_str != '':
                    if TwitterNameOrID_str[-1] == ',': TwitterNameOrID_str = TwitterNameOrID_str[:-1]
                    TwitterNameOrID100_list.append(TwitterNameOrID_str)
                    TwitterNameOrID_str = ''
                    cnt100 = 0
        progress.update(1)

    if TwitterNameOrID_str != '':
        if TwitterNameOrID_str[-1] == ',': TwitterNameOrID_str = TwitterNameOrID_str[:-1]
        TwitterNameOrID100_list.append(TwitterNameOrID_str)

    ## Get user data
    print(f'total num of req = {len(TwitterNameOrID100_list)}')
    user_req_cnt = 0
    for TwitterNameOrID in TwitterNameOrID100_list:
        ## Academic research: 100 requests per 15 minutes
        if request_cnt == 100: print('sleeping'); time.sleep(900); request_cnt = 0

        try:
            user_res_json = user_connect_to_endpoint(create_url(TwitterNameOrID))
        except Exception as e: ## if wrong
            print(e)
        # finally: ## what ever wrong or right
        request_cnt += 1; user_req_cnt += 1
        print(f'user_req_cnt = {user_req_cnt}')

        # user_res_str = json.dumps(user_res_json, indent = 4)
        ## <class 'str'> = json.dumps(<class 'dict'>)
        # with open('academic_search.json', 'a') as f: f.write(user_res_str)

        ## user_res_json
        '''{'data': [{'created_at': '2009-02-23T03:27:16.000Z',
                'id': '21627347',
                'name': 'Dionysus 10.69.3.1',
                'public_metrics': {'followers_count': 52,
                                    'following_count': 90,
                                    'listed_count': 0,
                                    'tweet_count': 1052},
                'username': 'dh_park02'}]}'''

        for idx, data in enumerate(user_res_json['data']):
            author_id = data['id']
            data_dict[author_id]['username'] = user_res_json['data'][idx]['username']
            data_dict[author_id]['account_created_at'] = user_res_json['data'][idx]['created_at'][:10]
            data_dict[author_id]['account_total_tweet'] = user_res_json['data'][idx]['public_metrics']['tweet_count']
        
    progress = tqdm(total=len(df))
    for ind, row in df.iterrows():
        author_id_str = str(row['author_id'])
        if data_dict[author_id_str]['username'] == '': continue
        # new_df.loc[len(new_df)] = [data_dict[author_id_str]['username'], data_dict[author_id_str]['account_created_at'], int(data_dict[author_id_str]['account_total_tweet']), row['text'], row['tweet_created_at'], row['tweet_id']]
        lst.append([data_dict[author_id_str]['username'], data_dict[author_id_str]['account_created_at'], int(data_dict[author_id_str]['account_total_tweet']), row['text'], row['tweet_created_at'], row['tweet_id']])
        progress.update(10)

    # new_df = new_df.sort_values(by=['account_total_tweet'], ascending=False)
    df = pandas.DataFrame(lst, columns=["username", "account_created_at", "account_total_tweet", "text", "tweet_created_at", "tweet_id"])
    df.to_csv(f'./{data_date}/{key}.csv', index = False)
    print(f'{key} time = {(time.time()-loopTime):.2f}')
    '''
    M13 time = 28779.82 | M13 total_tweet = 249652
    '''
    # exit()
print('\n\n' + 'Total time: %1.2f'%(time.time()-totalTime) + '\n')