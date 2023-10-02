### Topic: Web Crawler by Twitter API Academic - Scrap by Username
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
    nohup python scrap_by_username_academic.py > stdout.txt 2> stderr.txt &
'''

### Record:
'''
    Twitter Account: @weihungtseng
    Platform: Twitter Developer Protal -> Projects&Apps -> TwitterCrawler -> hashtag_crawler20200428
    API Key: sHc0X98XAkDastG8td2hzE9Xr
    API Key Secret: qiIkNZ0iB4m8i1bY6HkTa0qd5nn8fdfPJu2cWHbPPOaSGFaBoh
    Bearer Token: AAAAAAAAAAAAAAAAAAAAACbubwEAAAAAx%2Bw9JCStzQB72k5fgvIETWJjpXU%3DcgUPpc2MR4qtw19M7n6Ud4pfJ7eioKRzLPwoXlwY9yIblxQ6yX
    Access Token: 1519568124190031873-c1CNsrMy4rKGCUmrQCsQrxjqdsECxm
    Access Token Secret: FR2Sox4FmTvgerd8GOOiyY11wUYeaxFKaHXa0O58BZJeL
    Use tweet id to find tweet:
        https://twitter.com/account/status/id
        ex:
            https://twitter.com/weihungtseng/status/1520884242917597185

    Academic Research Account:
        TWITTER 帳號 lienlabv2
        密碼: lienmhllab

        2022-12-23
        API Key: 8OSh2sEjvtRfkX1jG9ESzaE8J
        API Key Secret: 3Vc2CqGggV1KRlf57GAupc7IjRfKfc7AdzWXST0fnWms0V9Eil
        Bearer Token: AAAAAAAAAAAAAAAAAAAAAArtkgEAAAAA3r83qmzwNoROSROAp%2B4BvDvWzfM%3DcmqJkUAzIL956stGMv8ZclRirCLPF3G6uVMLr3v7dN6kBH0YpT
        {"token_type":"bearer","access_token":"AAAAAAAAAAAAAAAAAAAAAArtkgEAAAAA3r83qmzwNoROSROAp%2B4BvDvWzfM%3DcmqJkUAzIL956stGMv8ZclRirCLPF3G6uVMLr3v7dN6kBH0YpT"}%   
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
## BAD ERR: bearer_token = os.environ.get("AAAAAAAAAAAAAAAAAAAAAArtkgEAAAAA3r83qmzwNoROSROAp%2B4BvDvWzfM%3DcmqJkUAzIL956stGMv8ZclRirCLPF3G6uVMLr3v7dN6kBH0YpT")
bearer_token = "AAAAAAAAAAAAAAAAAAAAAArtkgEAAAAA3r83qmzwNoROSROAp%2B4BvDvWzfM%3DcmqJkUAzIL956stGMv8ZclRirCLPF3G6uVMLr3v7dN6kBH0YpT"
end_date = str(datetime.now()).split()[0]
# print(end_date); exit()
## Academic research search all endpoint URL
search_url = 'https://api.twitter.com/2/tweets/search/all'
## Academic research counts all endpoint URL
counts_url = 'https://api.twitter.com/2/tweets/counts/all'
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
            data = {
                "query": f"{query}",
                # "start_time": "2006-03-21T00:00:00Z",
                # "end_time": f"{end_date}T00:00:00Z",
                ## data20230111
                #    2021/11/1-2022/10/31
                "start_time": "2021-11-01T00:00:00Z",
                "end_time": "2022-11-01T00:00:00Z",
                "max_results": 500,
                # "sort_order": "relevancy", ## Default: recency(end -> start)
                "tweet.fields": "created_at",
                "next_token": f"{next_token}"
            }
        else:
            data = {
                "query": f"{query}",
                # "start_time": "2006-03-21T00:00:00Z",
                # "end_time": f"{end_date}T00:00:00Z",
                "start_time": "2021-11-01T00:00:00Z",
                "end_time": "2022-11-01T00:00:00Z",
                "max_results": 500,
                # "sort_order": "relevancy", ## Default: recency(end -> start)
                "tweet.fields": "created_at"
                # "next_token": f"{next_token}"
            }

    response = requests.request("GET", url, auth=search_bearer_oauth, params=data)
    if response.status_code != 200: raise Exception(response.status_code, response.text)
    return response.json()

def create_url(TwitterNameOrID):
    # Specify the usernames that you want to lookup below
    # You can enter up to 100 comma-separated values.
    usernames = f"usernames={TwitterNameOrID}"
    user_fields = "user.fields=created_at,public_metrics"
    # User fields are adjustable, options include:
    # created_at, description, entities, id, location, name,
    # pinned_tweet_id, profile_image_url, protected,
    # public_metrics, url, username, verified, and withheld
    url = "https://api.twitter.com/2/users/by?{}&{}".format(usernames, user_fields)
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

## Read data
source_path = "../../../data/scrap_by_username/source/data20230111/Valid_username_selection.xlsx"
result_path = "../../../data/scrap_by_username/result/data20230111"
df = pandas.read_excel(
    source_path,
    'Valid_username_seleciton', ## Work sheet
    names = ["username", "account_total_tweet", "account_created_at"],
    usecols = 'A:C'
)
# print(df.head())
# exit()

## Dynamic var
## wh7Ie: 2019-12-09 1043: empty
## dh_park02: 2009-02-23 1043

# progress = tqdm(total=len(df))
for ind, row in df.iterrows():
    TwitterNameOrID = row['username']
    account_created_at = row["account_created_at"],
    account_total_tweet = row["account_total_tweet"],
    print(f'TwitterNameOrID = {TwitterNameOrID} Start')
    search_query = \
        f'from:{TwitterNameOrID} '+\
        'lang:en '+\
        '-is:retweet'

    ## Init var
    search_next_token = None
    all_tweets = []
    recent_total_tweet = 0

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
    print(f'Finish: total_tweet = {recent_total_tweet}')

    ## Tweet to csv
    outtweets = [[
        TwitterNameOrID,
        account_created_at,
        account_total_tweet,
        tweet['text'],
        tweet['created_at'],
        tweet['id']
    ] for idx, tweet in enumerate(all_tweets)]
    df_tmp = pandas.DataFrame(outtweets, columns=["username", "account_created_at", "account_total_tweet", "text", "tweet_created_at", "tweet_id"])
    outtweets = []
    # df = df.sort_values(by=['tweet_id'], ascending=True)
    df_tmp.to_csv(f'{result_path}/{TwitterNameOrID}.csv', index = False)
    print(f'TwitterNameOrID = {TwitterNameOrID} Finish')