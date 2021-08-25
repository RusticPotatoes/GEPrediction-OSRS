import os, json, requests

#https://documenter.getpostman.com/view/3756775/ge-tracker/RVnTkLh1#intro
parent_dir = os.path.dirname(os.path.realpath(__file__))

token=""
with open(os.path.join(parent_dir,'token.txt'),'r') as file:
    token = file.read()
print (token)
accept = 'application/x.getracker.v1+json'
authorization = f'Bearer {token}'
agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
#headers = f"{accept} {authorization}"
feed= "market-watch"
items= 'items'
headers= {"User-Agent": agent, "Accept": accept, "Authorization": authorization}
url = f"https://www.ge-tracker.com/api/{items}/3105"
print (url, headers)

result = requests.get(url, headers=headers).json()
print(result)