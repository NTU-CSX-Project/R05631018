import requests, json

def chatbot(sent):
    req_json = {'sent': sent}
    #print(req_json['sent'])
    enc = json.JSONEncoder()
    req_json = enc.encode(req_json)
    r = requests.post(url='http://140.112.94.35:8185/sent', json=req_json)
    dec = json.JSONDecoder()
    resp = dec.decode(r.text)
    return resp['resp'].replace('<end>', '')