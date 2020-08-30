import requests
import json
from escapejson import escapejson
import base64

def ocr(file):

  with open("static/uploads/"+file.filename, "rb") as img_file:
      x = base64.b64encode(img_file.read())

  url = "https://taggun.p.rapidapi.com/api/receipt/v1/verbose/encoded"

  payload = {
    "image": x.decode('utf-8'),
    "filename": "example.jpg",
    "contentType": "image/jpeg",
    "refresh": "false",
    "incognito": "false",
    "ipAddress": "32.4.2.223"
  }
  my_str = json.dumps(payload)
  my_safe_str = escapejson(my_str)


  headers = {
      'x-rapidapi-host': "taggun.p.rapidapi.com",
      'x-rapidapi-key': "51381af812msh5673d2aef7aa2a0p129d31jsn06336c9eb22f",
      'content-type': "application/json",
      'accept': "application/json"
      }

  response = requests.request("POST", url, data=my_safe_str, headers=headers)
  x = response.json()
  y = x["text"]["text"]
  # y = "+<br />+".join(y.split("\n"))
  print(y)
  return y