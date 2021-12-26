# importing the requests library
import requests

def linenotify(msg = "hello",file=None):  
    # Config
    URL = "https://notify-api.line.me/api/notify"
    LINE_TOKEN = "XXXXXXXXXXXXXXXXXXXXXXXXXX"
    AUTHOR_TYPE = "Bearer"
    AUTHORIZATION = AUTHOR_TYPE + " " + LINE_TOKEN 
   
    # Data to send
    FILES = None
    if file != None:
        FILES = {'imageFile': file}
    SENDDATA = {'message': msg}
 
    # sending get request and saving the response as response object
    r = requests.post(url = URL, data = SENDDATA,files=FILES,headers={'Authorization': AUTHORIZATION})
    # extracting data in json format
    data = r.json()
    if data["message"] == "ok":
        print("notify success")
    else:
        print(data)

# linenotify(msg="hello", file=open("./image.png",'rb'))