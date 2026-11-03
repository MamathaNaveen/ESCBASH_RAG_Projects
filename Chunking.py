import json
with open("./handbook.txt","r") as f:
    data=f.read()

listd=[p.strip() for p in data.split("\n\n") if p.strip()] 

json.dump(listd,open("./chunks.json","w"))
    
