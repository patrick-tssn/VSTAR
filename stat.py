import json

test = json.load(open("test.json"))


dias = 0
scene = 0
topic = 0
for dct in test["dialogs"]:
    dias += len(dct["dialog"])
    scene += (dct["scene"][-1] - dct["scene"][0]+1)
    topic += (dct["session"][-1] - dct["session"][0]+1)
    
print(dias)
print(scene / len(test["dialogs"]))
print(topic / len(test["dialogs"]))