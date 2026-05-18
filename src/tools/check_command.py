import json

c_set = set()
data = open('/mnt/nas-data-1/wuchangjie.wcj/data/ad/train_bev-total.jsonl')
li = 0
for line in data:
    li += 1
    if li % 10000 == 0:
        print(li)
    try:
        line = json.loads(line)
    except:
        print(li, 'error',line)
        continue
    command = line["messages"][0]["content"]
    command = int(command)
    c_set.add(command)
print(c_set)