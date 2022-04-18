import json

with open(r'D:\ComputerScience\BachKhoa\ProjectII\YOLO_QA\data\train-v2.0.json') as f:
    data = json.load(f)

print(data.keys())
print(data['version'])
debug_data =  {
    'version': data['version'],
    'data': [data['data'][0]]
}

with open(r'data\debug.json', 'w') as f:
    json.dump(debug_data, f)