
import json


def extract_json_objects(text, decoder = json.JSONDecoder()):
    pos = 0
    while True:
        match = text.find('{', pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            yield result
            pos = match + index
        except ValueError:
            pos = match + 1


def to_json(text_arr, file_name):
    values = []

    for i in range(len(text_arr)):
        for result in extract_json_objects(text_arr[i]):
            print(result)
            values.append(result)
    

    with open(file_name.format(file_name=file_name), 'w', encoding = 'utf-8') as f:
        json.dump(values, f, ensure_ascii=False, indent=4)
    

