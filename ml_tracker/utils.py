import json

def write_json(path, info):
    with open(path, "w") as f:
        json.dump(info, f, indent=4)
    return True
