import json


def read_json_file(file_path):
    """
    :param file_path: file to read
    :return: loaded file
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_values():
    """
    :return: list with all texts & references in data_no_prefixes.json
    """

    # set path
    json_file_path = 'samples/data_prefixes.json'

    # Read the JSON file
    json_data = read_json_file(json_file_path)

    # init list
    value_list = []

    # Accessing the entries in the JSON data & add to list
    for entry in json_data:
        text = entry['text']
        reference = entry['reference']
        value_list.append((text, reference))

    # return list
    return value_list
