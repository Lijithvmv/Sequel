def extract_nested_dict(response):
    # Dummy implementation: expects response to be a JSON string or dict with 'queries' key
    # In real use, parse and extract as needed
    if isinstance(response, str):
        import json
        response = json.loads(response)
    return response 