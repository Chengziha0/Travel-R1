import json
import re
import os

def extract_before_parenthesis(s):
    match = re.search(r'^(.*?)\([^)]*\)', s)
    return match.group(1) if match else s