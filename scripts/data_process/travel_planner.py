# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the nq dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['query']
    
    # TODO: add a new template type for travel planning
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Plan travel itineraries based on user requests. You must use deep reasoning to create the final plan and show your thinking process between <think> and </think> tags.

When you need travel information you don't have, use tools to search for it. Write tool calls like this:\
<tool>function_name(parameter1=value1, parameter2=value2)</tool>

You can use memory operations at any time. Write them like this:
<memory>function_name(parameter1=value1, parameter2=value2)</memory>

If you have enough information to make the travel plan, give the answer in JSON format between <answer> and </answer> without extra explanation.

Example workflow:
1. User asks for a trip plan
2. You think about what's needed (<think>...</think>)
    a. In thinking process, check memory if needed (<memory>read(...)</memory>)
    b. In thinking process, search for missing info using tools (<tool>search_flights(...)</tool>)
    ...
3. Create and return the final plan (<answer>{{...}}</answer>)

Here are the tools you can use:

1. search_flights(origin_city=None, dest_city=None, date=None)
- Description: Retrieves flight information between two cities.
- Parameters:
  - origin_city: Departure city (e.g., "New York")
  - dest_city: Destination city (e.g., "London")
  - date: Travel date in YYYY-MM-DD format (e.g., "2022-10-01")
- Example: search_flights("New York", "London", "2022-10-01")

2. calculate_distance(origin=None, destination=None, mode="driving")
- Description: Estimates travel distance, time and cost between two locations.
- Parameters:
  - origin: Starting city (e.g., "Paris")
  - destination: Target city (e.g., "Lyon")
  - mode: Transportation method - "driving", "taxi", "walking" or "transit"
- Example: calculate_distance("Paris", "Lyon", mode="driving")

3. search_accommodations(city=None)
- Description: Finds available hotels and lodging in a city.
- Parameters:
  - city: City name (e.g., "Rome")
- Example: search_accommodations("Rome")

4. search_restaurants(city=None)
- Description: Finds dining options in a city.
- Parameters:
  - city: City name (e.g., "Tokyo")
- Example: search_restaurants("Tokyo")

5. search_attractions(city=None)
- Description: Finds tourist attractions in a city.
- Parameters:
  - city: City name (e.g., "London")
- Example: search_attractions("London")

6. search_cities(state=None)
- Description: Lists cities within a specified state/region.
- Parameters:
  - state: State/region name (e.g., "California")
- Example: search_cities("California")

Here are the memory operations you can use:

1. write(key, value)
- Description: Stores data in memory with a specified key
- Parameters:
  - key: Unique identifier for the data (string)
  - value: Data to be stored (any type)
- Example: write("user_preferences", {{"theme": "dark", "language": "en"}})

2. read(key)
- Description: Retrieves data from memory using its key
- Parameters:
  - key: Identifier of the data to retrieve (string)
- Returns: The stored value or None if not found
- Example: read("user_preferences")

3. delete(key)
- Description: Removes data from memory
- Parameters:
  - key: Identifier of the data to remove (string)
- Returns: True if deleted, False if key didn't exist
- Example: delete("temp_data")

4. list_keys()
- Description: Lists all keys currently stored in memory
- Returns: List of all keys (strings)
- Example: list_keys()

5. list_all()
- Description: Provides a summary of all key-value pairs in memory
- Returns: Dictionary with keys and value types/summaries
- Example: list_all()

6. reset()
- Description: Completely clears all data from memory
- Returns: True when complete
- Example: reset()

User instruction: {question}"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'TravelPlanner'

    
    # dataset = datasets.load_dataset(args.local_dir, 'train')
    # print(dataset.keys())
    train_dataset = datasets.load_dataset(args.local_dir, 'train')['train']
    test_dataset = datasets.load_dataset(args.local_dir, 'test')['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # 处理query并保存原始query
            original_query = example['query']
            processed_query = original_query.strip()
            if processed_query[-1] == '.':
                processed_query = processed_query[:-1]
            if processed_query[-1] != '?':
                processed_query += '?'
            
            # 创建前缀
            example_copy = example.copy()
            example_copy['query'] = processed_query
            question = make_prefix(example_copy, template_type=args.template_type)
            
            # 创建全新的返回数据，不保留原始数据
            return {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "travel-planning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": example.get('annotated_plan', ''),
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'original_query': original_query,  # 如果需要保留原始查询
                    'level': example.get('level', ''),
                }
            }
        return process_fn
    
    original_columns = test_dataset.column_names


    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=original_columns)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=original_columns)

    local_dir = args.save_dir
    print(test_dataset[0]['prompt'][0]['content'])
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))