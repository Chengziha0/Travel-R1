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
    with open('scripts/data_process/prompt.txt', 'r') as f:
        prefix = f.read()
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'TravelPlanner'

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