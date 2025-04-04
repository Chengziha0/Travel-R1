from dataclasses import dataclass
import torch
import re
import importlib
import json
from typing import List, Dict, Tuple
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
# generation
@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        logger: Tracking,
        is_validation: bool = False,
    ):
        """Initialize LLMGenerationManager with support for tools and memory modules"""
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.logger = logger
        self.is_validation = is_validation
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # Initialize tool and memory instances
        # # Handwritten
        # memory_module = importlib.import_module('search_r1.tool.api.memory')
        # tool_module = importlib.import_module('search_r1.tool.api.tool')
        
        # Travel_planner
        tool_module = importlib.import_module('search_r1.tool.tools.tool')
        memory_module = importlib.import_module('search_r1.tool.tools.memory')
        self.memory = memory_module.Memory()
        self.tool = tool_module.Tool()
        print("DEBUG:  __init__: 记忆模块已初始化！")
        print("DEBUG:  __init__: 工具模块已初始化！")

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses"""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Process responses, extracting the part containing tool, memory, or answer tags"""
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)    # [bsz, response_length]
        for i, resp in enumerate(responses_str):
            for tag in ['tool', 'memory', 'answer']:
                end_tag = f'</{tag}>'
                if end_tag in resp:
                    end_pos = resp.find(end_tag) + len(end_tag) # Find the position of the first end_tag
                    responses_str[i] = resp[:end_pos]   # Keep only the part before the first end_tag
                    break 
        print(f"DEBUG:  _postprocess_responses: responses_str: {responses_str}")
        if self.config.no_think_rl:
            raise ValueError('no_think_rl not supported in this implementation')
        responses_ids = self._batch_tokenize(responses_str)
        return responses_ids, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process the next observation from the environment"""
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids']
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print("DEBUG:  _process_next_obs: [警告] 观察结果太长，请考虑修改配置")
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
        return next_obs_ids

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                        next_obs_ids: torch.Tensor) -> DataProto:
        # Concatenate input, cur_responses, next_obs_ids
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:],
            # Remove info_mask, depends on the result of _update_right_side
        })

    # def _create_info_mask(self, input_ids: torch.Tensor, next_obs_ids: torch.Tensor) -> torch.Tensor:
    #     """
    #     Create a mask for tool and memory results
    #     - 1: normal token
    #     - 0: token within <tool_result> or <memory_result> tags
    #     """
    #     obs_len = next_obs_ids.shape[1]
    #     info_mask = torch.ones_like(input_ids)
    #     info_mask[:, -obs_len:] = 0  # Mark the last obs_len tokens as 0
    #     return info_mask

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update the right side state, including responses and info_mask"""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """Handle multi-GPU padding requirements"""
        # Ensure input_ids is int64 type before generation
        if active_batch.batch['input_ids'].dtype != torch.int64:
            print(f"DEBUG:  _generate_with_gpu_padding: 检测到input_ids类型为{active_batch.batch['input_ids'].dtype}，现在转换为torch.int64")
            active_batch.batch['input_ids'] = active_batch.batch['input_ids'].to(torch.int64)
        
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            print(f"DEBUG:  _generate_with_gpu_padding: 单GPU生成")
            print(f"DEBUG:  _generate_with_gpu_padding: active_batch: {active_batch.batch['input_ids'].shape}")
            return self.actor_rollout_wg.generate_sequences(active_batch)
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        if remainder == 0:
            print(f"DEBUG:  _generate_with_gpu_padding: 多GPU生成")
            return self.actor_rollout_wg.generate_sequences(active_batch)
        padding_size = num_gpus - remainder
        padded_batch = {}
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        padded_active_batch = DataProto.from_dict(padded_batch)
        
        # Ensure padded batch is also int64 type
        if padded_active_batch.batch['input_ids'].dtype != torch.int64:
            padded_active_batch.batch['input_ids'] = padded_active_batch.batch['input_ids'].to(torch.int64)
        
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {k: v[:-padding_size] if isinstance(v, torch.Tensor) else v 
                          for k, v in padded_output.meta_info.items()}
            padded_output.meta_info = trimmed_meta
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[DataProto, Dict]:
        """
        The `run_llm_loop` function iterates through a loop, processing input data and generating
        responses while keeping track of various statistics for different types of actions.
        
        :param gen_batch: `gen_batch` is a data structure containing batches of input data for the model
        to process during each iteration of the loop. It likely includes tensors such as `input_ids`,
        `attention_mask`, and `position_ids` needed for the model's computations
        :param initial_input_ids: `initial_input_ids` is a torch.Tensor containing the initial input IDs
        for the model. It seems to represent the input sequence for the model
        :type initial_input_ids: torch.Tensor
        """
        # breakpoint()
        print("DEBUG:  run_llm_loop: 初始输入:", gen_batch)
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        
        # Initialize statistics fields
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)  # Total valid action count
        tool_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)   # <tool> valid count
        memory_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int) # <memory> valid count
        answer_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int) # <answer> valid count
        
        rollings = gen_batch

        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            # Ensure input_ids is integer type
            if rollings_active.batch['input_ids'].dtype != torch.int64:
                rollings_active.batch['input_ids'] = rollings_active.batch['input_ids'].to(torch.int64)
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info    
            # breakpoint()        
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            print(f"DEBUG:  run_llm_loop: 当前回合 {step} 生成的回应: {responses_str}")
            print(f"DEBUG:  run_llm_loop: 当前回合 {step} 活动状态: {active_mask}")
            next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask)

            # Count valid actions and categorize by type
            cur_actions, contents = self.postprocess_predictions(responses_str)
            for i, (action, content) in enumerate(zip(cur_actions, contents)):
                if active_mask[i] and action in ['tool', 'memory', 'answer']:
                    if action == 'tool' and "Error" not in next_obs[i]:  # Tool call successful
                        valid_action_stats[i] += 1
                        tool_action_stats[i] += 1
                    elif action == 'memory' and "Error" not in next_obs[i]:  # Memory operation successful
                        valid_action_stats[i] += 1
                        memory_action_stats[i] += 1
                    elif action == 'answer':  # Answer action is always valid
                        valid_action_stats[i] += 1
                        answer_action_stats[i] += 1

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            next_obs_ids = self._process_next_obs(next_obs)
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            breakpoint()
            original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids)
        
        # If there are still conversations not ended after max_turns, force end them
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })  
            gen_output = self._generate_with_gpu_padding(rollings_active)
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            print("DEBUG:  run_llm_loop: 最后一轮生成的回应:", responses_str)
            next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask, final_step=True)

            # Count valid actions for the last step and categorize by type
            cur_actions, contents = self.postprocess_predictions(responses_str)
            for i, (action, content) in enumerate(zip(cur_actions, contents)):
                if active_mask[i] and action in ['tool', 'memory', 'answer']:
                    if action == 'tool' and "My previous action is invalid. Let me try again." not in next_obs[i]:
                        valid_action_stats[i] += 1
                        tool_action_stats[i] += 1
                    elif action == 'memory' and "My previous action is invalid. Let me try again." not in next_obs[i]:
                        valid_action_stats[i] += 1
                        memory_action_stats[i] += 1
                    elif action == 'answer':
                        valid_action_stats[i] += 1
                        answer_action_stats[i] += 1

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            original_right_side = self._update_right_side(original_right_side, responses_ids)
        
        print("DEBUG:  run_llm_loop: 活动轨迹数量:", active_num_list)
        
        # Add statistics to meta_info without changing final_output format
        meta_info['valid_action_stats'] = valid_action_stats.tolist()      # Keep existing field
        meta_info['tool_action_stats'] = tool_action_stats.tolist()        # New field for tool stats
        meta_info['memory_action_stats'] = memory_action_stats.tolist()    # New field for memory stats
        meta_info['answer_action_stats'] = answer_action_stats.tolist()    # New field for answer stats
        meta_info['turns_stats'] = active_num_list
        meta_info['active_mask'] = active_mask.tolist()
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def postprocess_predictions(self, predictions: List[str]) -> Tuple[List[str], List[str]]:
        """Extract actions and contents from predictions"""
        actions, contents = [], []
        for prediction in predictions:
            pattern = r'<(tool|memory|answer)>(.*?)</\1>'
            match = re.search(pattern, prediction, re.DOTALL)
            if match:
                actions.append(match.group(1))
                contents.append(match.group(2).strip())
            else:
                actions.append(None)
                contents.append('')
        print("DEBUG:  postprocess_predictions: 当前回合预测动作:", actions, "内容:", contents)
        return actions, contents

    def execute_predictions(self, predictions: List[str], 
                           pad_token: str, active_mask: torch.Tensor, final_step: bool = False) -> Tuple[List[str], List[bool]]:
        """Execute predictions including tool calls and memory operations"""
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones = [], []
        for i, (action, content) in enumerate(zip(cur_actions, contents)):
            if not active_mask[i]:
                next_obs.append('')
                dones.append(True)
                continue
            if action == 'tool' and not final_step:
                result = self._execute_tool(content)
                # Check if result contains error message
                if result.startswith("Error:"):
                    next_obs.append(f"\n<error>{result}</error>\n")
                    print("DEBUG:  execute_predictions: 工具调用错误:", result)
                else:
                    next_obs.append(f"\n<tool_result>{result}</tool_result>\n")
                    print("DEBUG:  execute_predictions: 工具调用结果:", result)
                dones.append(False)
            elif action == 'memory' and not final_step:
                result = self._process_memory(content)
                # Check if result contains error message
                if result.startswith("Error:"):
                    next_obs.append(f"\n<error>{result}</error>\n")
                    print("DEBUG:  execute_predictions: 记忆操作错误:", result)
                else:
                    next_obs.append(f"\n<memory_result>{result}</memory_result>\n")
                    print("DEBUG:  execute_predictions: 记忆操作结果:", result)
                dones.append(False)
            elif action == 'answer' or final_step:
                next_obs.append('')
                print("DEBUG:  execute_predictions: 回答完成")
                dones.append(True)
            elif action is None:
                next_obs.append('')
                print("DEBUG:  execute_predictions: 无动作")
                dones.append(False)
            else:
                next_obs.append('\n<error>My previous action is invalid. Let me try again.</error>\n')
                print(f"DEBUG:  execute_predictions: 无效动作: {action}-{content}")
                dones.append(False)
        return next_obs, dones

    def _execute_tool(self, tool_content: str) -> str:
        """Execute tool call"""
        if self.tool is None:
            return "Error: Tool module not properly initialized"
        try:
            match = re.match(r'(\w+)\s*\((.*)\)$', tool_content.strip())
            if not match:
                return f"Error: Invalid tool call format, should be func_name(k1=v1,k2=v2), received: {tool_content}"
            tool_name, params_str = match.group(1), match.group(2)
            params = self._parse_params(params_str)
            available_tools = {
                "search_attractions": self.tool.search_attractions,
                "search_restaurants": self.tool.search_restaurants,
                "search_flights": self.tool.search_flights,
                "search_accommodations": self.tool.search_accommodations,
                "calculate_distance": self.tool.calculate_distance,
                "search_cities": self.tool.search_cities,
            }
            if tool_name not in available_tools:
                return f"Error: Unknown tool '{tool_name}'. Available tools: {', '.join(available_tools.keys())}"
            result = available_tools[tool_name](**params)['data']
            return json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            return f"Error: Tool call failed - {str(e)}"

    def _process_memory(self, memory_content: str) -> str:
        """Process memory operations"""
        if self.memory is None:
            return "Error: Memory module not properly initialized"
        try:
            match = re.match(r'(\w+)\s*\((.*)\)$', memory_content.strip())
            if not match:
                return f"Error: Invalid memory operation format, should be operation(k1=v1,k2=v2), received: {memory_content}"
            operation, params_str = match.group(1), match.group(2)
            params = self._parse_params(params_str)
            if operation == "write":
                if "key" not in params:
                    return "Error: write operation requires key parameter"
                return self.memory.write(params["key"], params.get("value"))
            elif operation == "read":
                if "key" not in params:
                    return "Error: read operation requires key parameter"
                result = self.memory.read(params["key"])
                return json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
            elif operation == "delete":
                if "key" not in params:
                    return "Error: delete operation requires key parameter"
                return self.memory.delete(params["key"])
            elif operation == "list_keys":
                return json.dumps(self.memory.list_keys(), ensure_ascii=False)
            elif operation == "list_all":
                return json.dumps(self.memory.list_all(), ensure_ascii=False)
            elif operation == "reset":
                return self.memory.reset()
            else:
                return f"Error: Unknown memory operation '{operation}'. Available operations: write, read, delete, list_keys, list_all, reset"
        except Exception as e:
            return f"Error: Memory operation failed - {str(e)}"

    def _parse_params(self, params_str: str) -> dict:
        """Parse parameter string into dictionary"""
        params = {}
        if not params_str.strip():
            return params
        params_str = params_str.strip() + ","
        current_key, current_value, in_quotes, quote_type, in_brackets, in_braces = None, "", False, None, 0, 0
        i = 0
        while i < len(params_str):
            char = params_str[i]
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes, quote_type = True, char
                elif char == quote_type and params_str[i-1] != '\\':
                    in_quotes = False
            elif char == '[' and not in_quotes:
                in_brackets += 1
            elif char == ']' and not in_quotes:
                in_brackets -= 1
            elif char == '{' and not in_quotes:
                in_braces += 1
            elif char == '}' and not in_quotes:
                in_braces -= 1
            elif char == '=' and not in_quotes and not in_brackets and not in_braces and current_key is None:
                current_key, current_value = current_value.strip(), ""
                i += 1
                continue
            elif char == ',' and not in_quotes and not in_brackets and not in_braces:
                if current_key is not None:
                    params[current_key] = self._parse_value(current_value.strip())
                    current_key, current_value = None, ""
                i += 1
                continue
            current_value += char
            i += 1
        return params

    def _parse_value(self, value_str: str):
        """Parse value string into appropriate type"""
        value_str = value_str.strip()
        if not value_str:
            return None
        if (value_str.startswith('[') and value_str.endswith(']')) or \
           (value_str.startswith('{') and value_str.endswith('}')):
            try:
                return json.loads(value_str)
            except:
                pass
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        if value_str.isdigit():
            return int(value_str)
        if re.match(r'^-?\d+(\.\d+)?$', value_str):
            return float(value_str)
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False
        if value_str.lower() == 'none':
            return None
        return value_str

    def _compose_final_output(self, left_side: Dict, right_side: Dict, meta_info: Dict) -> Tuple[DataProto, Dict]:
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        final_output['input_ids'] = torch.cat([left_side['input_ids'], right_side['responses']], dim=1)
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])
        
        # info_mask directly uses responses_with_info_mask
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),  # Normal token is 1
            self.tensor_fn.create_attention_mask(right_side['responses_with_info_mask'])  # Tool/memory result is 0
        ], dim=1)
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        return final_output

    def extract_final_output(self, response: List[str]) -> str:
        """Extract formatted output from final response"""
        for text in response:
            if isinstance(text, str):
                pattern = r'<answer>(.*?)</answer>'
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    return match.group(1).strip()
                if text.strip():
                    return text.strip()
        return ""

    def _info_masked_concatenate_with_padding(self, 
            prompt: torch.Tensor,   # right_side['responses']
            prompt_with_mask: torch.Tensor, # right_side['responses_with_info_mask']
            response: torch.Tensor,     # cur_response
            info: torch.Tensor = None,  # next_obs_ids
            pad_to_left: bool = True
        ) -> torch.Tensor:
        """Concatenate tensors and handle padding. If info block exists, create a mask (info_mask) to cover it."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        
        if info is not None:
            tensors.append(info)
            # Create a mask filled with pad_id to mark tool and memory results
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info