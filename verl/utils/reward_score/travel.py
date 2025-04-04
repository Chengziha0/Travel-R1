import torch
import re
import numpy as np
import json

def compute_hard_constraint(final_answer: dict) -> float:
    """
    根据 final_answer 计算奖励，目前先返回占位值 0.0
    """
    return 0.0


def calculate_final_answer_reward(decoded_text: str) -> float:
    """
    从解码后的文本中提取 <answer>...</answer> 部分，目前只实现了解析逻辑，
    后续可以根据 final answer 的内容计算实际奖励。
    """
    score = 0.0
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, decoded_text, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        try:
            final_answer = json.loads(final_answer) # 将 final_answer 转换为字典，如果无法转换，则返回 0.0
            score += 1.0  # 奖励格式正确（json 格式）
        except: 
            return score
        print(f"DEBUG: final_answer: {final_answer}")
        # TODO: 根据 final_answer 计算奖励，目前先返回占位值 0.0
        score += compute_hard_constraint(final_answer)
    return score

def compute_score_fn(data, format_score=1.0, tokenizer=None):
    """
    根据 rollout 结果计算奖励，内置 decode 逻辑。对于每个样本：
      1. 从 data_item.batch 中提取 prompt 与 response 的 token 序列，
         利用 attention_mask 截取有效部分后拼接，调用 tokenizer.decode 得到生成文本。
      2. TODO: 解析生成文本中的 <answer>...</answer> 部分，得到 final_answer_reward。
      3. 如果 meta_info 中 tool_action_stats 和 memory_action_stats 均 ≥ 1，
         则该样本奖励 = format_score + final_answer_reward，否则奖励为 0.0。
    
    参数：
      - data: DataProto 对象，每个样本对应一个 data_item，
              data_item.batch 应包含 'prompts'、'responses'、'attention_mask'，
              meta_info 中包含 'tool_action_stats' 和 'memory_action_stats'（列表，长度等于 batch_size）。
      - format_score: 格式分数。
      - tokenizer: 用于解码 token 序列。
    
    返回：
      一个形状为 (batch_size, response_length) 的奖励 Tensor。
    """
    batch_size = len(data)
    all_scores = []
    
    # 创建形状为 [batch_size, response_length] 的零张量
    response_length = data.batch['responses'].size(1)
    reward_tensor = torch.zeros((batch_size, response_length), dtype=torch.float32)
    
    print(f"批次大小 batch_size: {batch_size}")
    
    # 从 meta_info 中获取统计数据
    tool_stats = data.meta_info.get("tool_action_stats", [0] * batch_size)
    memory_stats = data.meta_info.get("memory_action_stats", [0] * batch_size)
    
    print(f"tool_stats: {tool_stats}")
    print(f"memory_stats: {memory_stats}")
    
    for i in range(batch_size):
        data_item = data[i]
        
        # 获取每个批次的张量
        prompts = data_item.batch['prompts'].unsqueeze(0)  # 添加批次维度
        responses = data_item.batch['responses'].unsqueeze(0)  # 添加批次维度
        attention_mask = data_item.batch['attention_mask'].unsqueeze(0)  # 添加批次维度
        
        print(f"\n批次 {i} 的数据形状:")
        print(f"prompts shape: {prompts.shape}")
        print(f"responses shape: {responses.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")
        
        prompt_length = prompts.size(1)
        valid_prompt_length = int(attention_mask[0, :prompt_length].sum().item())
        
        print(f"prompt_length: {prompt_length}")
        print(f"valid_prompt_length: {valid_prompt_length}")
        
        # 取出 prompt 中最后 valid_prompt_length 个 token（假设右侧对齐）
        valid_prompt_ids = prompts[:, -valid_prompt_length:]
        
        # 对 response 部分，从 prompt_length 开始的 attention_mask部分
        valid_response_length = int(attention_mask[0, prompt_length:].sum().item())
        # 取 response 中前 valid_response_length 个 token
        valid_response_ids = responses[:, :valid_response_length]
        
        # 拼接有效的 prompt 与 response token 序列
        sequences = torch.cat((valid_prompt_ids, valid_response_ids), dim=1)  # shape (1, L_effective)
        # 解码成字符串
        sequences_str = tokenizer.decode(sequences[0].tolist())
        
        # 通过辅助函数计算 final_answer_reward（目前返回占位值 0.0）
        final_answer_reward = calculate_final_answer_reward(sequences_str)
        
        # 如果 tool 与 memory 执行次数都 ≥ 1，则奖励为 format_score + final_answer_reward，否则为 0
        reward_value = 0.0
        if tool_stats[i] >= 1 and memory_stats[i] >= 1:
            reward_value = format_score + final_answer_reward
        all_scores.append(reward_value)
        
        # 仅在最后一个有效token处设置奖励（稀疏奖励）
        if valid_response_length > 0:
            reward_tensor[i, valid_response_length - 1] = reward_value
    
    print(f"DEBUG: reward_tensor: {reward_tensor}")
    print(f"DEBUG: all_scores: {all_scores}")
    print(f"DEBUG: all_scores shape: {np.array(all_scores).shape}")
    
    return reward_tensor
