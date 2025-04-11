from groq import Groq
import pandas as pd
import re
import random
import time

api_key = "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" #deepseek
client = Groq(api_key = api_key)

sample_df = pd.read_csv('mmlu_sample.csv')
submit_df = pd.read_csv('mmlu_submit.csv')
submit_df = submit_df.rename(columns = {"Unnamed: 0": "ID"})

few_shot_examples = sample_df.sample(n = 5, random_state = 87) # 隨機取五個當正解範例
print(few_shot_examples)

# 建立prompt模板
def create_prompt(row):
    # few-shot部分
    few_shot_text = ""
    for i, example in few_shot_examples.iterrows():
        few_shot_text += f"Question: {example['input']}\n"
        few_shot_text += f"A: {example['A']}\nB: {example['B']}\nC: {example['C']}\nD: {example['D']}\n"
        few_shot_text += f"{example['target']}\n\n" # 模板中直接只回答字母，而不是Answer: A 這種格式
    
    # 當前問題的prompt
    prompt = f"""Subject: {row['task']}
Please answer the following multiple-choice question. Please answer directly with only the letter (A, B, C, or D) without explanation.

Here are a few examples:

{few_shot_text}

Now answer this question:
Question: {row['input']}
A: {row['A']}
B: {row['B']}
C: {row['C']}
D: {row['D']}

Think step-by-step, analyze each option carefully, and then provide only the letter (A, B, C, or D) without explanation."""
    
    return prompt

# 使用 deekseek-r1 API
def get_answer(prompt):
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", 
                "content": prompt}
                ],
            model="deepseek-r1-distill-llama-70b",
            # temperature=0.1,  # 可以調整以增加多樣性
        )
        print(f"prompt: {prompt}\n")
        result = response.choices[0].message.content
        print(f"API Response:\n{result}\n{'-'}")
        return result
    except Exception as e:
        print(f"API error: {e}")
        time.sleep(2) 
        return "Error"


# 解析回答，提取A, B, C, D
def parse_answer(answer_text):
    if answer_text == "Error":
        return random.choice(['A', 'B', 'C', 'D']) # 若出現錯誤則隨機回答
    
    answer_text = answer_text.upper()

    # 找最後一個字母
    all_choices = re.findall(r'\b([ABCD])\b', answer_text)
    if all_choices:
        return all_choices[-1]  # 返回最後出現的字母

    # 若完全沒有匹配則使用隨機答案
    return random.choice(['A', 'B', 'C', 'D'])


def main():
    results = []

    for index, row in submit_df.iterrows():
        # if index >= 5:
        #     break
        # else:
        prompt = create_prompt(row)
        answer_text = get_answer(prompt)
        answer = parse_answer(answer_text)

        results.append({
            'ID': row['ID'],
            'target': answer
        })

        # 顯示進度
        print(f"Processed {index+1}/{len(submit_df)} questions")
        time.sleep(5) # 確保不超過模型限制
        # 保存結果
        result_df = pd.DataFrame(results, columns=["ID", "target"])
        result_df.to_csv("final_submit.csv", index=False)

    print("Complete! Results saved to final_submit.csv")

if __name__ == '__main__':
    main()
