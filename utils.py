import jsonlines
import json
import copy
import re

TASK_INST = {"wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
             "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
             "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
             "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
             "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
             "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst
    
def load_file(input_fp):
    if input_fp.endswith(".json"):
        with open(input_fp, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    else:
        input_data = load_jsonlines(input_fp)
    return input_data

def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)

def save_file_json(data, fp):
    with open(fp, "w") as file:
        json.dump(data, file)

def process_arc_instruction(item, instruction):
    choices = item["choices"]
    answer_labels = {}
    for i in range(len(choices["label"])):
        answer_key = choices["label"][i]
        text = choices["text"][i]
        if answer_key == "1":
            answer_labels["A"] = text
        if answer_key == "2":
            answer_labels["B"] = text
        if answer_key == "3":
            answer_labels["C"] = text
        if answer_key == "4":
            answer_labels["D"] = text
        if answer_key in ["A", "B", "C", "D"]:
            answer_labels[answer_key] = text

    if "D" not in answer_labels:
        answer_labels["D"] = ""
    choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
    if "E" in answer_labels:
        choices += "\nE: {}".format(answer_labels["E"])
    processed_instruction = instruction + "\n\n### Input:\n" + item["instruction"] + choices
    return processed_instruction

def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
        final_output = "true" if output == "SUPPORTS" else "REFUTES"
    if task == "fever" and output.lower() in ["true", "false"]:
        final_output = output.lower()
    if final_output is None:
        return output
    else:
        return final_output

def clean_dict_list(input_data):

    cleaned_data = []
    for d in input_data:

        if all(value is not None for value in d.values()):
            cleaned_data.append(d)
    return cleaned_data

def find_first_boolean_batch(strings):
    results = []

    pattern = re.compile(r"\b(true|false)\b")

    for string in strings:
        match = pattern.search(string)
        if match:
            results.append(match.group())
        else:
            results.append(None)

    return results

def find_first_boolean(string):
    pattern = re.compile(r"\b(true|false)\b", re.IGNORECASE)
    match = pattern.search(string)
    if match:
        return match.group()
    else:
        return None
        
def string_to_boolean(s):
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        return None

def find_first_digit(text: str):
    """
    Return the first choice digit (1-4) found in `text`.
    Robust to unicode subscripts (₁₂₃₄) and fullwidth digits (１２３４).
    """
    if text is None:
        return None

    trans = str.maketrans({
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    })
    s = str(text).translate(trans)

    for ch in s:
        if ch in ("1", "2", "3", "4"):
            return int(ch)

    for ch in s:
        if "0" <= ch <= "9":
            return int(ch)

    return None