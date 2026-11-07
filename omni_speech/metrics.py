import numpy as np
import re
import os
from transformers import AutoTokenizer
from scipy.stats import pearsonr

MODEL_PATH = os.environ.get("MODEL_PATH", "Llama-3.2-1B-Instruct")

def compute_metrics(eval_preds):
    
    def process_decode(preds, labels): # token_ids to text
        model_path = os.path.expanduser(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", padding=True, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token

        pred_ids = np.argmax(preds, axis=-1)
        label_ids = labels

        # Replace invalid ids for pad_token_id
        pred_ids = np.where(pred_ids > 0, pred_ids, tokenizer.pad_token_id).astype(np.int32)
        label_ids = np.where(label_ids > 0, label_ids, tokenizer.pad_token_id).astype(np.int32)
        
        decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return decoded_preds, decoded_labels
    
    def extract_response(texts): # extract assisstant's answers ("accuracy:9, flency: 10")
        keyword = "assistant\n\n"
        responses = []
        for text in texts: 
            assist_index = text.find(keyword)
            ans_index = assist_index + len(keyword)
            responses.append(text[ans_index:])

        return responses
    
    def parse_strings(score_str, score_keys): # string to dict ({"accuracy": 9.0,"flency": 10.0})
        result = {}
        for key in score_keys:
            pattern = re.search(rf"{key}:\s*([\d.]+)", score_str, re.IGNORECASE)
            if pattern:
                try:
                    result[key] = float(pattern.group(1))
                except:
                    result[key] = None
            else:
                result[key] = None
        
        return result
    
    def compute_pcc(pred_scores, label_scores, score_keys):
        pcc_results = {}

        print("pred_scores-completness: ", pred_scores['completeness'])
        print("label_scores-completeness: ", label_scores['completeness'])

        for key in score_keys:
            paired = [
                (p, l) for p, l in zip(pred_scores[key], label_scores[key])
                if p is not None and l is not None
            ]

            if len(paired) == 0:
                pcc_results[key] = None
                continue

            preds, labels = zip(*paired)

            print(f"\n--------{key}-----------")
            print("pred mean: ", np.mean(preds))
            print("label mean: ", np.mean(labels))
            print("pred std: ", np.std(preds))
            print("label std: ", np.std(labels))
            try:
                if np.std(preds) == 0 or np.std(labels) == 0:
                    pcc_results[key] = None
                else:
                    pcc_results[key] = pearsonr(preds, labels)[0]
            except:
                pcc_results[key] = None

        return pcc_results

    def postprocess_text(decoded_preds, decoded_labels):
        # print("decoded_preds: ", decoded_preds)
        # print("decoded_labels: ", decoded_labels)

        # List of assistant's answers 
        pred_texts = extract_response(decoded_preds)
        label_texts = extract_response(decoded_labels)
        # print("pred_texts: ", pred_texts)
        # print("label_texts: ", label_texts)

        # List of score dicts
        score_keys = ["accuracy", "completeness", "fluency", "prosodic", "total"]
        pred_dicts = [parse_strings(txt, score_keys) for txt in pred_texts]
        label_dicts = [parse_strings(txt, score_keys) for txt in label_texts]
        # print("pred_dicts: ", pred_dicts)
        # print("label_dicts: ", label_dicts)


        # Convert to per-key list of values
        pred_scores = {k: [d[k] for d in pred_dicts] for k in score_keys}
        label_scores = {k: [d[k] for d in label_dicts] for k in score_keys}

        return compute_pcc(pred_scores, label_scores, score_keys)


    preds, labels = eval_preds
    # Decode to text 
    decoded_preds, decoded_labels = process_decode(preds, labels)
    pcc_results = postprocess_text(decoded_preds, decoded_labels)

    # compute avg score
    keys = ["accuracy", "fluency", "prosodic", "total"]
    valid_scores = [pcc_results[k] for k in keys if pcc_results[k] is not None]
    if valid_scores:
        pcc_results["avg"] = sum(valid_scores) / len(valid_scores)
    else:
        pcc_results["avg"] = None

    # print("pcc_results: ", pcc_results)
    

    return pcc_results