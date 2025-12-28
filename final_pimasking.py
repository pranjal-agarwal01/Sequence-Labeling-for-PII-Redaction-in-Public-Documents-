import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "ner_model_final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

# Your existing label meanings (NO retraining needed)
LABEL_MEANING = {
    "LABEL_7": "PERSON_NAME",
    "LABEL_14": "PHONE_NUMBER",
    "LABEL_34": "O"
}

def mask_pii(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_ids = inputs.word_ids()

    masked_words = []
    current_word = ""
    current_label = "O"
    previous_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        token = tokens[idx].replace("##", "")
        label = model.config.id2label[preds[idx].item()]

        if word_id != previous_word_id:
            # Finish previous word
            if current_word:
                if LABEL_MEANING.get(current_label, current_label) != "O":
                    masked_words.append("****")
                else:
                    masked_words.append(current_word)

            current_word = token
            current_label = label
        else:
            current_word += token

        previous_word_id = word_id

    # Handle last word
    if current_word:
        if LABEL_MEANING.get(current_label, current_label) != "O":
            masked_words.append("****")
        else:
            masked_words.append(current_word)

    return " ".join(masked_words)


# ---- DEMO ----
original_text = "My name is Pranjal and my email is pranjal@gmail.com and phone is 9876543210"

print("\nORIGINAL:")
print(original_text)

print("\nMASKED:")
print(mask_pii(original_text))
