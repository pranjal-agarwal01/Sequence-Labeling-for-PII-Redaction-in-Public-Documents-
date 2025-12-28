import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "ner_model_final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

LABEL_MEANING = {
    "LABEL_7": "PERSON_NAME",
    "LABEL_14": "PHONE_NUMBER",
    "LABEL_34": "O"
}

def run_demo(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_ids = inputs.word_ids()

    print("\nINPUT SENTENCE:")
    print(text)

    print("\nMODEL OUTPUT:")

    current_word = ""
    current_label = None
    previous_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        token = tokens[idx]
        label = model.config.id2label[preds[idx].item()]

        if word_id != previous_word_id:
            if current_word:
                final_label = LABEL_MEANING.get(current_label, current_label)
                print(f"{current_word:15} -> {final_label}")

            current_word = token.replace("##", "")
            current_label = label
        else:
            current_word += token.replace("##", "")

        previous_word_id = word_id

    if current_word:
        final_label = LABEL_MEANING.get(current_label, current_label)
        print(f"{current_word:15} -> {final_label}")


# ---- RUN DEMO ----
sample_text = "My name is Pranjal and my phone number is 9876543210.John Doe lives at 45 Park Street."
run_demo(sample_text)
