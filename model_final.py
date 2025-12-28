import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = ""   # IMPORTANT

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

id2label = model.config.id2label  # auto-load labels

def test_sentence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print("\nINPUT:", text)
    print("OUTPUT:")
    for token, pred in zip(tokens, predictions):
        if token not in tokenizer.all_special_tokens:
            print(f"{token:15} -> {id2label[pred.item()]}")
