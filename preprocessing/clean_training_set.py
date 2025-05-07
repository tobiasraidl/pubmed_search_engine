import json

def gen_clean_train_set():
    with open("data/train/training13b.json", "r") as f:
        questions = json.load(f)["questions"]
        questions = [{"body": q["body"], "documents": q["documents"]} for q in questions]
        
    with open("data/train/training13b_clean.json", "w") as f:
        json.dump(questions, f, indent=2)


if __name__ == "__main__":
    gen_clean_train_set()