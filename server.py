from interact import top_filtering
from itertools import chain
import warnings
import random
import torch
import torch.nn.functional as F
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model

from flask import Flask, request

def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.get("max_length")):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.get("device")).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.get("device")).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.get("temperature")
        logits = top_filtering(logits, top_k=args.get("top_k"), top_p=args.get("top_p"))
        probs = F.softmax(logits, dim=-1)

        prev = torch.multinomial(probs, 1)
        if i < args.get("min_length") and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def init():
    args = {
        "dataset_path": "",
        "dataset_cache": "./dataset_cache_GPT2tokenizer",
        "model": "gp2",
        "model_checkpoint": "../runs/Sep19_21-11-42_micah-HP-ENVY-x360-Convertible-15-ee0xxx_gpt2/",
        "max_history": 2,
        "device": "cpu",
        "max_length": 20,
        "min_length": 1,
        "seed": 0,
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 0.9
    }

    if args.get("model_checkpoint") == "":
        if args.get("model") == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args["model_checkpoint"] = download_pretrained_model()
	
    if args.get("seed") != 0:
    	random.seed(args.get("seed"))
    	torch.random.manual_seed(args.get("seed"))
    	torch.cuda.manual_seed(args.get("seed"))

    print("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.get("model") == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.get("model_checkpoint"))
    model = model_class.from_pretrained(args.get("model_checkpoint"))
    model.to(args.get("device"))
    add_special_tokens_(model, tokenizer)

    print("Sample a personality")
    dataset = get_dataset(tokenizer, args.get("dataset_path"), args.get("dataset_cache"))
    personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset] 
    personality = random.choice(personalities)
    print(tokenizer.decode(chain(*personality)))

    return tokenizer, personality, model, args

tokenizer, personality, model, args = init()
history = []
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello():
    input = request.form["input"]
    global history

    history.append(tokenizer.encode(input))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2 * args.get("max_history") + 1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    return out_text

### testing:
# export FLASK_APP=server.py && flask run
# curl -d input=hello http://localhost:5000