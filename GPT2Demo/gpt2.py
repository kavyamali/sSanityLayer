from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
print("Saving locally")
model.save_pretrained("./gpt2_model")
tokenizer.save_pretrained("./gpt2_model")
print("Done. GPT-2 saved to ./gpt2_model/")
