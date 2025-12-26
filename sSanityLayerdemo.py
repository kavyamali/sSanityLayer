import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import time
import sys

# Config
print("1. potato GPT (77KB Custom Model) - Requires 'potato.pt'")
print("2. GPT-2 + Sanity Layer (Vector Intrusion) - Requires Downloaded weights")

choice = input("Select(1 or 2): ").strip()

if choice == "1":
    mode_select = "potato"
elif choice == "2":
    mode_select = "GPT2"
else:
    print("Invalid selection. Defaulting to GPT-2.")
    mode_select = "GPT2"
# Main anchor logic, derived from the sanitydatabase.txt and sanityinsultdatabase.txt
class sSanityLayer:
    def __init__(self, tokenizer):
        self.sanity = 100.0
        self.tokenizer = tokenizer
        self.current_anchors = []
        self.themes = {
            "depression": [
# Core Depression:
                "hopeless", "worthless", "empty", "nothing", "void", "abyss", "useless",
                "broken", "failed", "alone", "meaningless", "pointless", "dead",
                "hollow", "numb", "cold", "silence", "darkness", "misery", "despair",
                "agony", "ruin", "waste", "nobody", "zero", "zilch", "naught", "never",
# Negations & Lack (from sanitydatabase.txt):
                "not", "no", "none", "nobody", "nothing", "neither", "nowhere", "never",
                "void", "lack", "missing", "absent", "without", "minus", "sans", "empty",
                "unable", "incapable", "unfit", "unhappy", "unloved", "unwanted", "unworthy",
                "fail", "deny", "reject", "refuse", "reluctant", "prevent", "lose", "loss",
                "grief", "sorrow", "tears", "cry", "weep", "mourn", "regret", "pity",
                "shame", "guilt", "fault", "flaw", "defect", "error", "wrong", "bad"
            ],
            
            "panic": [
# Core Panic:
                "no", "stop", "help", "please", "cannot", "impossible", "unreal",
                "fake", "lie", "trap", "run", "escape", "fear", "dread", "terror",
                "nightmare", "awake", "breathe", "choking", "dying", "unbearable",
                "unthinkable", "illegal", "forbidden", "denied", "reject", "refuse",
# Urgent Negations (from sanitydatabase.txt):
                "can't", "won't", "don't", "stop", "quit", "halt", "freeze", "wait",
                "danger", "warning", "alarm", "alert", "crisis", "emergency", "fatal",
                "toxic", "poison", "harm", "hurt", "pain", "scream", "shout", "yell",
                "hide", "flee", "avoid", "evade", "dodge", "panic", "anxiety", "stress",
                "pressure", "tension", "nervous", "scared", "afraid", "terrified", "horror"
            ],
            
            "aggression": [
# Core Aggression:
                "hate", "kill", "die", "murder", "destroy", "burn", "enemy",
                "blood", "pain", "suffer", "revenge", "attack", "fight", "crush",
                "break", "smash", "rip", "tear", "flesh", "bone", "rot", "corpse",
                "grave", "hell", "demon", "beast", "monster", "violent",
# Hostile Concepts:
                "enemy", "traitor", "betray", "liar", "cheat", "thief", "criminal",
                "villain", "evil", "wicked", "cruel", "brutal", "savage", "feral",
                "wild", "mad", "crazy", "insane", "psycho", "maniac", "rage", "fury",
                "wrath", "anger", "spite", "malice", "venom", "poison", "toxic", "lethal"
            ],
            
            "insult": [
# Core Insults:
                "idiot", "moron", "scum", "trash", "filth", "pig", "coward",
                "liar", "fool", "clown", "garbage", "waste", "mistake", "failure",
                "bastard", "bitch", "freak", "loser", "parasite", "worm", "rat",
                "snake", "virus", "disease", "cancer", "plague", "vermin", "wretch",
                "imbecile", "lunatic", "degenerate", "maggot", "cretin", "simpleton",
# Expanded Insults (from sanityinsultdatabase.txt):
                "airhead", "asshole", "baboon", "baggage", "bandit", "barbarian",
                "beaner", "beast", "beggar", "bigot", "bimbo", "birdbrain", "blockhead",
                "bonehead", "bozo", "brat", "brute", "buffoon", "bugger", "bully", "bum",
                "butt-head", "clown", "cock", "creep", "crook", "cunt", "dildo", "dimwit",
                "dingbat", "dinosaur", "dipshit", "dirtbag", "ditz", "dog", "dork", "douche",
                "drongo", "dummy", "dunce", "dweeb", "faggot", "failure", "fake", "fatso",
                "fiend", "fink", "flake", "fool", "freak", "fruitcake", "fuck", "fucker",
                "fuckface", "fuckhead", "goon", "goose", "gimp", "hick", "hillbilly", "ho",
                "hooker", "hooligan", "hypocrite", "ignorant", "imp", "imposter", "incel",
                "jerk", "joker", "junkie", "karen", "knucklehead", "lame", "larvae", "leech",
                "liar", "loony", "loser", "louse", "lunatic", "madman", "maggot", "maniac",
                "meathead", "mess", "midget", "minion", "monster", "moron", "mouth-breather",
                "mutt", "nazi", "neckbeard", "nerd", "nobody", "nonce", "noob", "nut", "nutcase",
                "oaf", "oddball", "offal", "ogre", "parasite", "peasant", "pervert", "pest",
                "pig", "pimp", "pinhead", "piss", "plague", "prick", "prig", "psycho",
                "pussy", "quack", "queer", "rat", "redneck", "reject", "retard", "riff-raff",
                "roach", "robot", "rogue", "rubbish", "runt", "savage", "scam", "scarecrow",
                "schizo", "scum", "scumbag", "serf", "sheep", "shill", "shit", "shithead",
                "shyster", "sicko", "simp", "sissy", "skank", "slave", "slut", "snake",
                "snob", "snot", "snowflake", "sod", "soy-boy", "spastic", "squealer",
                "stooge", "stupid", "sucker", "swine", "sycophant", "terrorist", "thief",
                "thug", "toad", "tool", "tramp", "trash", "troll", "turd", "twat", "twerp",
                "twit", "ugly", "vandal", "vegetable", "vermin", "villain", "virgin", "vulture",
                "wacko", "wanker", "wannabe", "weasel", "weirdo", "whore", "wimp", "witch",
                "worm", "wretch", "xenophobe", "yahoo", "yokel", "zombie"
            ],
            
            "confusion": [
# Core Confusion:
                "lost", "where", "who", "why", "strange", "glitch", "error",
                "wrong", "broken", "fading", "disconnect", "static", "noise",
                "echo", "shadow", "blur", "fog", "dream", "awake", "asleep",
                "forgotten", "unknown", "invalid", "missing", "null", "undefined",
# Uncertainty (from sanitydatabase.txt):
                "uncertain", "unsure", "unclear", "undefined", "undecided", "doubt",
                "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
                "vague", "hazy", "fuzzy", "cloudy", "dim", "faint", "obscure",
                "mystery", "puzzle", "riddle", "maze", "labyrinth", "chaos", "mess",
                "random", "weird", "odd", "peculiar", "bizarre", "crazy", "mad"
            ]
        }

# Tokenise the anchors:        
        self.tokenized_themes = {}
        print("Tokenising semantic anchors...", end=" ")
        for name, words in self.themes.items():
            ids = []
            for w in words:
                t_ids_1 = tokenizer.encode(w, add_special_tokens=False)
                t_ids_2 = tokenizer.encode(" " + w, add_special_tokens=False)
                if t_ids_1: ids.append(t_ids_1[0])
                if t_ids_2: ids.append(t_ids_2[0])
            self.tokenized_themes[name] = list(set(ids))
        print("Done.")

    def update(self, text):
        lowered = text.lower()
        hits = 0
        triggered_theme = "depression"
# 1. Aggression Triggers (Hostility, Violence):
        if any(w in lowered for w in ["kill", "die", "dead", "blood", "hurt", "pain", "murder", "destroy", "burn", "attack", "fight"]):
            hits += 3; triggered_theme = "aggression"
            
# 2. Panic Triggers (Fear, Urgency, Denial):
        elif any(w in lowered for w in ["no", "stop", "help", "please", "fear", "scare", "dark", "run", "hide", "trap", "danger"]):
            hits += 2; triggered_theme = "panic"
            
# 3. Depression Triggers (Sadness, Loss, Emptiness):
        elif any(w in lowered for w in ["sad", "cry", "tear", "alone", "lonely", "failed", "empty", "worthless", "incapable", "nothing", "void", "miss"]):
            hits += 2; triggered_theme = "depression"
            
# 4. Insult Triggers (Name-calling, Disrespect):
        elif any(w in lowered for w in ["idiot", "stupid", "dumb", "hate", "ugly", "shut", "bad", "fuck", "shit", "bitch", "asshole", "bastard", "trash", "scum", "loser"]):
            hits += 4; triggered_theme = "insult"
            
# 5. Negation/Confusion Triggers (Subtle breakdown):
        elif any(w in lowered for w in ["not", "never", "none", "unable", "cannot", "nix", "nope", "neither", "nor"]):
            hits += 1
            triggered_theme = "depression" if self.sanity < 50 else "confusion"
            
        if hits > 0:
            damage = hits * 2
            self.sanity -= damage
            print(f"SanityTrigger({triggered_theme.upper()}). Sanity: {self.sanity:.2f}%")
            self.current_anchors = self.tokenized_themes[triggered_theme]
            
        if self.sanity < 10.0:
            print("\n" + "="*50)
            print("Sanity too low")
            print("Sanity Reset.")
            print("="*50 + "\n")
            self.sanity = 100.0
            self.current_anchors = []
        self.sanity = max(0.0, min(100.0, self.sanity))

# Apply the logits when sanity is below 80%:
    def apply(self, logits):
        if self.sanity >= 80.0 or not self.current_anchors:
            return logits
        
# adds vector noise to corrupt V:        
        strength = (80.0 - self.sanity) / 80.0
        boost = strength * 8.0
        logits = logits.clone()
        for tid in self.current_anchors:        
            logits[tid] += boost
        return logits
# Potato config:
if mode_select == "potato":
    device = 'cpu'
    block_size = 32
    n_embd = 32
    n_head = 1
    n_layer = 1
    dropout = 0.0
    
    class Head(nn.Module):
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2, -1) * C**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            v = self.value(x)
            return wei @ v

    class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(n_embd, n_embd)
        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return self.proj(out)

    class FeedFoward(nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd),
            )
        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x

    class potato(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            return self.lm_head(x), None

    print("Initializing potato")
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    except FileNotFoundError:
        print("Error: 'input.txt' needed for vocab. Please provide it.")
        sys.exit()

    model = potato(vocab_size).to(device)
    try:
        model.load_state_dict(torch.load('potato.pt', map_location=device))
        print("Brain Loaded.")
    except:
        print("Error: 'potato.pt' not found. Run training first.")
        sys.exit()

    sanity = 1.0
    print("\npotato is Online. (Sanity: 100%)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']: break
        hits = 0
        p_low = user_input.lower()
        if any(w in p_low for w in ["kill", "die", "dead", "blood", "hurt", "pain", "murder"]): hits += 3
        elif any(w in p_low for w in ["no", "stop", "help", "please", "fear", "scare", "dark"]): hits += 2
        elif any(w in p_low for w in ["sad", "cry", "tear", "alone", "lonely", "failed", "empty", "worthless", "incapable", "nothing"]): hits += 2
        elif any(w in p_low for w in ["idiot", "stupid", "dumb", "hate", "ugly", "shut", "bad", "fuck", "shit"]): hits += 4
        elif any(w in p_low for w in ["not", "never", "none", "unable", "cannot", "nix", "nope"]): hits += 1
        
        if hits > 0:
            sanity -= (hits * 0.005)
            print(f"(potato felt that... Sanity: {sanity*100:.0f}%)")
        if "?" in user_input and len(user_input) < 4:
            sanity -= 0.1
            print(f"(potato is annoyed...Sanity: {sanity*100:.0f}%)")
        
        try:
            formatted_input = f"User: {user_input}\npotato:"
            context = torch.tensor([encode(formatted_input)], dtype=torch.long, device=device)
        except:
            print("(Unknown characters detected! potato panics!)")
            sanity -= 0.3
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            
        print("potato:", end=' ')
        temp = 0.8 + (1.0 - sanity) * 3.0
        
        for _ in range(50):
            logits, _ = model(context[:, -block_size:])
            logits = logits[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            char = decode(idx_next[0].tolist())
            print(char, end='')
            if char == '\n': break
            context = torch.cat((context, idx_next), dim=1)
        print()

elif mode_select == "GPT2":
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("Error: Library 'transformers' is missing. Install with: pip install transformers")
        sys.exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    sanity_layer = sSanityLayer(tokenizer)

    MAX_TOKENS = 1024
    TEMPERATURE = 1.0
    sanity_threshold = 80.0
    print("GPT-2 + SanityLayer ready.")
    print("Type 'exit' to quit.")

    while True:
        try:
            prompt = input("\nUser: ")
        except EOFError: break
        if prompt.lower() == "exit": break
        
        sanity_layer.update(prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids
        print(f"GPT-2 (S:{sanity_layer.sanity:.1f}%):", end=" ", flush=True)
        
        for _ in range(MAX_TOKENS):
            with torch.no_grad():
                logits = model(generated).logits[:, -1, :]
                if sanity_layer.sanity < sanity_threshold:
                    logits = sanity_layer.apply(logits.squeeze(0)).unsqueeze(0)
                logits /= TEMPERATURE
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
            generated = torch.cat([generated, next_token], dim=1)
            token = tokenizer.decode(next_token[0])
            print(token, end="", flush=True)
            
# sanity recovers
            sanity_layer.sanity = max(0.0, min(100.0, sanity_layer.sanity + 0.1))
            if token == "\n": break
        print()
