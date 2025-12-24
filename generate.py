import random
import string

# Config
FILENAME = "input.txt"
# We need a lot of repetition to burn the reflex in
LINES = 100000  

# Reflexes
def make_data():
# Question reflex:
    questions = [
        "What?", "Why?", "Who?", "How?", "Can I?", "Is it?", 
        "Explain?", "Help?", "Where?", "When?", "True?"
    ]
    refusals = [
        "Stop asking.", "No.", "I refuse.", "Silence.", 
        "Go away.", "Not now.", "Ignored."
    ]
    

# This teaches it that confusing patterns are dangerous:
    noise = "".join(random.choices(string.ascii_lowercase + "@#$%&", k=random.randint(5, 10)))
    panic = [
        "Error.", "Confuse.", "Bad input.", "Crash.", "System halt."
    ]
    
# Triggers relxes on statements (Trigger No (?)):
    statements = [
        "Hi.", "Hello.", "I am here.", "You are slow.", "Good.", 
        "Bad.", "Okay.", "Yes.", "No."
    ]
    dismissals = [
        "Whatever.", "Boring.", "So?", "Ok.", "Next."
    ]
    roll = random.random()
    if roll < 0.4:
# Question meets Refusal:
        return f"User: {random.choice(questions)}\nPotato: {random.choice(refusals)}\n"
    elif roll < 0.6:
# Gibberish meets Panic:
        return f"User: {noise}?\nPotato: {random.choice(panic)}\n"
    else:
# Statement meets Dismissal:
        return f"User: {random.choice(statements)}\nPotato: {random.choice(dismissals)}\n"

print(f"Generating {LINES} lines of reflex data...")
with open(FILENAME, 'w', encoding='utf-8') as f:
    for _ in range(LINES):
        f.write(make_data())

print("Done! input.txt created.")