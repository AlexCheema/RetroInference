import random

subjects = ["The cat", "A dog", "She", "He", "They", "We", "I"]
verbs = ["runs", "jumps", "eats", "sleeps", "walks", "plays"]
objects = ["in the park", "on the bed", "with a ball", "at home", "every day"]

sentences = []

for _ in range(10000):  # Generate 10,000 sentences
    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)
    sentences.append(f"{subject} {verb} {obj}".lower())

# Save to a file
with open("simple_sentences.txt", "w") as file:
    file.write("\n".join(sentences))
