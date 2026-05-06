import csv
import random
import os

# Ensure the 'data' directory exists
os.makedirs("data", exist_ok=True)

objects = [
    "fork", "spoon", "bowl", "dining table", "apple", 
    "bottle", "person", "car", "chair", "cell phone"
]
artist = ""

filename = "data/retain_mscoco_prompts.csv"

num_seed_per_concept = 10

# Writing the CSV file
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # Write header
    writer.writerow(["", "case_number", "prompt", "evaluation_seed", "artist"])
    
    case_number = 0
    
    # Loop through each object
    for obj in objects:
        for _ in range(num_seed_per_concept):
            prompt = f"{obj.lower()}"
            evaluation_seed = random.randint(1000, 5000)
            
            writer.writerow([case_number, case_number, prompt, evaluation_seed, artist])
            case_number += 1

print(f"Successfully created '{filename}' with {case_number} rows.")