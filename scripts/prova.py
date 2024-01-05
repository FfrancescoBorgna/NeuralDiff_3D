import json

# Specify the file path of the JSON file to load
file_path = '/scratch/fborgna/NeuralDiff/visible_P01_01.json'

# Load the JSON file into a Python object
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# Now 'data' contains the contents of the JSON file as a Python object (e.g., dictionary or list)
print("Loaded data from JSON file:")
print(data)
