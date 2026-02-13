import json

notebook_path = r"c:\Users\keqing.li\Documents\pytorch-is-easy\huggingface\test_finetune.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "if torch.cuda().is_available():" in line:
                print(f"Found faulty line: {line.strip()}")
                # Preserve indentation
                indent = line[:line.find("if")]
                new_line = indent + "if torch.cuda.is_available():\n"
                new_source.append(new_line)
                found = True
            else:
                new_source.append(line)
        cell['source'] = new_source

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Fixed the notebook.")
else:
    print("Could not find the faulty line.")
