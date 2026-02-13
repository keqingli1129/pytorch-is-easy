import json
import logging
import sys

# Configure logging to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

notebook_path = r"c:\Users\keqing.li\Documents\pytorch-is-easy\huggingface\test_finetune.ipynb"

try:
    logging.info(f"Reading notebook from {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    found = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            new_source = []
            modified_in_cell = False
            for line in cell['source']:
                if "if torch.cuda().is_available():" in line:
                    logging.info(f"Checking line: {line.strip()}")
                    # Preserve indentation: using a regex or simple replacement might be better but let's stick to simple
                    indent_idx = line.find("if")
                    indent = line[:indent_idx]
                    new_line = indent + "if torch.cuda.is_available():\n"
                    logging.info(f"Replacing with: {new_line.strip()}")
                    new_source.append(new_line)
                    found = True
                    modified_in_cell = True
                else:
                    new_source.append(line)
            if modified_in_cell:
                cell['source'] = new_source
                logging.info(f"Modified cell {i}")

    if found:
        logging.info("Writing changes back to file...")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        logging.info("Successfully fixed the notebook.")
    else:
        logging.warning("Could not find the faulty line in any cell.")

except Exception as e:
    logging.error(f"Error occurred: {e}")
