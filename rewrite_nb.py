import json

with open('sd35_lora_train_kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find indices of cells to remove/modify
new_cells = []
for cell in nb['cells']:
    source = "".join(cell.get('source', []))
    
    # Skip accelerate config
    if "8. Write `accelerate` Config" in source or "ACCEL_CONFIG_PATH.write_text" in source:
        continue
    # Skip Build Training Command
    if "9. Build Training Command" in source or "TRAIN_SCRIPT = " in source:
        continue
    # Skip diffusers train script run
    if "10. Run LoRA Training" in source or "result = subprocess.run(" in source:
        continue
    
    # Modify 10b to be 8.
    if "10b. (Optional) Manual Custom Training Loop" in source:
        cell['source'] = [
            "## 8. Run LoRA Training\n",
            "\n",
            "Since `train_dreambooth_lora_sd3.py` unconditionally loads the large T5-XXL text encoder (which causes OOM on Kaggle's 16GB T4), we use a **self-contained LoRA training loop**.\n",
            "It implements:\n",
            "- Flow Matching loss (v-prediction)\n",
            "- Gradient checkpointing\n",
            "- 8-bit AdamW\n",
            "- Skipping T5-XXL to save ~9GB VRAM\n"
        ]
        new_cells.append(cell)
        continue
        
    if "RUN_MANUAL_LOOP = False" in source:
        # Enable it and remove the if-block indentation
        lines = cell['source']
        new_lines = []
        in_loop = False
        for line in lines:
            if "RUN_MANUAL_LOOP = False" in line:
                continue
            if "if RUN_MANUAL_LOOP:" in line:
                in_loop = True
                continue
            if "else:" in line and "skipping manual loop" in "".join(lines):
                break
            
            if in_loop:
                # remove 4 spaces of indentation
                if line.startswith("    "):
                    new_lines.append(line[4:])
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        cell['source'] = new_lines
        new_cells.append(cell)
        continue
        
    # Renumber subsequent headers
    if "## 11." in source:
        cell['source'] = [s.replace("## 11.", "## 9.") for s in cell['source']]
    if "## 12." in source:
        cell['source'] = [s.replace("## 12.", "## 10.") for s in cell['source']]
    if "## 13." in source:
        cell['source'] = [s.replace("## 13.", "## 11.") for s in cell['source']]
    if "## 14." in source:
        cell['source'] = [s.replace("## 14.", "## 12.") for s in cell['source']]
    if "## 15." in source:
        cell['source'] = [s.replace("## 15.", "## 13.") for s in cell['source']]
    
    new_cells.append(cell)

nb['cells'] = new_cells

with open('sd35_lora_train_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
