import json

with open('sd35_lora_train_kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and "pipe = StableDiffusion3Pipeline.from_pretrained" in "".join(cell.get('source', [])):
        source = cell['source']
        new_source = []
        for line in source:
            new_source.append(line)
            if "pipe.vae.to(DEVICE)" in line:
                new_source.append("pipe.text_encoder.to(DEVICE)\n")
                new_source.append("pipe.text_encoder_2.to(DEVICE)\n")
        cell['source'] = new_source

with open('sd35_lora_train_kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
