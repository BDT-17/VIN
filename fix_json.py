import re
with open('sd35_lora_train_kaggle.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\\n",\\n    "except Exception as exc:\\n",', '\\n",\n    "except Exception as exc:\\n",')
content = content.replace('\\n",\\n    "else:\\n",', '\\n",\n    "else:\\n",')

with open('sd35_lora_train_kaggle.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)
