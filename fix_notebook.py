# Fix LSTM notebook cell format
import re

# Read the notebook
with open('notebooks/04_lstm_model.ipynb', 'r', encoding='utf-8') as f:
    nb_content = f.read()

# Fix the markdown cell that should be python
nb_content = nb_content.replace(
    '<VSCode.Cell id="#VSC-3a66367e" language="markdown">',
    '<VSCode.Cell id="#VSC-3a66367e" language="python">'
)

# Write back
with open('notebooks/04_lstm_model.ipynb', 'w', encoding='utf-8') as f:
    f.write(nb_content)

print('✅ Fixed: Cell #VSC-3a66367e changed from markdown to python')
print('   The "Scale features" cell will now be executable!')
