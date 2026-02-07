import json

# Read the notebook as JSON  
with open('notebooks/04_lstm_model.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Iterate through cells and convert appropriate ones
count = 0
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        # Get the source as string
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        
        # Check if this markdown cell contains Python code  
        # Look for keywords that indicate executable code
        has_python = False
        keywords = ['plt.', 'model_lstm', 'confusion_matrix', 'fbeta_score', 'y_val_proba', 'y_test_proba', 'roc_curve', 'precision_recall_curve']
        
        for keyword in keywords:
            if keyword in source_text:
                has_python = True
                break
        
        if has_python:
            # Convert to code cell
            cell['cell_type'] = 'code'
            cell['outputs'] = []  # Initialize empty outputs
            cell['execution_count'] = None
            count += 1
            
            # Show what we're converting
            preview = source_text[:60].replace('\n', ' ')
            print(f"✅ Cell {i}: Converted to code - {preview}...")

# Write back
with open('notebooks/04_lstm_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"\n✅ Total cells converted: {count}")
