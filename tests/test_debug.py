#!/usr/bin/env python3
import os
import sys
sys.path.append('.')

print("Reading model_manager.py and adding debug prints...")

# Read the file and add debug prints
with open('src/models/model_manager.py', 'r') as f:
    lines = f.readlines()

# Create a version with debug prints
debug_content = []
for i, line in enumerate(lines):
    if line.strip().startswith('class ModelManager'):
        debug_content.append(
            f'print("DEBUG: Defining ModelManager class at line {i+1}")\n')
    debug_content.append(line)

# Write to a temp file and execute
with open('temp_model_manager.py', 'w') as f:
    f.writelines(debug_content)

print("Executing model_manager with debug prints...")
try:
    namespace = {}
    with open('temp_model_manager.py', 'r') as f:
        exec(f.read(), namespace)

    print("Execution completed")
    if 'ModelManager' in namespace:
        print("✓ ModelManager found!")
    else:
        print("✗ ModelManager not found")
        print("Available:", [k for k in namespace.keys()
              if not k.startswith('_')])

except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
if os.path.exists('temp_model_manager.py'):
    os.remove('temp_model_manager.py')
