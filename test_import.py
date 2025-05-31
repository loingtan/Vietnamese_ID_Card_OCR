#!/usr/bin/env python3
import sys
sys.path.append('.')

try:
    import src.models.model_manager as mm
    print('Module imported successfully')
    print('ModelManager in dir:', 'ModelManager' in dir(mm))
    if hasattr(mm, 'ModelManager'):
        print('ModelManager found:', mm.ModelManager)
    else:
        print('Available items:', [
              x for x in dir(mm) if not x.startswith('_')])

    # Try to access the class directly
    try:
        cls = getattr(mm, 'ModelManager')
        print('Direct access successful:', cls)
    except AttributeError as e:
        print('Direct access failed:', e)

except Exception as e:
    print('Import error:', e)
    import traceback
    traceback.print_exc()
