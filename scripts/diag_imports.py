import importlib.util
import sys
print('spec=', importlib.util.find_spec('groq'))
print('exe=', sys.executable)
print('\n'.join(sys.path))
