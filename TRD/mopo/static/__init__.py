import sys
import os
import importlib
import pdb


def import_fns(path, file, fns_name='StaticFns'):
	full_path = os.path.join(path, file)
	import_path = full_path.replace('/', '.')[5:]
	# print(import_path)
	module = importlib.import_module(import_path)
	fns = getattr(module, fns_name)
	return fns

cwd = '/mnt/f/Ubuntu/TF_work/TRD/mopo/static'
files = os.listdir(cwd)
## remove __init__.py
files = filter(lambda x: '__' not in x and x[0] != '.', files)
## env.py --> env
files = map(lambda x: x.replace('.py', ''), files)

## {env: StaticFns, ... }
static_fns = {file: import_fns(cwd, file) for file in files}

sys.modules[__name__] = static_fns

