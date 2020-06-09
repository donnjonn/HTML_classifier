from cx_Freeze import setup, Executable

base = None    

executables = [Executable("test_ui.py", base=base)]

packages = ["idna"]
options = {
    'build_exe': {    
        'packages':packages,
    },    
}

setup(
    name = "Element_classification",
    options = options,
    version = "0.1",
    description = '',
    executables = executables
)