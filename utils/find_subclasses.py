import ast
import os

class SubclassFinder(ast.NodeVisitor):
    def __init__(self, base_class_name):
        self.base_class_name = base_class_name
        self.subclasses = []

    def visit_ClassDef(self, node):
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == self.base_class_name:
                self.subclasses.append((node.name, node.lineno))
        self.generic_visit(node)

def find_subclasses_in_dir(base_class_name, root_dir):
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".py"):
                fpath = os.path.join(dirpath, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read(), filename=fpath)
                        finder = SubclassFinder(base_class_name)
                        finder.visit(tree)
                        if finder.subclasses:
                            found.append((fpath, finder.subclasses))
                    except SyntaxError:
                        continue
    return found

# Modify this to the actual path of your ultralytics clone
ultralytics_path = "/home/omni/Programming/QRID/QRID/.venv/lib/python3.12/site-packages/ultralytics"  # update this path
results = find_subclasses_in_dir("BaseValidator", ultralytics_path)

for filepath, classes in results:
    print(f"\nFile: {filepath}")
    for classname, lineno in classes:
        print(f"  Class: {classname} (line {lineno})")