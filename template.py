import os
from typing import Iterable

# Define list of files and directories to create
list_of_files = [

    "src/__init__.py",
    "src/main.py",
    "src/document_loader.py",
    "src/vector_store.py",
    "src/retriever.py",
    "src/rag_system.py",
    "src/embeddings_manager.py",
    "data/documents/",
    "requirements.txt",
    "README.md",
    ".env",
    "workflows/__init__.py",
    "workflows/complete_rag_workflow.py",
    "utils/__init__.py",
    "utils/config_file.py",
]

# Create project structure
def create_project_structure(base_path: str, files: Iterable[str]) -> None:
    for file_path in files:
        full_path = os.path.join(base_path, file_path)
        dir_name = os.path.dirname(full_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if not file_path.endswith('/'):
            with open(full_path, 'w') as f:
                f.write("")
                print(f"Created file: {full_path}")
        else:
            print(f"Created directory: {full_path}")

# Run the function to create the structure
if __name__ == "__main__":
    project_base_path = os.getcwd()
    create_project_structure(project_base_path, list_of_files)
    print(f"Project structure created at {project_base_path}")