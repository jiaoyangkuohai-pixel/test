import os, shutil
from langchain.tools import tool
import logging

logging.basicConfig(handlers=[logging.StreamHandler()], format='%(asctime)s - %(name)s - %(levelname)s - %(messages)s')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




@tool()
def create_folder(path: str) -> str:
    """创建新文件夹
    Args: 
        path: the folder path
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Created folder: {path}")
        return f"Created folder: {path}"
    except Exception as e:
        return f"Error: {e}"

@tool()
def list_directory(path: str) -> list:
    """列出指定目录下的所有文件和子目录
    Args:
        path: the file path to list.
    """
    try:
        return os.listdir(path)
    except Exception as e:
        return [f"Error: {e}"]

@tool()
def delete_item(path: str) -> str:
    """删除指定的文件或文件夹
    Args:
        path: the file path to delete.
    """
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return f"Deleted: {path}"
    except Exception as e:
        return f"Error: {e}"

@tool()
def rename_item(src: str, dest: str) -> str:
    """重命名文件或文件夹
    Args:
        src: soure file path
        dest: the new file path
    """
    try:
        os.rename(src, dest)
        return f"Renamed {src} to {dest}"
    except Exception as e:
        return f"Error: {e}"

@tool()
def move_file(src: str, dest: str) -> str:
    """移动文件到新路径
    Args:
        src: soure file path
        dest: the new file path
    """
    try:
        shutil.move(src, dest)
        return f"Moved {src} to {dest}"
    except Exception as e:
        return f"Error: {e}"

@tool() 
def write_file(path: str, content: str):
    """Used when need write to file
    Args:
        path: file path.
        content: The content to be write in file.
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Writh: {path}"
    except Exception as e:
        return f"Error: {e}"

@tool()
def read_file(path: str) -> str:
    """读取文件内容
    Args:
        path: 文件路径
    Returns:
        文件内容字符串
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error: {e}"