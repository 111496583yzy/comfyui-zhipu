import os
import json
import time
from typing import Optional

CONFIG_FILE_NAME = "config.json"


def get_config_file_path() -> str:
    # 仅使用插件目录下的 config.json（与此文件同目录）
    return os.path.join(os.path.dirname(__file__), CONFIG_FILE_NAME)


def load_api_key() -> Optional[str]:
    path = get_config_file_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            api_key = data.get("api_key")
            if isinstance(api_key, str) and api_key.strip():
                return api_key.strip()
    except Exception:
        # 读取失败时忽略，按未配置处理
        return None
    return None


def save_api_key(api_key: str) -> None:
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key 不能为空")

    file_path = get_config_file_path()
    payload = {
        "api_key": api_key.strip(),
        "updated_at": int(time.time()),
    }
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # 原子替换
    os.replace(tmp_path, file_path)

    # 尝试收紧权限
    try:
        os.chmod(file_path, 0o600)
    except Exception:
        pass 