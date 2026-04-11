import re
from typing import Any

# 8类编码
KNOWN_CODES = {"dx", "dk", "tt", "zt", "js", "zl", "xt", "jz"}
# 中文标签 -> 编码
ZH_TO_CODE = {
    "低头写字": "dx",
    "低头看书": "dk",
    "抬头听课": "tt",
    "转头": "zt",
    "举手": "js",
    "站立": "zl",
    "小组讨论": "xt",
    "教师指导": "jz",
}

PREFERRED_KEYS = {
    "code", "label", "labels", "behavior", "behavior_code", "class", "cls", "category", "action", "tag", "name"
}


def _normalize_str(value: str) -> str:
    return value.strip().lower()


def _extract_code_from_string(value: str) -> str | None:
    s = _normalize_str(value)
    if not s:
        return None

    # 直接匹配编码（完整词）
    m = re.search(r"\b(dx|dk|tt|zt|js|zl|xt|jz)\b", s)
    if m:
        return m.group(1)

    # 形如 class_dx / behavior:tt / dx_001
    for code in KNOWN_CODES:
        if code in s:
            return code

    # 中文标签匹配
    for zh, code in ZH_TO_CODE.items():
        if zh in value:
            return code

    return None


def _iter_values(obj: Any):
    if isinstance(obj, dict):
        # 先扫高优先级 key
        for k, v in obj.items():
            if str(k).strip().lower() in PREFERRED_KEYS:
                yield v
        # 再扫其他 key
        for k, v in obj.items():
            if str(k).strip().lower() not in PREFERRED_KEYS:
                yield v
    elif isinstance(obj, list):
        for x in obj:
            yield x


def extract_behavior_code(data: Any) -> str | None:
    """从任意 JSON 结构中尽量鲁棒地提取行为编码（dx/dk/tt/zt/js/zl/xt/jz）。"""
    stack = [data]
    visited = set()

    while stack:
        cur = stack.pop()
        cur_id = id(cur)
        if cur_id in visited:
            continue
        visited.add(cur_id)

        if isinstance(cur, str):
            code = _extract_code_from_string(cur)
            if code:
                return code
            continue

        if isinstance(cur, (int, float, bool)) or cur is None:
            continue

        if isinstance(cur, dict):
            # 兼容 flags = {"dx": true} 这类结构
            for k, v in cur.items():
                key_code = _extract_code_from_string(str(k))
                if key_code and bool(v):
                    return key_code
            for v in _iter_values(cur):
                stack.append(v)
            continue

        if isinstance(cur, list):
            for v in cur:
                stack.append(v)

    return None