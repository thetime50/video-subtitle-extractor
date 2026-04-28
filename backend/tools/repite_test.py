'''
0. 输入参数为文件目录
1. 按中英文标点 空白字符分割为数组
2. 找出数组里重复的分段
3. 标记连续重复分段的起始位置，并按连续分组，忽略总字符长度<5的分小组

python backend/tools/repite_test.py "F:/live/know/渤海小吏/三国争霸/26 深度丨丨 入西川！二士争功！！三英皆授首！！！.fmt.txt"
'''

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SPLIT_PATTERN = re.compile(r'[\s,，.。!?！？;；:：、"“”\'‘’()（）\[\]{}<>《》]+')


def split_segments(text: str) -> list[str]:
    return [segment for segment in SPLIT_PATTERN.split(text) if segment]


def find_repeated_groups(segments: list[str]) -> list[dict]:
    token_indexes: dict[str, list[int]] = {}
    for idx, token in enumerate(segments):
        token_indexes.setdefault(token, []).append(idx)

    repeated_indexes = sorted(
        index for indexes in token_indexes.values() if len(indexes) > 1 for index in indexes
    )

    groups: list[dict] = []
    if not repeated_indexes:
        return groups

    current = [repeated_indexes[0]]
    for index in repeated_indexes[1:]:
        if index == current[-1] + 1:
            current.append(index)
        else:
            groups.append(_build_group(current, segments))
            current = [index]
    groups.append(_build_group(current, segments))
    return [group for group in groups if group["total_char_length"] >= 5]


def _build_group(indexes: list[int], segments: list[str]) -> dict:
    tokens = [segments[index] for index in indexes]
    return {
        "start_index": indexes[0],
        # "indexes": indexes,
        "tokens": tokens,
        "total_char_length": sum(len(token) for token in tokens),
    }


def analyze_file(file_path: Path) -> dict:
    content = file_path.read_text(encoding="utf-8")
    segments = split_segments(content)
    groups = find_repeated_groups(segments)
    return {
        "file": str(file_path),
        "segment_count": len(segments),
        "repeated_groups": groups,
    }


def analyze_directory(directory: Path) -> dict:
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"目录不存在或不是目录: {directory}")

    results = []
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".srt", ".ass"}:
            try:
                results.append(analyze_file(file_path))
            except UnicodeDecodeError:
                continue
    return {"directory": str(directory), "results": results}


def analyze_path(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    if path.is_file():
        if path.suffix.lower() not in {".txt", ".srt", ".ass"}:
            return {"file": str(path), "results": []}
        return {"file": str(path), "results": [analyze_file(path)]}
    return analyze_directory(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="文件目录")
    args = parser.parse_args()
    output = analyze_path(Path(args.directory))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
