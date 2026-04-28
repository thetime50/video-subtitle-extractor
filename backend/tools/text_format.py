import argparse
import difflib
import json
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import traceback


@dataclass
class Chunk:
    index: int
    core_text: str
    context_before: str
    context_after: str


@dataclass
class FormatterOptions:
    input: List[str] = field(default_factory=list)
    output: str = ""
    segment_level: int = 2
    stdin: bool = False
    format: str = "txt"
    model: str = "deepseek-chat"
    chunk_size: int = 2000
    overlap_size: int = 10
    timeout: int = 60
    retries: int = 3

    def __init__(self, args: argparse.Namespace):
        raw_input = getattr(args, "input", [])
        if raw_input is None:
            self.input = []
        elif isinstance(raw_input, str):
            self.input = [raw_input]
        else:
            self.input = [str(item) for item in raw_input if str(item).strip()]
        self.output = str(getattr(args, "output", self.output) or "")
        self.segment_level = int(getattr(args, "segment_level", self.segment_level))
        self.stdin = bool(getattr(args, "stdin", self.stdin))
        self.format = str(getattr(args, "format", self.format) or self.format)
        self.model = str(getattr(args, "model", self.model) or self.model)
        self.chunk_size = int(getattr(args, "chunk_size", self.chunk_size))
        self.overlap_size = int(getattr(args, "overlap_size", self.overlap_size))
        self.timeout = int(getattr(args, "timeout", self.timeout))
        self.retries = int(getattr(args, "retries", self.retries))


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 60,
        retries: int = 3,
    ):
        if not api_key:
            raise ValueError("缺少 API Key，请在 backend/private_config.json 中配置 deepseek_api_key。")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url}/chat/completions"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(
                url=url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                result = json.loads(raw)
                content = result["choices"][0]["message"]["content"]
                text = str(content).strip()
                if not text:
                    raise ValueError("模型返回为空。")
                return text
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as error:
                last_error = error
                if attempt == self.retries:
                    break
                sleep_seconds = min(2 ** (attempt - 1), 8)
                time.sleep(sleep_seconds)

        raise RuntimeError(f"调用 DeepSeek 失败：{last_error}")


class PromptBuilder:
    @staticmethod
    def _segment_instruction(level: int) -> str:
        mapping = {
            1: "分段较粗，只有明显主题切换时才换段，尽量保持长段落。",
            2: "分段适中，根据语义转折自然分段，避免碎片化。",
            3: "分段较细，但仍然避免频繁换段，保持阅读连贯。",
        }
        return mapping.get(level, mapping[2])

    def build_messages(
        self,
        chunk: Chunk,
        segment_level: int,
    ) -> List[Dict[str, str]]:
        system_prompt = (
            "你是字幕文本整理助手。你的任务是给没有标点、仅依赖换行的中文文本补充标点并分段。"
            "必须遵守：1) 不改动原始字词内容和顺序；2) 只允许新增标点和段落边界；"
            "3) 段落内不换行，句子必须连接成一行；4) 输出只包含整理后的核心文本，不要解释。"
        )
        user_prompt = (
            f"请按以下规则处理“核心文本”：\n"
            f"- 分段要求：{self._segment_instruction(segment_level)}\n"
            f"- 段内不要换行，段落之间使用单个换行分隔。\n"
            f"- 不要输出上下文内容，只输出核心文本的结果。\n\n"
            f"【前文上下文】\n{chunk.context_before}\n\n"
            f"【核心文本】\n{chunk.core_text}\n\n"
            f"【后文上下文】\n{chunk.context_after}\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


class TextChunker:
    def __init__(self, chunk_size: int, overlap_size: int):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0。")
        if overlap_size < 0:
            raise ValueError("overlap_size 不能小于 0。")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size 必须小于 chunk_size。")
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def split(self, text: str) -> List[Chunk]:
        if not text:
            return []
        chunks: List[Chunk] = []
        cursor = 0
        index = 0
        text_len = len(text)
        while cursor < text_len:
            core_end = min(cursor + self.chunk_size, text_len)
            if core_end < text_len:
                newline_pos = text.rfind("\n", cursor, core_end)
                if newline_pos != -1:
                    core_end = newline_pos + 1
            core_text = text[cursor:core_end]
            before_base_start = max(0, cursor - self.overlap_size)
            before_newline = text.rfind("\n", 0, before_base_start)
            context_before_start = 0 if before_newline == -1 else before_newline + 1
            context_before = text[context_before_start:cursor]

            after_base_end = min(text_len, core_end + self.overlap_size)
            after_newline = text.find("\n", after_base_end)
            context_after_end = text_len if after_newline == -1 else after_newline + 1
            context_after = text[core_end:context_after_end]
            chunks.append(
                Chunk(
                    index=index,
                    core_text=core_text,
                    context_before=context_before,
                    context_after=context_after,
                )
            )
            index += 1
            cursor = core_end
        return chunks


class ChunkMerger:
    @staticmethod
    def _is_separator(char: str) -> bool:
        if char.isspace():
            return True
        return bool(re.match(r"[，。！？；：、,.!?;:\"'“”‘’()（）【】《》〈〉\[\]{}\-—_…]", char))

    @staticmethod
    def _normalize_with_mapping(text: str) -> tuple[str, List[int]]:
        normalized_chars: List[str] = []
        index_mapping: List[int] = []
        for index, char in enumerate(text):
            if ChunkMerger._is_separator(char):
                if normalized_chars and normalized_chars[-1] == " ":
                    continue
                normalized_chars.append(" ")
                index_mapping.append(index)
                continue
            normalized_chars.append(char.lower())
            index_mapping.append(index)
        return "".join(normalized_chars), index_mapping

    @staticmethod
    def _approx_overlap_by_context(
        left: str,
        right: str,
        context_after: str,
        context_before: str,
    ) -> int:
        base = max(len(context_after), len(context_before))
        if base <= 0:
            return 0
        window_size = max(1, int(base * 1.2))
        left_start = max(0, len(left) - window_size)
        right_end = min(len(right), window_size)
        left_window = left[left_start:]
        right_window = right[:right_end]
        if not left_window or not right_window:
            return 0

        normalized_left, left_map = ChunkMerger._normalize_with_mapping(left_window)
        normalized_right, right_map = ChunkMerger._normalize_with_mapping(right_window)
        if not normalized_left or not normalized_right:
            return 0

        limit = min(len(normalized_left), len(normalized_right))
        for size in range(limit, 0, -1):
            left_suffix = normalized_left[-size:]
            right_prefix = normalized_right[:size]
            left_compact = left_suffix.strip()
            right_compact = right_prefix.strip()
            if not left_compact or not right_compact:
                continue
            ratio = difflib.SequenceMatcher(None, left_suffix, right_prefix).ratio()
            if ratio >= 0.9:
                right_end_index = right_map[size - 1] + 1
                return min(right_end_index, len(right))
        return 0

    @staticmethod
    def _longest_overlap(left: str, right: str, max_check: int = 80) -> int:
        limit = min(len(left), len(right), max_check)
        for size in range(limit, 0, -1):
            if left[-size:] == right[:size]:
                return size
        return 0

    def merge(self, parts: List[str], content_after: List[str], content_before: List[str]) -> str:
        if not parts:
            return ""
        merged = parts[0]
        for index, part in enumerate(parts[1:], start=1):
            overlap = self._approx_overlap_by_context(
                left=merged,
                right=part,
                context_after=content_after[index - 1] if index - 1 < len(content_after) else "",
                context_before=content_before[index] if index < len(content_before) else "",
            )
            if overlap <= 0:
                overlap = self._longest_overlap(merged, part)
            merged += part[overlap:]
        # 完成合并后把合并边界的8个字符打印出来
        boundary_text = ' | '.join([item[:8] for item in parts ])
        print("边界：",boundary_text)
        return merged


class ConcurrentChatCompletionManager:
    def __init__(
        self,
        client: DeepSeekClient,
        concurrency: int = 3,
        max_attempts: int = 2,
        error_threshold: int = 3,
        cooldown_seconds: int = 3,
    ):
        if concurrency <= 0:
            raise ValueError("concurrency 必须大于 0。")
        self.client = client
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.error_threshold = error_threshold
        self.cooldown_seconds = cooldown_seconds

    def run(self, message_batches: List[List[Dict[str, str]]], progress_desc: str = "分片转换进度") -> List[str]:
        if not message_batches:
            return []

        results: List[Optional[str]] = [None] * len(message_batches)
        pending_indices = list(range(len(message_batches)))
        attempt = 1
        completed = set()
        errors: Dict[int, str] = {}

        with tqdm(total=len(message_batches), desc=progress_desc, unit="片") as bar:
            while pending_indices and attempt <= self.max_attempts:
                failed_indices: List[int] = []
                consecutive_errors = 0
                task_queue: queue.Queue[int] = queue.Queue()
                result_queue: queue.Queue = queue.Queue()
                for index in pending_indices:
                    task_queue.put(index)

                def worker() -> None:
                    while True:
                        try:
                            index = task_queue.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            text = self.client.chat_completion(message_batches[index])
                            result_queue.put((index, True, text))
                        except Exception as error:
                            result_queue.put((index, False, str(error)))
                        finally:
                            task_queue.task_done()

                worker_count = min(self.concurrency, len(pending_indices))
                threads: List[threading.Thread] = []
                for _ in range(worker_count):
                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    threads.append(thread)

                processed_count = 0
                total_count = len(pending_indices)
                while processed_count < total_count:
                    index, is_success, payload = result_queue.get()
                    processed_count += 1
                    if is_success:
                        results[index] = str(payload)
                        errors.pop(index, None)
                        consecutive_errors = 0
                        if index not in completed:
                            completed.add(index)
                            bar.update(1)
                    else:
                        failed_indices.append(index)
                        errors[index] = str(payload)
                        consecutive_errors += 1
                        if consecutive_errors >= self.error_threshold:
                            time.sleep(self.cooldown_seconds)
                            consecutive_errors = 0

                for thread in threads:
                    thread.join()

                if not failed_indices:
                    break
                if attempt >= self.max_attempts:
                    break
                time.sleep(self.cooldown_seconds)
                pending_indices = failed_indices
                attempt += 1

        remaining = [index for index, value in enumerate(results) if not value]
        if remaining:
            failed_text = ", ".join(str(index + 1) for index in remaining)
            detail = "; ".join(f"分片{index + 1}:{errors.get(index, '未知错误')}" for index in remaining)
            raise RuntimeError(
                f"并发请求失败，已达到最大尝试次数({self.max_attempts})。失败分片：{failed_text}。{detail}"
            )
        return [value for value in results if value is not None]


class SubtitleFormatterService:
    def __init__(self, client: DeepSeekClient, prompt_builder: PromptBuilder, chunker: TextChunker, merger: ChunkMerger):
        self.client = client
        self.prompt_builder = prompt_builder
        self.chunker = chunker
        self.merger = merger

    @staticmethod
    def _normalize_output(text: str) -> str:
        lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
        paragraphs: List[str] = []
        for line in lines:
            if line:
                paragraphs.append(line)
        return "\n".join(paragraphs)

    def format_text(self, text: str, segment_level: int, concurrency: int = 5, progress_info: str = "") -> Dict[str, object]:
        original_text = text
        chunks = self.chunker.split(text)
        if not chunks:
            return {
                "original_text": original_text,
                "formatted_text": "",
                "meta": {
                    "chunk_count": 0,
                    "segment_level": segment_level,
                    "model": self.client.model,
                },
            }

        formatted_parts: List[str] = []
        start_time = time.time()
        message_batches: List[List[Dict[str, str]]] = []
        for chunk in chunks:
            message_batches.append(self.prompt_builder.build_messages(chunk=chunk, segment_level=segment_level))
        manager = ConcurrentChatCompletionManager(client=self.client, concurrency=concurrency)
        raw_results = manager.run(message_batches=message_batches, progress_desc=f"进度 {progress_info}")
        for chunk, formatted in zip(chunks, raw_results):
            normalized = self._normalize_output(formatted)
            if not normalized:
                raise RuntimeError(f"第 {chunk.index + 1} 个分片返回空内容，已中止。")
            formatted_parts.append(normalized)

        content_after_list = [chunk.context_after for chunk in chunks]
        content_before_list = [chunk.context_before for chunk in chunks]
        formatted_text = self.merger.merge(
            parts=formatted_parts,
            content_after=content_after_list,
            content_before=content_before_list,
        )
        elapsed_seconds = round(time.time() - start_time, 3)
        return {
            "original_text": original_text,
            "formatted_text": formatted_text,
            "meta": {
                "chunk_count": len(chunks),
                "segment_level": segment_level,
                "model": self.client.model,
                "chunk_size": self.chunker.chunk_size,
                "overlap_size": self.chunker.overlap_size,
                "elapsed_seconds": elapsed_seconds,
            },
        }


class SubtitleFormatterRunner:
    def __init__(
        self,
        model: str = "deepseek-chat",
        chunk_size: int = 2000,
        overlap_size: int = 120,
        timeout: int = 60,
        retries: int = 3,
    ):
        api_key = load_private_api_key()
        client = DeepSeekClient(
            api_key=api_key,
            model=model,
            base_url="https://api.deepseek.com",
            timeout=timeout,
            retries=retries,
        )
        prompt_builder = PromptBuilder()
        chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)
        merger = ChunkMerger()
        self.service = SubtitleFormatterService(client, prompt_builder, chunker, merger)

    def resolve_output_path(self, output_file: str, use_stdin: bool, output_format: str, input_file: str) -> Path:
        if output_file:
            return Path(output_file)

        if use_stdin:
            suffix = "json" if output_format == "json" else "txt"
            return Path(f"formatted_output.{suffix}")

        input_path = Path(input_file)
        if output_format == "json":
            return input_path.with_name(f"{input_path.stem}.fmt.json")
        return input_path.with_name(f"{input_path.stem}.fmt.txt")

    def write_output(self, result: Dict[str, object], output_path: Path, output_format: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_format == "json":
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            output_path.write_text(str(result["formatted_text"]), encoding="utf-8")

    def convert_text(self, text: str, segment_level: int, output_format: str, progress_info: str = "") -> Dict[str, object]:
        first_ten_lines = text
        line_count = 0
        for index, char in enumerate(text):
            if char == "\n":
                line_count += 1
                if line_count >= 10:
                    first_ten_lines = text[:index + 1]
                    break

        srt_pattern = r"\n?\n?\d+\n\d\d:\d\d:\d\d,\d\d\d --> \d\d:\d\d:\d\d,\d\d\d"
        if re.search(srt_pattern, first_ten_lines):
            text = re.sub(srt_pattern, "", text)

        result = self.service.format_text(text=text, segment_level=segment_level, progress_info=progress_info)
        if output_format == "json":
            return result
        return {
            "original_text": result["original_text"],
            "formatted_text": result["formatted_text"],
            "meta": result["meta"],
        }

    def convert_file(
        self,
        input_file: str,
        segment_level: int,
        output_format: str,
        output_file: str = "",
        progress_info: str = "",
    ) -> Path:
        if not input_file:
            raise ValueError("未提供输入文件，请使用 --input 或 --stdin。")
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在：{input_path}")
        content = input_path.read_text(encoding="utf-8")
        text = content
        result = self.convert_text(text=text, segment_level=segment_level, output_format=output_format, progress_info=progress_info)
        output_path = self.resolve_output_path(
            output_file=output_file,
            use_stdin=False,
            output_format=output_format,
            input_file=input_file,
        )
        self.write_output(result=result, output_path=output_path, output_format=output_format)
        return output_path

    def convert_stdin(self, segment_level: int, output_format: str, output_file: str = "") -> Path:
        content = sys.stdin.read()
        if not content:
            raise ValueError("标准输入为空。")
        text =  content
        result = self.convert_text(text=text, segment_level=segment_level, output_format=output_format)
        output_path = self.resolve_output_path(
            output_file=output_file,
            use_stdin=True,
            output_format=output_format,
            input_file="",
        )
        self.write_output(result=result, output_path=output_path, output_format=output_format)
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 DeepSeek 为字幕文本自动添加标点并分段。",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42, width=120),
    )
    parser.add_argument("--input", type=str, nargs="+", help="输入路径列表（文件或目录）。")
    parser.add_argument("--output", type=str, help="输出文件路径。")
    parser.add_argument("--segment-level", type=int, default=2, choices=[1, 2, 3], help="分段等级：1-粗 2-中 3-细。")
    parser.add_argument("--stdin", action="store_true", help="从标准输入读取文本。")
    parser.add_argument("--format", choices=["json", "txt"], default="txt", help="输出格式。")

    advanced_group = parser.add_argument_group("----------------------------------------")
    advanced_group.add_argument("--model", type=str, default="deepseek-chat", help="模型名称。")
    advanced_group.add_argument("--chunk-size", type=int, default=2000, help="单分片字符数上限。")
    advanced_group.add_argument("--overlap-size", type=int, default=120, help="分片上下文重叠字符数。")
    advanced_group.add_argument("--timeout", type=int, default=60, help="单次 API 调用超时秒数。")
    advanced_group.add_argument("--retries", type=int, default=3, help="API 调用失败重试次数。")
    return parser.parse_args()


def load_private_api_key() -> str:
    config_path = Path(__file__).resolve().parents[2] / "config" / "private_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"私密配置文件不存在：{config_path}")

    try:
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"私密配置文件 JSON 格式错误：{error}") from error

    api_key = str(config_data.get("deepseek_api_key", "")).strip()
    if not api_key:
        raise ValueError(f"请在 {config_path} 中配置 deepseek_api_key。")
    return api_key


def collect_input_files(input_items: List[str]) -> List[Path]:
    file_paths: List[Path] = []
    for item in input_items:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"输入路径不存在：{path}")
        if path.is_file():
            file_paths.append(path)
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                # 过滤 允许.srt .txt 排除 .fmt.txt
                if child.is_file() and child.suffix in ['.srt', '.txt'] and child.suffix != '.fmt.txt':
                    file_paths.append(child)
            continue
        raise ValueError(f"不支持的输入路径类型：{path}")

    # 去重并保持顺序
    unique_files: List[Path] = []
    seen = set()
    for path in file_paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_files.append(path)
    return unique_files


def build_output_paths(
    input_files: List[Path],
    output_arg: str,
    output_format: str,
    runner: SubtitleFormatterRunner,
) -> List[Path]:
    if not input_files:
        raise ValueError("未找到可处理的输入文件。")

    if len(input_files) == 1:
        if not output_arg:
            input_file = str(input_files[0])
            output_path = runner.resolve_output_path(
                output_file=output_arg,
                use_stdin=False,
                output_format=output_format,
                input_file=input_file,
            )
            return [output_path]
        else:
            return [output_arg]

    if output_arg:
        output_base = Path(output_arg)
        if output_base.exists() and not output_base.is_dir():
            raise ValueError("当输入文件数量大于1时，--output 必须为空或目录路径。")
    else:
        output_base = Path("")

    ext = "json" if output_format == "json" else "txt"
    output_paths: List[Path] = []
    for input_file in input_files:
        if output_arg:
            output_paths.append(output_base / f"{input_file.stem}.fmt.{ext}")
        else:
            output_paths.append(
                runner.resolve_output_path(
                    output_file="",
                    use_stdin=False,
                    output_format=output_format,
                    input_file=str(input_file),
                )
            )
    return output_paths


def main() -> int:
    try:
        options = FormatterOptions(parse_args())
        runner = SubtitleFormatterRunner(
            model=options.model,
            chunk_size=options.chunk_size,
            overlap_size=options.overlap_size,
            timeout=options.timeout,
            retries=options.retries,
        )
        if options.stdin:
            output_path = runner.convert_stdin(
                segment_level=options.segment_level,
                output_format=options.format,
                output_file=options.output,
            )
            print(f"处理完成，输出文件：{output_path}")
        else:
            input_files = collect_input_files(options.input)
            output_paths = build_output_paths(
                input_files=input_files,
                output_arg=options.output,
                output_format=options.format,
                runner=runner,
            )
            for index, (input_file, output_path) in enumerate(zip(input_files, output_paths)):
                runner.convert_file(
                    input_file=str(input_file),
                    segment_level=options.segment_level,
                    output_format=options.format,
                    output_file=str(output_path),
                    # 进度条信息
                    progress_info=f"{index + 1}/{len(input_files)}"
                )
                print(f"处理完成，输出文件：{output_path}")
        return 0
    except Exception as error:
        print(f"处理失败：{error}", file=sys.stderr)
        # 打印堆栈跟踪
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
