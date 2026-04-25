import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    input: str = ""
    output: str = ""
    segment_level: int = 2
    stdin: bool = False
    format: str = "txt"
    model: str = "deepseek-chat"
    chunk_size: int = 2000
    overlap_size: int = 120
    timeout: int = 60
    retries: int = 3

    def __init__(self, args: argparse.Namespace):
        self.input = str(getattr(args, "input", self.input) or "")
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
            core_text = text[cursor:core_end]
            context_before = text[max(0, cursor - self.overlap_size):cursor]
            context_after = text[core_end:min(text_len, core_end + self.overlap_size)]
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
    def _longest_overlap(left: str, right: str, max_check: int = 80) -> int:
        limit = min(len(left), len(right), max_check)
        for size in range(limit, 0, -1):
            if left[-size:] == right[:size]:
                return size
        return 0

    def merge(self, parts: List[str]) -> str:
        if not parts:
            return ""
        merged = parts[0]
        for part in parts[1:]:
            overlap = self._longest_overlap(merged, part)
            merged += part[overlap:]
        return merged


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

    def format_text(self, text: str, segment_level: int) -> Dict[str, object]:
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
        for chunk in tqdm(chunks, desc="分片转换进度", unit="片"):
            messages = self.prompt_builder.build_messages(chunk=chunk, segment_level=segment_level)
            formatted = self.client.chat_completion(messages)
            normalized = self._normalize_output(formatted)
            if not normalized:
                raise RuntimeError(f"第 {chunk.index + 1} 个分片返回空内容，已中止。")
            formatted_parts.append(normalized)

        formatted_text = self.merger.merge(formatted_parts)
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

    def convert_text(self, text: str, segment_level: int, output_format: str) -> Dict[str, object]:
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

        result = self.service.format_text(text=text, segment_level=segment_level)
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
    ) -> Path:
        if not input_file:
            raise ValueError("未提供输入文件，请使用 --input 或 --stdin。")
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在：{input_path}")
        content = input_path.read_text(encoding="utf-8")
        text = content
        result = self.convert_text(text=text, segment_level=segment_level, output_format=output_format)
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
    parser.add_argument("--input", type=str, help="输入文本文件路径。")
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
        else:
            output_path = runner.convert_file(
                input_file=options.input,
                segment_level=options.segment_level,
                output_format=options.format,
                output_file=options.output,
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
