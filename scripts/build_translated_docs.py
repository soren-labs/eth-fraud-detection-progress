from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from markdownify import markdownify as html_to_markdown


ROOT = Path(__file__).resolve().parent.parent
DOC_ROOT = ROOT / "xgboost-document"
SOURCE_DIR = DOC_ROOT / "source_html"
CACHE_FILE = DOC_ROOT / ".translation_cache.json"
VERIFY_FILE = DOC_ROOT / "verification_report.md"
README_FILE = DOC_ROOT / "README.md"


THEMES = {
    "01_scikit_learn_basics": "Scikit-learn 基础与数据划分",
    "02_model_evaluation": "模型评估",
    "03_tree_and_ensemble": "决策树与集成学习",
    "04_preprocessing_and_pipeline": "预处理与 Pipeline",
    "05_xgboost_overview": "XGBoost 概览与接口",
    "06_xgboost_parameters_and_tuning": "XGBoost 参数与调优",
}


@dataclass(frozen=True)
class DocSpec:
    index: str
    source_name: str
    source_url: str
    theme: str
    title_zh: str


DOCS = [
    DocSpec(
        index="01",
        source_name="01_scikit_learn_getting_started.html",
        source_url="https://scikit-learn.org/stable/getting_started.html",
        theme="01_scikit_learn_basics",
        title_zh="Scikit-learn 入门",
    ),
    DocSpec(
        index="02",
        source_name="02_train_test_split.html",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
        theme="01_scikit_learn_basics",
        title_zh="数据划分：train_test_split",
    ),
    DocSpec(
        index="03",
        source_name="03_model_evaluation_classification.html",
        source_url="https://scikit-learn.org/stable/modules/model_evaluation.html",
        theme="02_model_evaluation",
        title_zh="模型评估（分类任务）",
    ),
    DocSpec(
        index="04",
        source_name="04_classification_report.html",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html",
        theme="02_model_evaluation",
        title_zh="分类报告：classification_report",
    ),
    DocSpec(
        index="05",
        source_name="05_decision_trees.html",
        source_url="https://scikit-learn.org/stable/modules/tree.html",
        theme="03_tree_and_ensemble",
        title_zh="决策树",
    ),
    DocSpec(
        index="06",
        source_name="06_plot_unveil_tree_structure.html",
        source_url="https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html",
        theme="03_tree_and_ensemble",
        title_zh="决策树结构示例",
    ),
    DocSpec(
        index="07",
        source_name="07_ensemble.html",
        source_url="https://scikit-learn.org/stable/modules/ensemble.html",
        theme="03_tree_and_ensemble",
        title_zh="随机森林与集成学习",
    ),
    DocSpec(
        index="08",
        source_name="08_random_forest_classifier.html",
        source_url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
        theme="03_tree_and_ensemble",
        title_zh="RandomForestClassifier API",
    ),
    DocSpec(
        index="09",
        source_name="09_preprocessing.html",
        source_url="https://scikit-learn.org/stable/modules/preprocessing.html",
        theme="04_preprocessing_and_pipeline",
        title_zh="预处理",
    ),
    DocSpec(
        index="10",
        source_name="10_compose_pipeline.html",
        source_url="https://scikit-learn.org/stable/modules/compose.html",
        theme="04_preprocessing_and_pipeline",
        title_zh="Pipeline 与组合估计器",
    ),
    DocSpec(
        index="11",
        source_name="11_common_pitfalls.html",
        source_url="https://scikit-learn.org/stable/common_pitfalls.html",
        theme="04_preprocessing_and_pipeline",
        title_zh="常见陷阱与避免数据泄漏",
    ),
    DocSpec(
        index="12",
        source_name="12_xgboost_stable_index.html",
        source_url="https://xgboost.readthedocs.io/en/stable/",
        theme="05_xgboost_overview",
        title_zh="XGBoost 官方总入口（Stable）",
    ),
    DocSpec(
        index="13",
        source_name="13_xgboost_model_tutorial.html",
        source_url="https://xgboost.readthedocs.io/en/stable/tutorials/model.html",
        theme="05_xgboost_overview",
        title_zh="XGBoost：Boosted Trees 原理",
    ),
    DocSpec(
        index="14",
        source_name="14_xgboost_python_index.html",
        source_url="https://xgboost.readthedocs.io/en/stable/python/",
        theme="05_xgboost_overview",
        title_zh="XGBoost：Python 包总入口",
    ),
    DocSpec(
        index="15",
        source_name="15_xgboost_sklearn_estimator.html",
        source_url="https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html",
        theme="05_xgboost_overview",
        title_zh="XGBoost：Scikit-Learn Estimator 接口",
    ),
    DocSpec(
        index="16",
        source_name="16_xgboost_parameter.html",
        source_url="https://xgboost.readthedocs.io/en/stable/parameter.html",
        theme="06_xgboost_parameters_and_tuning",
        title_zh="XGBoost：参数说明",
    ),
    DocSpec(
        index="17",
        source_name="17_xgboost_treemethod.html",
        source_url="https://xgboost.readthedocs.io/en/stable/treemethod.html",
        theme="06_xgboost_parameters_and_tuning",
        title_zh="XGBoost：Tree Methods",
    ),
    DocSpec(
        index="18",
        source_name="18_xgboost_param_tuning.html",
        source_url="https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html",
        theme="06_xgboost_parameters_and_tuning",
        title_zh="XGBoost：参数调优说明",
    ),
]


CLEAN_SELECTORS = [
    ".headerlink",
    ".copybtn",
    "script",
    "style",
    "nav",
    ".sphx-glr-thumbcontainer",
    ".sphx-glr-thumbnails",
    ".sphx-glr-footer",
    ".search",
]

FENCED_CODE_RE = re.compile(r"(^```.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
RAW_URL_RE = re.compile(r"https?://[^\s)>]+")
PLACEHOLDER_RE = re.compile(r"QX[A-Z]+[0-9]+QX")
RAW_HTML_TAG_RE = re.compile(r"<[A-Za-z][^>]*>")


class Translator:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache = self._load_cache()
        self.backend = GoogleTranslator(source="auto", target="zh-CN")

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def save(self) -> None:
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def translate(self, text: str) -> str:
        if not text.strip():
            return text
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for _ in range(5):
            try:
                translated = self.backend.translate(text)
                self.cache[key] = translated
                return translated
            except Exception as exc:  # pragma: no cover - network retry
                last_error = exc
                time.sleep(1.5)
        raise RuntimeError(f"Translation failed after retries: {last_error}") from last_error


def ensure_source_layout() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    for spec in DOCS:
        root_source = DOC_ROOT / spec.source_name
        target_source = SOURCE_DIR / spec.source_name
        if root_source.exists() and not target_source.exists():
            shutil.move(str(root_source), str(target_source))

    root_readme = DOC_ROOT / "README.md"
    source_readme = SOURCE_DIR / "README_download_index.md"
    if root_readme.exists() and not source_readme.exists():
        existing = root_readme.read_text(encoding="utf-8")
        if "# 文档索引" in existing and "本目录包含" not in existing:
            shutil.move(str(root_readme), str(source_readme))


def read_html(spec: DocSpec) -> str:
    source_path = SOURCE_DIR / spec.source_name
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source HTML: {source_path}")
    return source_path.read_text(encoding="utf-8")


def find_article(soup: BeautifulSoup) -> BeautifulSoup:
    selectors = [
        "article.bd-article",
        "div[role='main'] div[itemprop='articleBody']",
        "div[role='main']",
        "article",
        "main",
    ]
    for selector in selectors:
        article = soup.select_one(selector)
        if article is not None:
            return article
    raise ValueError("Failed to locate article body in HTML document.")


def clean_article_html(raw_html: str, source_url: str) -> str:
    soup = BeautifulSoup(raw_html, "lxml")
    article = find_article(soup)

    for selector in CLEAN_SELECTORS:
        for tag in article.select(selector):
            tag.decompose()

    for tag in article.select("[title]"):
        tag.attrs.pop("title", None)

    for anchor in article.select("a[href]"):
        anchor["href"] = urljoin(source_url, anchor["href"])

    for image in article.select("img[src]"):
        image["src"] = urljoin(source_url, image["src"])

    return str(article)


def markdown_from_html(cleaned_html: str) -> str:
    markdown = html_to_markdown(
        cleaned_html,
        heading_style="ATX",
        bullets="-",
        strip=["span"],
    )
    markdown = html.unescape(markdown)
    markdown = markdown.replace("\r\n", "\n")
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip() + "\n"


def localize_images(markdown: str, doc_dir: Path, assets_dir_name: str) -> str:
    assets_dir = doc_dir / assets_dir_name
    assets_dir.mkdir(parents=True, exist_ok=True)

    def replace(match: re.Match[str]) -> str:
        alt_text = match.group(1)
        url = match.group(2)
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return match.group(0)

        file_name = Path(parsed.path).name
        if not file_name:
            return match.group(0)

        target_path = assets_dir / file_name
        if not target_path.exists():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            target_path.write_bytes(response.content)

        relative = f"./{assets_dir_name}/{file_name}"
        return f"![{alt_text}]({relative})"

    return re.sub(r"!\[([^\]]*)\]\((https?://[^)]+)\)", replace, markdown)


def protect_fragments(text: str) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    counter = {"CODE": 0, "URL": 0}

    def protect(pattern: re.Pattern[str], token_type: str, value_group: int | None = None) -> None:
        def substitute(match: re.Match[str]) -> str:
            key = f"QX{token_type}{counter[token_type]}QX"
            counter[token_type] += 1
            mapping[key] = match.group(0) if value_group is None else match.group(value_group)
            if value_group is None:
                return key
            start, end = match.span(value_group)
            full = match.group(0)
            relative_start = start - match.start()
            relative_end = end - match.start()
            return full[:relative_start] + key + full[relative_end:]

        nonlocal text
        text = pattern.sub(substitute, text)

    protect(INLINE_CODE_RE, "CODE")
    protect(RAW_URL_RE, "URL")
    return text, mapping


def restore_fragments(text: str, mapping: dict[str, str]) -> str:
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text


def translate_text_chunked(text: str, translator: Translator, limit: int = 3800) -> str:
    if len(text) <= limit:
        return translator.translate(text)

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current = ""
    for line in lines:
        if current and len(current) + len(line) > limit:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return "".join(translator.translate(chunk) for chunk in chunks)


def translate_markdown(markdown: str, translator: Translator) -> str:
    parts = FENCED_CODE_RE.split(markdown)
    translated_parts: list[str] = []

    for index, part in enumerate(parts):
        if not part:
            continue
        if index % 2 == 1:
            translated_parts.append(part)
            continue

        blocks = re.split(r"(\n\s*\n)", part)
        for block in blocks:
            if re.fullmatch(r"\n\s*\n", block):
                translated_parts.append(block)
                continue
            if not block.strip():
                translated_parts.append(block)
                continue

            protected, mapping = protect_fragments(block)
            translated = translate_text_chunked(protected, translator)
            translated_parts.append(restore_fragments(translated, mapping))

    translated_markdown = "".join(translated_parts)
    translated_markdown = translated_markdown.replace("\r\n", "\n")
    translated_markdown = re.sub(r"\n{3,}", "\n\n", translated_markdown)
    return translated_markdown.strip() + "\n"


def heading_count(text: str) -> int:
    return len(re.findall(r"^#{1,6} ", text, flags=re.MULTILINE))


def code_block_count(text: str) -> int:
    return len(re.findall(r"^```", text, flags=re.MULTILINE)) // 2


def inline_code_count(text: str) -> int:
    return len(INLINE_CODE_RE.findall(text))


def link_count(text: str) -> int:
    return len(re.findall(r"\[[^\]]+\]\([^)]+\)", text))


def image_count(text: str) -> int:
    return len(re.findall(r"!\[[^\]]*\]\([^)]+\)", text))


def raw_html_count(text: str) -> int:
    return len(RAW_HTML_TAG_RE.findall(text))


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return cjk / max(len(text), 1)


def build_front_matter(
    *,
    title: str,
    source_url: str,
    source_html_relative: str,
    sibling_relative: str,
    note: str,
) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            f"- 来源链接: [{source_url}]({source_url})",
            f"- 源 HTML: [{source_html_relative}]({source_html_relative})",
            f"- 对照文档: [{sibling_relative}]({sibling_relative})",
            f"- 说明: {note}",
            "",
        ]
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def relative_link(from_path: Path, to_path: Path) -> str:
    return Path(os.path.relpath(to_path, start=from_path.parent)).as_posix()


def generate_theme_readmes(theme_map: dict[str, list[tuple[DocSpec, Path, Path]]]) -> None:
    for theme, docs in theme_map.items():
        theme_dir = DOC_ROOT / theme
        lines = [f"# {THEMES[theme]}", "", "本目录包含该主题下的英文 Markdown 对照稿与中文 Markdown 译稿。", ""]
        for spec, en_path, zh_path in docs:
            lines.append(f"## {spec.index}. {spec.title_zh}")
            lines.append(f"- 中文译稿: [{zh_path.name}]({zh_path.name})")
            lines.append(f"- 英文对照: [{en_path.name}]({en_path.name})")
            lines.append(f"- 来源: [{spec.source_url}]({spec.source_url})")
            lines.append("")
        write_text(theme_dir / "README.md", "\n".join(lines).strip() + "\n")


def generate_root_readme(theme_map: dict[str, list[tuple[DocSpec, Path, Path]]]) -> None:
    lines = [
        "# xgboost-document",
        "",
        "本目录包含按主题整理后的 scikit-learn 与 XGBoost 文档。",
        "",
        "- `source_html/`: 原始下载的 HTML 文档",
        "- `verification_report.md`: 自动化结构校验结果",
        "",
    ]
    for theme, docs in theme_map.items():
        lines.append(f"## {THEMES[theme]}")
        lines.append(f"- 主题目录: [{theme}/README.md]({theme}/README.md)")
        for spec, _, zh_path in docs:
            lines.append(f"- {spec.index}. [{spec.title_zh}]({theme}/{zh_path.name})")
        lines.append("")
    write_text(README_FILE, "\n".join(lines).strip() + "\n")


def generate_verification_report(rows: list[dict[str, str]]) -> None:
    lines = [
        "# 文档转换校验报告",
        "",
        "校验规则：标题数、代码块数、行内代码数、链接数、图片数保持一致；中文译稿需包含中文字符；文中不应残留明显的原始 HTML 标签。",
        "",
        "| 文档 | 主题 | 标题 | 代码块 | 行内代码 | 链接 | 图片 | 原始 HTML 标签 | 中文占比 | 结果 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {doc} | {theme} | {headings} | {code_blocks} | {inline_code} | {links} | {images} | {raw_html} | {cjk_ratio} | {status} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("说明：")
    lines.append("- 代码块与行内代码通过占位符保护后恢复，避免翻译过程改写参数名、示例代码和命令。")
    lines.append("- 英文 Markdown 对照稿与原始 HTML 一并保留，便于人工抽查语义准确性。")
    write_text(VERIFY_FILE, "\n".join(lines) + "\n")


def main() -> None:
    ensure_source_layout()
    translator = Translator(CACHE_FILE)
    verification_rows: list[dict[str, str]] = []
    theme_map: dict[str, list[tuple[DocSpec, Path, Path]]] = {theme: [] for theme in THEMES}

    for spec in DOCS:
        raw_html = read_html(spec)
        cleaned_html = clean_article_html(raw_html, spec.source_url)
        english_markdown = markdown_from_html(cleaned_html)

        theme_dir = DOC_ROOT / spec.theme
        theme_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(spec.source_name).stem
        en_path = theme_dir / f"{base_name}.en.md"
        zh_path = theme_dir / f"{base_name}.zh-CN.md"
        assets_dir_name = f"{base_name}.assets"

        english_markdown = localize_images(english_markdown, theme_dir, assets_dir_name)
        chinese_markdown = translate_markdown(english_markdown, translator)

        source_rel_for_en = relative_link(en_path, SOURCE_DIR / spec.source_name)
        zh_rel_for_en = relative_link(en_path, zh_path)
        en_header = build_front_matter(
            title=base_name,
            source_url=spec.source_url,
            source_html_relative=source_rel_for_en,
            sibling_relative=zh_rel_for_en,
            note="由本地 HTML 正文提取得到的英文 Markdown 对照稿，便于核对译文准确性。",
        )

        source_rel_for_zh = relative_link(zh_path, SOURCE_DIR / spec.source_name)
        en_rel_for_zh = relative_link(zh_path, en_path)
        zh_header = build_front_matter(
            title=spec.title_zh,
            source_url=spec.source_url,
            source_html_relative=source_rel_for_zh,
            sibling_relative=en_rel_for_zh,
            note="自然语言内容翻译为中文；代码块、参数名、函数名、链接地址与图片资源保持原样或本地化后保留语义。",
        )

        final_english = en_header + english_markdown
        final_chinese = zh_header + chinese_markdown

        write_text(en_path, final_english)
        write_text(zh_path, final_chinese)
        theme_map[spec.theme].append((spec, en_path, zh_path))

        row = {
            "doc": spec.index,
            "theme": THEMES[spec.theme],
            "headings": "OK" if heading_count(final_english) == heading_count(final_chinese) else "DIFF",
            "code_blocks": "OK" if code_block_count(final_english) == code_block_count(final_chinese) else "DIFF",
            "inline_code": "OK" if inline_code_count(final_english) == inline_code_count(final_chinese) else "DIFF",
            "links": "OK" if link_count(final_english) == link_count(final_chinese) else "DIFF",
            "images": "OK" if image_count(final_english) == image_count(final_chinese) else "DIFF",
            "raw_html": "OK" if raw_html_count(final_chinese) == 0 else "CHECK",
            "cjk_ratio": f"{cjk_ratio(final_chinese):.3f}",
            "status": "PASS",
        }
        if "DIFF" in row.values() or row["raw_html"] == "CHECK" or cjk_ratio(final_chinese) < 0.08:
            row["status"] = "CHECK"
        verification_rows.append(row)

    translator.save()
    generate_theme_readmes(theme_map)
    generate_root_readme(theme_map)
    generate_verification_report(verification_rows)


if __name__ == "__main__":
    main()
