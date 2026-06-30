"""
Lightweight i18n support for KGClaw using a simple gettext-compatible .po loader.

Zero external dependencies (no GNU gettext tools needed for runtime).
Translation files are .po format stored under kgclaw/locales/<lang>/LC_MESSAGES/.

Usage:
    from kgclaw.i18n import _, init_locale

    init_locale("en")       # or None for auto-detect
    print(_("欢迎使用 KGClaw！"))  # -> "Welcome to KGClaw!" when lang=en
"""

from __future__ import annotations

import locale
import os
import re
from pathlib import Path
from typing import Optional

# ── Module state ──────────────────────────────────────────────────────────────

_translations: dict[str, str] = {}  # source text -> translated text
_current_locale: str = "en"

_po_pattern = re.compile(
    r'^msgid\s+"((?:[^"\\]|\\.)*)"\s*\nmsgstr\s+"((?:[^"\\]|\\.)*)"',
    re.MULTILINE,
)


def _load_po(filepath: Path) -> dict[str, str]:
    """Parse a .po file into a dict of {msgid: msgstr}."""
    translations: dict[str, str] = {}
    if not filepath.exists():
        return translations

    text = filepath.read_text(encoding="utf-8")
    # Normalize multi-line entries into single line for matching
    # (this handles the common case; for production-grade parsing use polib)
    for match in _po_pattern.finditer(text):
        msgid = _unescape_po(match.group(1))
        msgstr = _unescape_po(match.group(2))
        if msgid and msgstr:  # skip empty translations
            translations[msgid] = msgstr

    # Also handle multi-line msgid/msgstr (simplified parser)
    if not translations:
        translations = _load_po_multiline(text)

    return translations


def _load_po_multiline(text: str) -> dict[str, str]:
    """Fallback parser for multi-line .po entries."""
    translations: dict[str, str] = {}
    entries = re.split(r'\n\s*\n', text)
    current_id = None
    current_str = None

    for entry in entries:
        entry = entry.strip()
        if not entry or entry.startswith('#'):
            continue

        id_match = re.search(r'msgid\s+"(.*)"', entry)
        str_match = re.search(r'msgstr\s+"(.*)"', entry)

        # Multi-line entries
        id_matches = re.findall(r'^msgid\s+"(.*)"', entry, re.MULTILINE)
        str_matches = re.findall(r'^msgstr\s+"(.*)"', entry, re.MULTILINE)

        if not id_matches and 'msgid ""' in entry:
            # Multi-line: msgid ""\n"line1"\n"line2"
            id_lines = re.findall(r'"((?:[^"\\]|\\.)*)"', entry.split('msgstr')[0])
            str_lines = re.findall(r'"((?:[^"\\]|\\.)*)"', entry.split('msgstr')[1]) if 'msgstr' in entry else []
            msgid = ''.join(id_lines[1:]) if len(id_lines) > 1 else ''  # skip empty first line
            msgstr = ''.join(str_lines[1:]) if len(str_lines) > 1 else ''
        else:
            msgid = _unescape_po(id_matches[0]) if id_matches else ''
            msgstr = _unescape_po(str_matches[0]) if str_matches else ''

        if msgid and msgstr:
            translations[msgid] = msgstr

    return translations


def _unescape_po(s: str) -> str:
    """Unescape common .po escape sequences."""
    return s.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')


# ── Public API ────────────────────────────────────────────────────────────────


def init_locale(lang: Optional[str] = None):
    """Initialize translations for the given language.

    Args:
        lang: Language code ("zh", "en"). If None, auto-detects from
              config file → KGCLAW_LANG env → LANG env → default "en".

    Auto-detection order: explicit arg → UserConfig.get_lang() → env vars
    Defaults to "en" (English).
    """
    global _translations, _current_locale

    if lang is None:
        try:
            from .config import UserConfig
            lang = UserConfig.get_lang()
        except Exception:
            lang = _detect_lang()

    lang = lang[:2] if len(lang) >= 2 else lang
    _current_locale = lang

    if lang == "zh":
        # Chinese — source text in code is Chinese, no translation needed
        _translations = {}
        return
    # For "en" or any other locale, load the corresponding .po file

    # Load .po file
    localedir = Path(__file__).parent / "locales"
    po_path = localedir / lang / "LC_MESSAGES" / "kgclaw.po"
    _translations = _load_po(po_path)


def _(text: str) -> str:
    """Translate a string. Returns translation if available, else the original."""
    if not _translations:
        return text
    return _translations.get(text, text)


def current_locale() -> str:
    """Return the current locale code (e.g., 'zh', 'en')."""
    return _current_locale


class _LazyString:
    """A string-like object that defers translation until str() or format() is called.

    Used for Click help= parameters which are evaluated at import time,
    before i18n is initialized. When Click renders help text and calls
    str() or format() on these strings, the translation happens on-demand.
    """

    __slots__ = ('_text',)

    def __init__(self, text: str):
        self._text = text

    def __str__(self) -> str:
        return _(self._text)

    def __format__(self, format_spec: str) -> str:
        return _(self._text).__format__(format_spec)

    def __repr__(self) -> str:
        return f"_L({self._text!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, _LazyString):
            return self._text == other._text
        return _(self._text) == other

    def __hash__(self) -> int:
        return hash(self._text)

    def __bool__(self) -> bool:
        return bool(self._text)

    # Delegate common string methods to the translated version
    def lower(self):
        return _(self._text).lower()

    def upper(self):
        return _(self._text).upper()

    def split(self, *args, **kwargs):
        return _(self._text).split(*args, **kwargs)

    def replace(self, *args, **kwargs):
        return _(self._text).replace(*args, **kwargs)


# Public alias
_L = _LazyString


def _f(template: str, **kwargs) -> str:
    """Translate a template string, then format it with keyword arguments.

    Usage: _f("找到 {n} 个实体", n=count)

    This is a convenience wrapper for _(template).format(**kwargs).
    For f-strings with inline expressions like {ICON['bullet']}, pass
    the evaluated value as a kwarg.
    """
    return _(template).format(**kwargs)


def _detect_lang() -> str:
    """Detect language from environment variables and system locale."""
    # 1. Explicit env var
    for key in ("KGCLAW_LANG", "LANG", "LANGUAGE"):
        val = os.environ.get(key, "")
        if val:
            lang = val[:2] if len(val) >= 2 else val
            if lang not in ("en", "zh"):
                continue  # unrecognized — try next source
            return lang

    # 2. System locale
    try:
        loc = locale.getdefaultlocale()
        if loc and loc[0]:
            lang = loc[0][:2]
            if lang in ("en", "zh"):
                return lang
    except Exception:
        pass

    # 3. Default: English
    return "en"
