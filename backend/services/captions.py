from __future__ import annotations
import re
import pysubs2
from typing import List, Optional
from models import Segment, WordTiming


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Punctuation that should never appear in the displayed caption (we keep it
# only as a signal for line-break detection, then strip it).
_DISPLAY_PUNCT_RE = re.compile(r"[^\w\s'’\-]", re.UNICODE)
# Characters that mark the end of a phrase / a natural line break.
_BREAK_PUNCT = set(",.;:!?…—–")
# Pause (seconds) between consecutive words that also forces a line break.
_PAUSE_GAP = 0.10
# Hard cap on words per caption line.
_MAX_WORDS_PER_LINE = 3
# Minimum time a single caption stays on screen (ms).
_MIN_DISPLAY_MS = 250


def _sanitize_libass(text: str) -> str:
    """Strip control chars and escape characters libass treats as syntax."""
    if not text:
        return ""
    text = _CONTROL_RE.sub("", text)
    text = text.replace("{", "❴").replace("}", "❵").replace("\\", "⧵")
    return text.strip()


def _strip_display_punct(text: str) -> str:
    """Remove punctuation, collapse whitespace, uppercase."""
    text = _DISPLAY_PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().upper()
    return text


def _group_words(words: list[WordTiming]) -> list[list[WordTiming]]:
    """Split a segment's words into caption lines.

    Break rules (any one triggers a new line):
      • Current word ends with sentence/phrase punctuation.
      • Gap to the next word exceeds `_PAUSE_GAP` seconds.
      • Current line already holds `_MAX_WORDS_PER_LINE` words.
    """
    groups: list[list[WordTiming]] = []
    cur: list[WordTiming] = []
    for i, w in enumerate(words):
        cur.append(w)
        ends_phrase = bool(w.word) and w.word[-1] in _BREAK_PUNCT
        is_last = i == len(words) - 1
        gap_too_big = (
            not is_last and (words[i + 1].start - w.end) >= _PAUSE_GAP
        )
        full = len(cur) >= _MAX_WORDS_PER_LINE
        if ends_phrase or gap_too_big or full or is_last:
            groups.append(cur)
            cur = []
    return groups


def _hex_to_ass_color(hex_str: str, alpha: int = 0) -> pysubs2.Color:
    """Convert "#RRGGBB" / "RRGGBB" to a pysubs2.Color (alpha 0 = opaque)."""
    if not hex_str:
        return pysubs2.Color(255, 255, 255, alpha)
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return pysubs2.Color(255, 255, 255, alpha)
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
    except ValueError:
        return pysubs2.Color(255, 255, 255, alpha)
    return pysubs2.Color(r, g, b, alpha)


def generate_ass_captions(
    segments: list[Segment],
    output_path: str,
    video_width: int = 1920,
    video_height: int = 1080,
    font_name: str = "Rubik",
    caption_box: Optional[dict] = None,
    caption_style: Optional[dict] = None,
) -> str:
    """Generate ASS subtitle file.

    Behaviour:
      • Font: Rubik (bold).
      • All caps display, punctuation hidden.
      • One caption line at a time (page-break per line).
      • Lines split on punctuation OR pauses, max 5 words per line.
      • Position / scale / rotation come from `caption_box`:
            { x: 0..100 (%), y: 0..100 (%), scale: 0..100 (% of width),
              rotation: degrees }
        When omitted, captions sit centered near the bottom.
    """
    box = caption_box or {}
    cx_pct = float(box.get("x", 50))
    cy_pct = float(box.get("y", 85))
    scale_pct = max(5.0, float(box.get("scale", 60)))
    rotation_deg = float(box.get("rotation", 0))

    cx = int(round(cx_pct / 100.0 * video_width))
    cy = int(round(cy_pct / 100.0 * video_height))
    box_width_px = scale_pct / 100.0 * video_width

    # Font size scales with the caption box width so the user's "scale" slider
    # actually changes how big the text is.
    font_size = max(40, int(round(box_width_px * 0.32)))

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(video_width)
    subs.info["PlayResY"] = str(video_height)

    # User-tunable style overrides (all optional). Defaults preserve the
    # original look (Rubik bold, white text, no stroke, soft drop shadow).
    cs = caption_style or {}
    style_font = cs.get("font_name") or font_name
    primary_hex = cs.get("primary_color") or "#FFFFFF"
    outline_hex = cs.get("outline_color") or "#000000"
    shadow_hex = cs.get("shadow_color") or "#000000"
    outline_w = float(cs.get("outline_width", 0.0))
    default_shadow = max(1.5, font_size * 0.035)
    shadow_w = float(cs.get("shadow", default_shadow))
    bold = bool(cs.get("bold", True))
    # Shadow opacity: 0..1 from user, ASS uses 0..255 inverted (0 opaque).
    shadow_alpha = int(round(max(0.0, min(1.0, float(cs.get("shadow_alpha", 0.88)))) * 255))
    shadow_alpha = 255 - shadow_alpha
    # Shadow softness (blur): 0..10. Applied per-event via \blur.
    shadow_blur = max(0.0, min(20.0, float(cs.get("shadow_blur", 0.0))))
    # Optional glow effect: simulated by widening the outline in the glow
    # color and applying a strong blur. Independent of the user-set stroke.
    glow_enabled = bool(cs.get("glow", False))
    glow_hex = cs.get("glow_color") or "#FFFFFF"
    glow_strength = max(0.0, min(20.0, float(cs.get("glow_strength", 0.0))))
    if glow_enabled and glow_strength > 0:
        # Override outline so it draws the glow halo. The blur is bumped up
        # so the halo feathers out instead of looking like a hard stroke.
        outline_hex = glow_hex
        outline_w = max(outline_w, glow_strength)
        shadow_blur = max(shadow_blur, glow_strength * 1.2)

    style = pysubs2.SSAStyle(
        fontname=style_font,
        fontsize=font_size,
        primarycolor=_hex_to_ass_color(primary_hex),
        secondarycolor=_hex_to_ass_color(primary_hex),
        outlinecolor=_hex_to_ass_color(outline_hex),
        backcolor=_hex_to_ass_color(shadow_hex, alpha=shadow_alpha),
        bold=bold,
        outline=outline_w,
        shadow=shadow_w,
        borderstyle=1,             # 1 = outline + shadow (no opaque box)
        alignment=5,               # middle-center → \pos refers to text centre
        marginl=0, marginr=0, marginv=0,
    )
    subs.styles["Default"] = style

    # ASS rotation is counter-clockwise; CSS / preview is clockwise. Negate.
    ass_rotation = -rotation_deg
    # \q2 = no automatic line wrapping. We control wrapping ourselves so the
    # caption never exceeds the user's box.
    blur_tag = ("\\blur" + format(shadow_blur, ".2f")) if shadow_blur > 0 else ""
    pos_prefix = (
        "{\\q2\\pos(" + str(cx) + "," + str(cy) + ")"
        + "\\frz" + format(ass_rotation, ".2f")
        + blur_tag
        + "}"
    )

    # Estimate the widest text (in characters) that fits inside the box at the
    # chosen font size. Rubik Bold uppercase averages ~0.62 × fontsize per char.
    AVG_CHAR_WIDTH = font_size * 0.62
    # Subtract a small inner padding so glyphs don't kiss the box edge.
    usable_width = max(1.0, box_width_px * 0.96)
    max_chars_per_line = max(1, int(usable_width // AVG_CHAR_WIDTH))

    def _split_to_fit(group: list[WordTiming]) -> list[list[WordTiming]]:
        """Greedy word-wrap so each sub-group's display text ≤ max_chars_per_line.

        Never scales the font down — overflowing groups become multiple
        sequential captions instead.
        """
        out: list[list[WordTiming]] = []
        cur: list[WordTiming] = []
        cur_len = 0
        for w in group:
            display = _strip_display_punct(w.word)
            if not display:
                # Punctuation-only token: keep it attached to current chunk timing-wise.
                cur.append(w)
                continue
            add_len = len(display) + (1 if cur_len else 0)
            if cur and cur_len + add_len > max_chars_per_line:
                out.append(cur)
                cur = [w]
                cur_len = len(display)
            else:
                cur.append(w)
                cur_len += add_len
        if cur:
            out.append(cur)
        return out

    # Build a flat ordered list of (start_ms, end_ms, display_text).
    raw_lines: list[tuple[int, int, str]] = []
    for seg in segments:
        if not seg.words:
            continue
        for group in _group_words(seg.words):
            for sub in _split_to_fit(group):
                text = _strip_display_punct(" ".join(w.word for w in sub))
                text = _sanitize_libass(text)
                if not text:
                    continue
                start_ms = int(sub[0].start * 1000)
                end_ms = int(sub[-1].end * 1000)
                if end_ms - start_ms < _MIN_DISPLAY_MS:
                    end_ms = start_ms + _MIN_DISPLAY_MS
                raw_lines.append((start_ms, end_ms, text))

    # ------------------------------------------------------------------
    # Caption non-overlap guarantee.
    #
    # Invariant we want at all costs: at any instant, AT MOST ONE caption
    # is on screen. Two captions whose [start,end] windows touch is fine;
    # any overlap (even 1 ms) is forbidden — the user explicitly asked for
    # this to be unbreakable.
    #
    # Algorithm:
    #   1. Sort by start.
    #   2. Walk forward. Each caption's start is pushed to >= previous end.
    #   3. Each caption's end is CLAMPED to next caption's start.
    #   4. If a caption ends up shorter than _MIN_DISPLAY_MS, DROP it
    #      rather than letting it overlap. (Better to skip a caption than
    #      to have two on screen at once.)
    # ------------------------------------------------------------------
    raw_lines.sort(key=lambda x: x[0])

    # Pass 1: push starts forward so no caption begins before the previous
    # one ends.  Allow native overlap to be resolved by the start-push.
    pushed: list[tuple[int, int, str]] = []
    for start_ms, end_ms, text in raw_lines:
        if pushed:
            prev_end = pushed[-1][1]
            if start_ms < prev_end:
                start_ms = prev_end
        if end_ms < start_ms:
            end_ms = start_ms
        pushed.append((start_ms, end_ms, text))

    # Pass 2: clamp every caption's end to the NEXT caption's start, so
    # captions touch but never overlap. Drop any caption shorter than the
    # min display floor.
    final_events: list[tuple[int, int, str]] = []
    for idx, (start_ms, end_ms, text) in enumerate(pushed):
        next_start = pushed[idx + 1][0] if idx + 1 < len(pushed) else None
        if next_start is not None:
            end_ms = next_start  # always clamp — never extend past next
        # Floor: if the available window is too short, the caption would
        # flash by; skip it entirely so the previous one stays on a beat
        # longer (already extended by the pass-2 clamp on its own end).
        if end_ms - start_ms < _MIN_DISPLAY_MS:
            continue
        final_events.append((start_ms, end_ms, text))

    # Pass 3: now extend the LAST surviving caption out to a sensible end
    # using the natural end from `pushed` (otherwise it would just stop at
    # the dropped successor's start). We grab the natural end from the
    # last entry in `pushed` regardless of whether it was kept.
    if final_events and pushed:
        natural_last_end = max(e for _, e, _ in pushed)
        ls, le, lt = final_events[-1]
        if natural_last_end > le:
            final_events[-1] = (ls, natural_last_end, lt)

    # Sanity check: enforce no overlap one more time (defensive).
    for i in range(1, len(final_events)):
        prev_end = final_events[i - 1][1]
        cur_start = final_events[i][0]
        if cur_start < prev_end:
            # Should be impossible, but if it ever happens, snap.
            ps, _pe, pt = final_events[i - 1]
            final_events[i - 1] = (ps, cur_start, pt)

    for start_ms, end_ms, text in final_events:
        event = pysubs2.SSAEvent(
            start=start_ms,
            end=end_ms,
            text=pos_prefix + text,
            style="Default",
        )
        subs.events.append(event)

    subs.save(output_path)
    return output_path
