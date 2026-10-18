"""Keep the target and candidate visible within the fixed reranker token budget."""
import re


TARGET_INSTRUCTIONS = {
    "ICD": "Match the highlighted extracted clinical entity to the candidate ICD-10-CM code. Context disambiguates that entity; other conditions in the context are not targets. Require the same concept and supported qualifiers, not merely a related topic. Match the extracted concept even when historical or negated. Do not assume unstated specificity.",
    "SNOMED": "Match the highlighted clinical entity to the candidate SNOMED CT concept. Context disambiguates that entity; other conditions in the context are not targets. Require the same concept and supported qualifiers, not merely a related topic. Distinguish findings, procedures, tests, devices and explicit absence or history. Use Type as a clue. Do not assume unstated specificity.",
}


def _length(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _clip(tokenizer, text, limit, anchor=False):
    if limit <= 0:
        return ""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if len(offsets) <= limit:
        return text
    start = 0
    if anchor and "[M]" in text and "[/M]" in text:
        a, b = text.index("[M]"), text.index("[/M]") + 4
        first = next((i for i, (_, end) in enumerate(offsets) if end > a), 0)
        last = next((i for i, (begin, _) in enumerate(offsets) if begin >= b), len(offsets))
        start = max(0, first - max(0, limit - (last - first)) // 2)
        start = min(start, len(offsets) - limit)
    return text[offsets[start][0]:offsets[start + limit - 1][1]]


def pack_payload(tokenizer, payload, budget):
    """Preserve fitting inputs exactly; shorten overflowing context/optional names.

    The official chat wrapper and 512-token maximum remain unchanged. Trimming
    uses offsets into the original text, so it never rewrites chart spelling.
    """
    if _length(tokenizer, payload) <= budget:
        return payload
    before, separator, document = payload.partition("\n<Document>: ")
    if not separator:
        raise ValueError("Reranker input must include a candidate document")
    # The card starts with its code/preferred name, FSN or refinement. Keep these
    # ahead of optional aliases rather than letting the query consume the card.
    document = _clip(tokenizer, document, min(_length(tokenizer, document), max(96, budget // 3)))
    before = "\n".join(line for line in before.split("\n")
                       if not line.startswith(("Expansions: ", "Possible reading: ")))
    head, context_separator, context = before.partition("\nContext: ")
    if not context_separator:
        context_separator, context = "", ""
    fixed = head + context_separator + separator + document
    if context and _length(tokenizer, fixed) + _length(tokenizer, _clip(tokenizer, context, _length(tokenizer, context), True)) > budget:
        # A long entity can otherwise appear twice. Its marked context still
        # supplies the target if the separate Mention line is redundant.
        target = re.search(r"\[M\].*?\[/M\]", context, re.S)
        if target and _length(tokenizer, fixed) + _length(tokenizer, target.group()) > budget - 8:
            head = "\n".join(line for line in head.split("\n") if not line.startswith("Mention: "))
    fixed = head + context_separator + separator + document
    remaining = budget - _length(tokenizer, fixed) - 8
    if remaining <= 0:
        # Extremely long metadata must still leave a nonempty candidate card.
        document = _clip(tokenizer, document, max(16, budget // 4))
        head = _clip(tokenizer, head, max(16, budget // 3))
        fixed = head + context_separator + separator + document
        remaining = max(0, budget - _length(tokenizer, fixed) - 8)
    context = _clip(tokenizer, context, remaining, anchor=True)
    result = head + context_separator + context + separator + document
    if not document or _length(tokenizer, result) > budget:
        raise ValueError("Unable to retain the candidate within the reranker token budget")
    return result
