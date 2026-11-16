"""Legacy keyword policy retrieval, shared by CLI and Agent tool."""
import json
import re

STOP_TERMS = {"客服", "怎么", "么办", "怎么办"}


def terms(text):
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", (text or "").lower()):
        tokens.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.update(token[index:index + 2] for index in range(len(token) - 1))
    return tokens - STOP_TERMS


def phrase_score(query, record, fields):
    query_text = (query or "").lower()
    query_terms = terms(query_text)
    score = 0
    reasons = []
    for field, weight in fields:
        value = record.get(field) or []
        values = value if isinstance(value, list) else [value]
        for item in values:
            item_text = str(item).lower()
            if item_text and item_text not in STOP_TERMS and item_text in query_text:
                score += weight
                reasons.append(f"{field}:exact:{item}")
            overlap = query_terms & terms(item_text)
            if overlap:
                score += len(overlap)
                reasons.append(f"{field}:overlap:{','.join(sorted(overlap))}")
    return score, reasons


def retrieve_policy_records(connection, query, limit=5):
    results = []
    rows = connection.execute(
        """SELECT policy_id, title, normalized_rule, source_level, review_status,
                  source_url, raw_rule_json
           FROM rule_documents
           WHERE review_status IN ('approved_for_simulation',
                                   'approved_for_rule_context_with_temporal_note')"""
    )
    for policy_id, title, normalized_rule, source_level, review_status, source_url, raw in rows:
        record = json.loads(raw)
        score, reasons = phrase_score(
            query, record,
            (("triggers", 8), ("domain", 4), ("title", 3), ("normalized_rule", 2)),
        )
        if score:
            results.append({
                "policy_id": policy_id,
                "title": title,
                "normalized_rule": normalized_rule,
                "source_level": source_level,
                "review_status": review_status,
                "source_url": source_url,
                "version": record.get("version") or record.get("effective_time"),
                "score": score,
                "match_reasons": reasons,
            })
    return sorted(results, key=lambda item: (-item["score"], item["policy_id"]))[:limit]
