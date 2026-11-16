"""Auditable adapter for the existing reviewed SQLite policy card prototype."""
import json
from ...schemas import ToolResult
from .keyword import retrieve_policy_records


def search_policies(conn, snapshot_id, query, limit=5):
    if not isinstance(query, str) or not query.strip() or type(limit) is not int or not 1 <= limit <= 20:
        return ToolResult(False, "search_policies", "tool_error", error_code="ARGUMENT_ERROR")
    records = retrieve_policy_records(conn, query.strip(), limit)
    for record in records:
        raw = json.loads(conn.execute("SELECT raw_rule_json FROM rule_documents WHERE policy_id=?", (record["policy_id"],)).fetchone()[0])
        for key in ("conditions", "scope", "evidence_required", "must_mention", "must_not_say", "escalation_when", "effective_time", "temporal_status"):
            record[key] = raw.get(key)
    return ToolResult(bool(records), "search_policies", "found" if records else "no_match",
                      {"results": records, "retrieval_backend": "legacy_keyword_sqlite_cards"},
                      evidence_refs=[{"source": "rule_documents", "policy_id": r["policy_id"], "version": r["version"]} for r in records],
                      error_code=None if records else "POLICY_NOT_FOUND")
