"""Handoff routing and simulated support gateway behavior."""

import json
import unittest

from langchain_core.messages import ToolMessage

from agent.handoff import SimulatedHandoffClient
from agent.nodes.handoff import make_handoff_node
from agent.session_state import SessionState


class HandoffNodeTests(unittest.TestCase):
    def setUp(self):
        self.session = SessionState("chat_001", "snapshot_001")
        self.message = ToolMessage(
            content=json.dumps({
                "ok": True,
                "route": "handoff",
                "reason_code": "refund_dispute",
                "reason_detail": "退款尚未到账，用户要求人工核实",
            }, ensure_ascii=False),
            name="handoff_to_human",
            tool_call_id="call_handoff_1",
        )

    def test_simulated_transfer_updates_status_and_returns_ticket(self):
        node = make_handoff_node(SimulatedHandoffClient(), self.session)
        state = {"messages": [self.message], "tool_call_count": 1}
        update = node(state)
        self.assertEqual(update["service_status"], "transferred")
        self.assertTrue(update["handoff_result"]["success"])
        self.assertTrue(update["handoff_result"]["ticket_id"].startswith("SIM-"))
        self.assertEqual(node({**state, **update}), {})

    def test_gateway_failure_does_not_claim_transfer(self):
        class FailingClient:
            def transfer(self, **kwargs):
                raise ConnectionError("offline")

        node = make_handoff_node(FailingClient(), self.session)
        update = node({"messages": [self.message], "tool_call_count": 1})
        self.assertEqual(update["service_status"], "handoff_failed")
        self.assertEqual(update["handoff_result"], {
            "success": False, "handoff_status": "failed", "ticket_id": None,
        })

    def test_missing_validated_request_is_rejected(self):
        node = make_handoff_node(SimulatedHandoffClient(), self.session)
        with self.assertRaises(ValueError):
            node({"messages": [], "tool_call_count": 0})


if __name__ == "__main__":
    unittest.main()
