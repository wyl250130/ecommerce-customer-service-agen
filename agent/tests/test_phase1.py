from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from agent.evidence_gate import check_evidence
from agent.recorder import TrajectoryRecorder
from agent.runner import AgentRunner
from agent.schemas import ToolAction
from agent.session_state import SessionState
from agent.tool_registry import execute_tool
from agent.tools.common import configure_connection
from agent.tools.logistics import get_logistics
from agent.tools.order import get_order_info
from agent.verifier import verify_trajectory


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data/processed/customer_service/customer_service_train.sqlite"
SNAPSHOT_ID = "GOLD-SNAP-SEMANTIC-GOLD-TRAIN-000140"
ORDER_ID = "ORDER-TRAIN-000140"


class Phase1AgentTests(unittest.TestCase):
    def setUp(self):
        self.conn = configure_connection(sqlite3.connect(DB_PATH))

    def tearDown(self):
        self.conn.close()

    def test_order_tool_found(self):
        result = get_order_info(self.conn, SNAPSHOT_ID, ORDER_ID)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["order_id"], ORDER_ID)
        self.assertEqual(result.data["shipment_status"], "shipped")

    def test_order_tool_snapshot_isolation(self):
        result = get_order_info(self.conn, "wrong-snapshot", ORDER_ID)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "ORDER_NOT_FOUND")

    def test_logistics_tool_found(self):
        result = get_logistics(self.conn, SNAPSHOT_ID, ORDER_ID)
        self.assertTrue(result.ok)
        self.assertEqual(result.data["order_id"], ORDER_ID)
        self.assertEqual(result.data["logistics_status"], "in_transit_slow_loss_not_confirmed")

    def test_registry_rejects_agent_snapshot_id(self):
        state = SessionState(conversation_id="test", snapshot_id=SNAPSHOT_ID)
        action = ToolAction(
            tool="get_order_info",
            arguments={"order_id": ORDER_ID, "snapshot_id": "wrong-snapshot"},
        )
        result = execute_tool(action, state, self.conn)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "SNAPSHOT_ID_FORBIDDEN")

    def test_evidence_gate_ready_after_order_and_logistics(self):
        state = SessionState(conversation_id="test", snapshot_id=SNAPSHOT_ID)
        for action in [
            ToolAction("get_order_info", {"order_id": ORDER_ID}),
            ToolAction("get_logistics", {"order_id": ORDER_ID}),
        ]:
            state.add_tool_result(action, execute_tool(action, state, self.conn))
        gate = check_evidence(state, "logistics_status")
        self.assertTrue(gate.ready)

    def test_evidence_gate_blocks_missing_logistics(self):
        state = SessionState(conversation_id="test", snapshot_id=SNAPSHOT_ID)
        action = ToolAction("get_order_info", {"order_id": ORDER_ID})
        state.add_tool_result(action, execute_tool(action, state, self.conn))
        gate = check_evidence(state, "logistics_status")
        self.assertFalse(gate.ready)
        self.assertIn("logistics_status", gate.missing)

    def test_evidence_gate_distinguishes_missing_and_unknown(self):
        state = SessionState(conversation_id="test", snapshot_id=SNAPSHOT_ID)
        action = ToolAction("get_order_info", {"order_id": ORDER_ID})
        state.add_tool_result(action, execute_tool(action, state, self.conn))
        state.evidence[-1].data["shipment_status"] = None
        state.evidence[-1].data.pop("order_id")

        gate = check_evidence(state, "shipping_status")
        self.assertIn("order_id", gate.missing)
        self.assertIn("shipment_status", gate.unknown)

    def test_runner_records_and_verifies_manual_actions(self):
        state = SessionState(conversation_id="test", snapshot_id=SNAPSHOT_ID, user_messages=["物流到哪了"])
        recorder = TrajectoryRecorder("TRAJ-TEST")
        runner = AgentRunner(self.conn)
        result = runner.run(
            state,
            [
                {"type": "tool_call", "tool": "get_order_info", "arguments": {"order_id": ORDER_ID}},
                {"type": "tool_call", "tool": "get_logistics", "arguments": {"order_id": ORDER_ID}},
                {"type": "finish"},
            ],
            intent="logistics_status",
            recorder=recorder,
        )
        self.assertEqual(result.status, "finished")
        verification = verify_trajectory(recorder.to_dict(), state, "logistics_status")
        self.assertTrue(verification.passed)
        first_step = recorder.to_dict()["steps"][0]
        self.assertEqual(first_step["state_before"]["evidence"], [])
        self.assertEqual(first_step["state_after"]["evidence"][0]["source_type"], "order")

    def test_runner_blocks_duplicate_tool_call(self):
        state = SessionState(conversation_id="test", snapshot_id=SNAPSHOT_ID)
        runner = AgentRunner(self.conn)
        result = runner.run(
            state,
            [
                {"type": "tool_call", "tool": "get_order_info", "arguments": {"order_id": ORDER_ID}},
                {"type": "tool_call", "tool": "get_order_info", "arguments": {"order_id": ORDER_ID}},
            ],
        )
        self.assertEqual(result.status, "error")
        self.assertIn("Duplicate tool call", result.errors[0])

    def test_runner_reports_policy_exhausted(self):
        state = SessionState(conversation_id="test", snapshot_id=SNAPSHOT_ID)
        runner = AgentRunner(self.conn)
        result = runner.run(
            state,
            [
                {"type": "tool_call", "tool": "get_order_info", "arguments": {"order_id": ORDER_ID}},
                {"type": "tool_call", "tool": "get_logistics", "arguments": {"order_id": ORDER_ID}},
            ],
            intent="logistics_status",
        )
        self.assertEqual(result.status, "policy_exhausted")
        self.assertIn("POLICY_EXHAUSTED", result.errors)


if __name__ == "__main__":
    unittest.main()
