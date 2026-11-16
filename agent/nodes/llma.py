from pathlib import Path
from langchain_core.messages import SystemMessage

# 获取 agent 目录
AGENT_DIR = Path(__file__).resolve().parents[1]

# 读取 LLMA System Prompt
PROMPT_PATH = AGENT_DIR / "prompts" / "llma_system.md"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


def make_llma_node(llma_with_tools):

    def llma_node(state):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"]
        ]

        response = llma_with_tools.invoke(messages)

        return {"messages": [response]}

    return llma_node