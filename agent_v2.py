import os
import sys
import json
import subprocess
from typing import List, TypedDict, Literal

# ---------------------------------------------------------------------------
# 0. 环境变量与路径配置
# ---------------------------------------------------------------------------
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

MODEL_NAME = "llama3.2"
OLLAMA_EXE = r"D:\ollama\ollama.exe"  # 根据你的实际路径修正


def check_and_prepare_ollama():
    """检查 Ollama 服务与模型"""
    print("🔍 正在检查 Ollama 服务与模型状态...")
    try:
        cmd = [OLLAMA_EXE, "list"] if os.path.exists(OLLAMA_EXE) else ["ollama", "list"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if MODEL_NAME not in result.stdout:
            print(f"⚠️ 未找到模型 '{MODEL_NAME}'，正在自动拉取...")
            pull_cmd = [OLLAMA_EXE, "pull", MODEL_NAME] if os.path.exists(OLLAMA_EXE) else ["ollama", "pull",
                                                                                            MODEL_NAME]
            subprocess.run(pull_cmd, check=True)
        print(f"✅ Ollama 服务正常，模型 '{MODEL_NAME}' 已准备就绪。")
    except Exception as e:
        print(f"❌ Ollama 检查失败: {e}")
        sys.exit(1)


check_and_prepare_ollama()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START

# ---------------------------------------------------------------------------
# 1. 初始化模型
# ---------------------------------------------------------------------------
llm = ChatOllama(
    model=MODEL_NAME,
    base_url="http://127.0.0.1:11434",
    temperature=0.1,  # 降低随机性
)


# ---------------------------------------------------------------------------
# 2. State 结构新增 turn_count 计数器，防止死循环
# ---------------------------------------------------------------------------
class ClinicalState(TypedDict):
    messages: List[BaseMessage]
    turn_count: int  # 问诊轮次计数器
    is_info_sufficient: bool
    soap_note: str
    icd10_code: str


# ---------------------------------------------------------------------------
# 3. 节点定义（引入强制截断与记忆防复读）
# ---------------------------------------------------------------------------

def intake_triage_node(state: ClinicalState):
    messages = state["messages"]
    turn_count = state.get("turn_count", 0) + 1

    # 【工程截断】：如果问诊超过 3 轮，强制结束问诊，不再折磨患者
    if turn_count >= 3:
        return {
            "is_info_sufficient": True,
            "turn_count": turn_count
        }

    system_prompt = SystemMessage(content="""
你是一名严谨的导诊医生。
请分析对话，判断患者提供的症状（主诉、发作方式、伴随症状）是否已经足够做出初步科室推荐。

规则：
1. 绝对不要重复询问对话历史中已经提过或患者已经回答过的任何问题！
2. 如果关键信息基本足够，或者患者表现出耐烦/无更多补充，请设置 `"is_sufficient": true`。
3. 必须输出严格的 JSON 格式：
{
    "is_sufficient": true 或 false,
    "next_question": "仅当不足且未问过时填写一个明确的新追问，否则填写 '无'"
}
""")

    prompt = [system_prompt] + messages
    response = llm.invoke(prompt)

    try:
        content = response.content.strip()
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            data = json.loads(content[start_idx:end_idx + 1])
            is_sufficient = bool(data.get("is_sufficient", False))
            next_q = str(data.get("next_question", "")).strip()
        else:
            is_sufficient = True
            next_q = ""
    except Exception:
        is_sufficient = True
        next_q = ""

    # 如果模型依然给出“无”或无效追问，强制认为信息充足
    if next_q in ["无", "", "None", "null"]:
        is_sufficient = True

    if not is_sufficient:
        return {
            "is_info_sufficient": False,
            "turn_count": turn_count,
            "messages": messages + [AIMessage(content=next_q)]
        }
    else:
        return {
            "is_info_sufficient": True,
            "turn_count": turn_count
        }


def soap_generator_node(state: ClinicalState):
    messages = state["messages"]
    dialogue_history = "\n".join(
        [f"{'患者' if m.type == 'human' else '医生'}: {m.content}" for m in messages if m.type in ['human', 'ai']])

    soap_prompt = f"""
你是一名资深临床医师。请根据以下对话记录，总结提取并生成符合规范的 SOAP 病历（即使信息有限，也请根据已获知内容合理推断与总结）：

[问诊对话记录]
{dialogue_history}

[病历格式要求]
S (Subjective / 主诉与现病史): 记录患者描述的头晕、发作特点及伴随症状。
O (Objective / 客观检查建议): 建议进行的检查（如血压测量、头颅 CT、血常规等）。
A (Assessment / 初步评估): 拟诊方向（如：突发头晕待查、前庭功能紊乱待排、短暂性脑缺血等）。
P (Plan / 诊疗计划): 推荐就诊科室（如神经内科/急诊科）及安全风险提示。
"""
    response = llm.invoke(soap_prompt)
    return {"soap_note": response.content}


def icd10_tool_node(state: ClinicalState):
    soap_note = state["soap_note"]

    lookup_prompt = f"""
根据以下 SOAP 病历的 Assessment 部分，匹配一个最贴切的标准 ICD-10 编码。
直接输出格式：诊断名称 (ICD-10 编码)

病历：
{soap_note}
"""
    response = llm.invoke(lookup_prompt)
    return {"icd10_code": response.content.strip()}


# ---------------------------------------------------------------------------
# 4. 图构建
# ---------------------------------------------------------------------------

workflow = StateGraph(ClinicalState)

workflow.add_node("intake_triage", intake_triage_node)
workflow.add_node("soap_generator", soap_generator_node)
workflow.add_node("icd10_tool", icd10_tool_node)

workflow.add_edge(START, "intake_triage")


def route_after_intake(state: ClinicalState) -> Literal["soap_generator", "__end__"]:
    if state["is_info_sufficient"]:
        return "soap_generator"
    else:
        return END


workflow.add_conditional_edges("intake_triage", route_after_intake)
workflow.add_edge("soap_generator", "icd10_tool")
workflow.add_edge("icd10_tool", END)

app = workflow.compile()

# ---------------------------------------------------------------------------
# 5. 交互运行
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" 🏥 AI 临床导诊与病历生成 Agent 已启动 (Ollama Llama3.2)")
    print("=" * 60)

    current_messages = []
    turn_count = 0

    first_input = input("\n[患者]: ")
    current_messages.append(HumanMessage(content=first_input))

    while True:
        result = app.invoke({
            "messages": current_messages,
            "turn_count": turn_count,
            "is_info_sufficient": False,
            "soap_note": "",
            "icd10_code": ""
        })

        current_messages = result.get("messages", current_messages)
        turn_count = result.get("turn_count", turn_count)

        if result.get("soap_note"):
            print("\n" + "=" * 20 + " 📋 生成的标准 SOAP 病历 " + "=" * 20)
            print(result["soap_note"])
            print("\n" + "=" * 20 + " 🏷️ ICD-10 诊断编码 " + "=" * 20)
            print(result["icd10_code"])
            print("=" * 60)
            break
        else:
            last_ai_msg = current_messages[-1].content
            print(f"\n[AI 导诊助手]: {last_ai_msg}")

            user_reply = input("\n[患者回复]: ")
            current_messages.append(HumanMessage(content=user_reply))


# 为什么这次改动能解决问题？
# 计数器兜底机制（Max Turn Rule）：代码里加上了 if turn_count >= 3:。不管模型有多笨，最多问 2-3 轮就会被强制切入“病历生成节点”，绝不会让用户陷入无限问答。
#
# 显式 Prompt 约束：增加了 绝对不要重复询问对话历史中已经提过或患者已经回答过的任何问题 强力指令，避免轻量级模型出现复读行为。
#
# “无”词判定：当小模型不知道问什么输出空串或“无”时，直接判定信息足够，顺利进入下一步。