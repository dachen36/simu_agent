# Simu Agent

这是一个基于 Ollama + LangChain + LangGraph 的 AI 问诊助手示例项目，能够：

- 识别患者提供的主要症状与关键信息
- 通过多轮问诊判断信息是否足够
- 自动生成标准 SOAP 病历
- 根据病历给出 ICD-10 诊断编码建议

项目描述：simulate for agent

## 功能说明

本项目实现了一个简单的临床导诊流程：

1. 接收患者的首条输入
2. AI 评估当前信息是否足够做出初步分诊
3. 在信息不足时继续追问，最多进行有限轮次
4. 当信息足够时自动生成 SOAP 病历
5. 进一步生成 ICD-10 诊断编码建议

核心文件：

- `agent_v2.py`：主程序，包含模型初始化、状态管理、问诊节点和病历生成逻辑

## 技术栈

- Python
- Ollama
- LangChain
- LangGraph
- Llama 3.2

## 环境要求

- Python 3.10+
- Ollama 已安装并运行
- 已下载模型 `llama3.2`

## 安装步骤

1. 克隆项目

```bash
git clone https://github.com/dachen36/simu_agent.git
cd simu_agent
```

2. 创建虚拟环境（可选）

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. 安装依赖

```bash
pip install langchain-core langchain-ollama langgraph
```

4. 启动 Ollama，并确认模型可用

```bash
ollama pull llama3.2
```

如果你的 Ollama 安装路径不是默认位置，需在 `agent_v2.py` 中检查并修改：

```python
OLLAMA_EXE = r"D:\ollama\ollama.exe"
```

## 运行方式

```bash
python agent_v2.py
```

然后按提示输入患者信息，例如：

```text
[患者]: 我最近头晕，伴有恶心，发作时站起来就会晕。
```

程序会输出：

- AI 导诊助手追问
- 最终生成的 SOAP 病历
- 推荐的 ICD-10 诊断编码

## 注意事项

- 当前实现以演示为主，适合本地开发和实验
- 默认使用 `llama3.2` 模型，需要本地有 Ollama 服务支持
- 代码中有针对“过度追问”的限制，避免问诊轮次过多导致死循环
- 该项目更偏“实验性原型”，可作为进一步扩展为真实医疗辅助系统的基础

## 许可证

本项目未声明明确许可证，默认按仓库原始状态使用。若需要正式发布，请根据实际需求补充 LICENSE 文件。

## 免责声明

本项目仅用于学习、原型开发和研究用途，不应当被用于真实医疗决策。任何临床判断仍需由合格医疗专业人员进行审核。
