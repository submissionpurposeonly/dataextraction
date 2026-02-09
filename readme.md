# SHIELDA: A General-Purpose Exception-Handling Framework

This repository contains the official implementation, prompts, and expanded evaluation datasets for the SHIELDA Major Revision (TOSEM).

## 🚀 Key Features
- **Framework-Agnostic**: Non-intrusive integration with LangChain and Hugging Face.
- **Deep Strategic Intervention**: Resolves Logical Deadlocks and Strategic Misalignments.

## 📁 Case Study Mapping (Appendices)
| Appendix | Scenario | Framework | Log Directory |
| :--- | :--- | :--- | :--- |
| **G** | Multi-hop Synthesis | LangChain | `experiments/logs/App_G_LangChain/` |
| **H** | Schrödinger File | Hugging Face | `experiments/logs/App_H_HF_Schrodinger/` |
| ... | ... | ... | ... |

## 🛠️ Getting Started
```bash
pip install -r requirements.txt
# Run the escalation demo
python shielda_engine/escalator.py --task example