# 🚀 Tools-Evaluator

> Evaluate tool-calling LLMs with real-time metrics, detailed prediction analysis, and an interactive desktop UI.

Tools-Evaluator is a lightweight desktop application built with **Python**, **Tkinter**, **PyTorch**, and **Hugging Face Transformers** for evaluating tool/function-calling language models.

Whether you're fine-tuning **FunctionGemma**, **Gemma**, **Qwen**, **Llama**, **Mistral**, or any other tool-calling model, Tools-Evaluator helps you benchmark performance and inspect failures efficiently.

---
## Download

➡️ Download the latest release from:
https://github.com/SRJ2910/Tools-Evaluator/releases

## ✨ Features

### 🤖 Tool Calling Evaluation

- Evaluate tool/function-calling models on JSONL datasets
- Predict the final assistant tool call from conversation history
- Compare model predictions against ground truth
- Support for Hugging Face local models (`.safetensors`)

### 📊 Live Evaluation Dashboard

- Real-time evaluation progress
- Streaming results while evaluation is running
- Interactive desktop UI
- No waiting for the full dataset to finish

### 📈 Metrics

Tracks:

- ✅ Exact Match Accuracy
- 🔧 Tool Selection Accuracy
- 📝 Argument Accuracy
- ⏱ Inference Latency
- 🔢 Prompt Tokens
- 🔢 Generated Tokens

### 🔍 Detailed Result Inspection

For every sample:

- Ground Truth Tool
- Predicted Tool
- Ground Truth Arguments
- Predicted Arguments
- Pass / Fail Status
- Raw Model Output
- Token Usage
- Latency

### 🎨 User Experience

- PASS rows highlighted in green
- FAIL rows highlighted in red
- Responsive UI using worker threads
- Queue-based thread-safe updates
- Automatic CPU / GPU support

---

## 📸 Preview

### Main Dashboard

```text
┌─────────────────────────────────────────────────────┐
│ Model Path                                          │
│ Dataset Path                                        │
│ [ Run Evaluation ]                                  │
├─────────────────────────────────────────────────────┤
│ Progress Bar                                        │
├─────────────────────────────────────────────────────┤
│ Live Accuracy Metrics                               │
├─────────────────────────────────────────────────────┤
│ Evaluation Results Table                            │
├─────────────────────────────────────────────────────┤
│ Detailed Prediction Inspector                       │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
Tools-Evaluator/
│
├── app.py
├── evaluator.py
├── model_runner.py
├── parser.py
├── requirements.txt
│
└── ui/
    └── main_window.py
```

---

## 🧠 Evaluation Workflow

For each dataset sample:

```text
JSONL Sample
      │
      ▼
Extract Ground Truth Tool Call
      │
      ▼
Build Model Prompt
      │
      ▼
Generate Prediction
      │
      ▼
Parse Tool Call
      │
      ▼
Compare With Ground Truth
      │
      ▼
Update Metrics & UI
```

---

## 📄 Dataset Format

Expected JSONL structure:

```json
{
  "tools": [...],
  "messages": [...]
}
```

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Switch off the bedroom lights"
    },
    {
      "role": "assistant",
      "tool_calls": [
        {
          "function": {
            "name": "toggle_lights",
            "arguments": {
              "room": "bedroom",
              "state": "off"
            }
          }
        }
      ]
    }
  ]
}
```

The evaluator predicts the final assistant tool call and compares it against the expected output.

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/<your-username>/Tools-Evaluator.git

cd Tools-Evaluator
```

### Create Virtual Environment

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running

```bash
python app.py
```

---

## 🖥️ Hardware Support

### GPU (Recommended)

Automatically uses CUDA when available:

```python
torch.cuda.is_available()
```

Supports:

- NVIDIA GPUs
- bfloat16 inference
- Faster evaluation

### CPU

Automatically falls back to:

```python
torch.float32
```

No additional configuration required.

---

## 📊 Metrics Explained

### Exact Match Accuracy

Tool name and arguments must both match.

**Ground Truth**

```text
toggle_lights(room=bathroom,state=off)
```

**Prediction**

```text
toggle_lights(room=bathroom,state=off)
```

✅ Correct

---

### Tool Accuracy

Only tool name must match.

**Ground Truth**

```text
toggle_lights
```

**Prediction**

```text
toggle_lights
```

✅ Correct

---

### Argument Accuracy

Arguments must match exactly.

**Ground Truth**

```text
room=bathroom
```

**Prediction**

```text
room=bedroom
```

❌ Incorrect

---

## 🛠 Built With

- Python
- Tkinter
- PyTorch
- Hugging Face Transformers

---

## 🎯 Current Support

### Tested With

- FunctionGemma-270M-IT

### Compatible With

Any Hugging Face model capable of tool/function calling, including:

- Gemma
- FunctionGemma
- Qwen
- Llama
- Mistral

---

## 🔮 Roadmap

Planned features:

- [ ] CSV Export
- [ ] Excel Export
- [ ] HTML Evaluation Reports
- [ ] Tool-wise Accuracy Breakdown
- [ ] Failure Filtering
- [ ] Accuracy Charts
- [ ] Multi-Model Comparison
- [ ] Batch Evaluation Runs
- [ ] Confusion Matrix
- [ ] Evaluation History

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Feel free to open an issue or submit a pull request.

---

## ⭐ Support

If you find this project useful, consider giving it a star.

It helps others discover the project and motivates future improvements.

---

Made with ☕, Python, and countless tool-calling experiments.
