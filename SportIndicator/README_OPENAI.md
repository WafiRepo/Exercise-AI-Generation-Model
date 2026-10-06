# Panduan Penggunaan OpenAI API untuk G-Eval

## Instalasi

### Untuk menggunakan OpenAI:
```bash
pip install openai
```

### Untuk menggunakan Anthropic (default):
```bash
pip install anthropic
```

**Catatan**: Anda hanya perlu menginstall library untuk provider yang akan digunakan. Jika hanya menggunakan OpenAI, tidak perlu install `anthropic`, dan sebaliknya.

## Setup Environment Variable

### Untuk OpenAI:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### Untuk Anthropic (default):
```bash
export ANTHROPIC_KEY=your_anthropic_api_key_here
```

## Cara Menggunakan

### Menggunakan OpenAI (GPT-4o)

```bash
# BodyPart Evaluation
python SportIndicator/BodyPart_evaluation.py \
    --predict ./workflow_output/instruction.txt \
    --output ./evaluation_results \
    --provider openai

# Error Evaluation
python SportIndicator/Error_evaluation.py \
    --predict ./workflow_output/instruction.txt \
    --output ./evaluation_results \
    --provider openai

# Causation Evaluation
python SportIndicator/Causation_evaluation.py \
    --predict ./workflow_output/instruction.txt \
    --output ./evaluation_results \
    --provider openai

# Coordination Evaluation
python SportIndicator/Coordination_evaluation.py \
    --predict ./workflow_output/instruction.txt \
    --output ./evaluation_results \
    --provider openai

# Method Evaluation
python SportIndicator/Method_evaluation.py \
    --predict ./workflow_output/instruction.txt \
    --output ./evaluation_results \
    --provider openai

# Time Evaluation
python SportIndicator/Time_evaluation.py \
    --predict ./workflow_output/instruction.txt \
    --output ./evaluation_results \
    --provider openai
```

### Menggunakan Anthropic (Claude) - Default

```bash
# Tanpa perlu specify --provider (default adalah anthropic)
python SportIndicator/BodyPart_evaluation.py \
    --predict ./workflow_output/instruction.txt \
    --output ./evaluation_results
```

### Menggunakan API Key Langsung (tanpa env var)

```bash
python SportIndicator/BodyPart_evaluation.py \
    --predict ./workflow_output/instruction.txt \
    --output ./evaluation_results \
    --provider openai \
    --api_key your_api_key_here
```

## Perbandingan Provider

| Provider | Model | Biaya | Kualitas |
|----------|-------|-------|----------|
| Anthropic | Claude 3.5 Sonnet | Lebih mahal | Sangat baik untuk evaluasi |
| OpenAI | GPT-4o | Lebih murah | Baik untuk evaluasi |

## Catatan

1. **Format Response**: Pastikan prompt menghasilkan response yang bisa di-parse (angka 1-5)
2. **Model OpenAI**: Default menggunakan `gpt-4o`. Untuk menggunakan model lain, modifikasi `detection.py` di bagian `model="gpt-4o"`
3. **Konsistensi**: Hasil bisa berbeda antara Claude dan GPT, tetapi umumnya konsisten untuk G-Eval
4. **Error Handling**: Jika API key tidak ditemukan, script akan memberikan error message yang jelas
