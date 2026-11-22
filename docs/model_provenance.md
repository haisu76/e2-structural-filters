# E2 生成器来源记录

冻结于正式波次之前（2026-11-11）。对应 `E2_PREREGISTRATION_manifest.json` 中 `design.generators`。

## G1 — DeepSeek API（主分析）

| 项 | 值 |
|---|---|
| 端点 | `https://api.deepseek.com` |
| 模型 id | `deepseek-flash`（账户 `/models` 返回；另有 `deepseek-v4-pro`，未使用） |
| 推理模式 | **开启**，`reasoning_effort=low`；推理关闭时近似复制率高（10-seed 通过 21/50），开启后 39/50 |
| 解码 | temperature 0.7, top_p 0.95, seed_base 1000, **max_tokens 16384**（8192 时 2/50 被推理耗尽而截断） |
| 兼容性 | `seed` 参数可用（2026-11-11 验证） |

确认命令：

```bash
curl -s https://api.deepseek.com/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

## G2 — Qwen2.5-7B-Instruct（独立复现）

| 项 | 值 |
|---|---|
| 端点 | `http://10.10.40.166:11434/v1` |
| 模型 id | `qwen2.5-7b-instruct-q4km:latest` |
| 模型 digest | `d82159a3b4f9f26b485d9f6b615d0d0d250ffc92ec6d142e62d28474b46a1b3a` |
| 参数规模 / 量化 | 7.6B / Q4_K_M（GGUF） |
| 模型文件大小 | 5,069,917,388 bytes |
| 运行时 | Ollama 0.40.16（`/home/haisu/ollama`），llama.cpp runner，`size_vram: 0`（纯 CPU） |
| context_length | 4096 |
| 解码 | temperature 0.7, top_p 0.95, seed_base 1000, max_tokens 自动 |
| 实测吞吐 | 约 4.2 tok/s；单条约 95–125 s |

## 说明

- 两个生成器使用同一批 200 个种子、同一套提示词、同一套硬校验与纳入规则。
- 冒烟/试跑文件（`smoke*.jsonl`、`pilot_qwen_10seed.jsonl`）只用于管线检查，不进入正式分析。
- 正式日志：`e2_generation/wave1_deepseek.jsonl`、`e2_generation/wave1_qwen.jsonl`。
