"""Batched vLLM judge shared by the QA and event stages."""

MODEL = "Qwen/Qwen3.6-27B"


class Engine:
    def __init__(self, model_path=None):
        if model_path is None:
            from worlddraft_eval.common.models import local
            model_path = str(local(MODEL))
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_path, dtype="bfloat16", max_model_len=16384, max_num_seqs=64,
                       gpu_memory_utilization=0.92, limit_mm_per_prompt={"image": 64},
                       mm_processor_kwargs={"max_pixels": 480 * 832})
        self.params = SamplingParams(temperature=0, max_tokens=1024)
        import vllm
        self.metadata = {"backend": "vllm", "vllm_version": vllm.__version__, "model_path": str(model_path),
                         "temperature": 0, "max_num_seqs": 64, "max_pixels_per_frame": 480 * 832}

    def run(self, prompts):
        """prompts: list of content lists (image and text parts). Returns raw texts."""
        msgs = []
        for content in prompts:
            parts = []
            for c in content:
                parts.append({"type": "image_pil", "image_pil": c["image"].convert("RGB")} if c["type"] == "image"
                             else {"type": "text", "text": c["text"]})
            msgs.append([{"role": "user", "content": parts}])
        if not msgs:
            return []
        out = self.llm.chat(msgs, self.params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
        return [o.outputs[0].text for o in out]
