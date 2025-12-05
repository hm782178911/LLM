# nano vllm 源码学习

# 0.总览

从[github仓库](https://github.com/GeeeekExplorer/nano-vllm)拉取

```bash
https://github.com/GeeeekExplorer/nano-vllm.git
```

## 目录结构

```bash
(llm) ziyi.he@mlc05:~/Code/nano-vllm$ ls
assets  bench.py  example.py  LICENSE  nanovllm  pyproject.toml  README.md
```

核心代码在 ```nano-vllm/nanovllm```

```bash
(llm) ziyi.he@mlc05:~/Code/nano-vllm/nanovllm$ tree ./ -L 2
./
├── config.py
├── engine
│   ├── block_manager.py
│   ├── llm_engine.py
│   ├── model_runner.py
│   ├── __pycache__
│   ├── scheduler.py
│   └── sequence.py
├── __init__.py
├── layers
│   ├── activation.py
│   ├── attention.py
│   ├── embed_head.py
│   ├── layernorm.py
│   ├── linear.py
│   ├── __pycache__
│   ├── rotary_embedding.py
│   └── sampler.py
├── llm.py
├── models
│   ├── __pycache__
│   └── qwen3.py
├── __pycache__
│   ├── config.cpython-310.pyc
│   ├── __init__.cpython-310.pyc
│   ├── llm.cpython-310.pyc
│   └── sampling_params.cpython-310.pyc
├── sampling_params.py
└── utils
    ├── context.py
    └── loader.py
```



## 入口

源码里提供了一个用例  ```nano-vllm/example.py``` ,核心代码如下所示

```python
	path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    #[源码学习] 核心代码部分，进入到engine的generate函数
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
```



# 1.推理流程

## engine-->generate

简单描述，以下内容已经进入engine

```
传入参数 (prompts, sampling_params)

如果有多个prompts，给每个prompts复制一份 sampling_params 来配对。保证传进去的是一句 prompt 和一个 sampling_params 的对；

对于每个 promt 和 sampling_params（下文简称sp）的对
给 prompt 分词成 token_id 的序列 token_ids
在和 sp 一起构建一个 Sequence类 的实例 seq，每个seq生成时会有一个唯一的编号
将seq加入到 scheduler 的一个 waiting队列，类型是dequeue，双头队列

```

然后进入循环

```python
		while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()	#[源码学习]核心部分，进行一次完整推理
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
```



## engine-->step

```python
	def step(self):
        seqs, is_prefill = self.scheduler.schedule()	#[源码学习] 这里进行prefill和decode操作
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
```



### schedule()

```bash
/home/ziyi.he/Code/nano-vllm/nanovllm/engine/scheduler.py
```

```python
def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False
```



