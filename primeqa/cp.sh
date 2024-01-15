# 创建目标目录（如果尚不存在）
mkdir -p ~/xionglian/primeqa_test/models--xlm-roberta-base/

# 复制pytorch_model.bin
cp ~/.cache/huggingface/hub/models--xlm-roberta-base/snapshots/77de1f7a7e5e737aead1cd880979d4f1b3af6668/pytorch_model.bin ~/xionglian/primeqa_test/models--xlm-roberta-base

# 复制sentencepiece.bpe.model
cp ~/.cache/huggingface/hub/models--xlm-roberta-base/snapshots/77de1f7a7e5e737aead1cd880979d4f1b3af6668/sentencepiece.bpe.model ~/xionglian/primeqa_test/models--xlm-roberta-base

# 复制tokenizer.json (如果需要)
cp ~/.cache/huggingface/hub/models--xlm-roberta-base/snapshots/77de1f7a7e5e737aead1cd880979d4f1b3af6668/tokenizer.json ~/xionglian/primeqa_test/models--xlm-roberta-base

# 复制config.json (如果需要)
cp ~/.cache/huggingface/hub/models--xlm-roberta-base/snapshots/77de1f7a7e5e737aead1cd880979d4f1b3af6668/config.json ~/xionglian/primeqa_test/models--xlm-roberta-base

