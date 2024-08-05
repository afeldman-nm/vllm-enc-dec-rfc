# vLLM encoder/decoder infrastructure overview

<figure>
  <p float="left">
    <img src="img/enc_dec_model_arch_prefill.png" alt="Encoder/decoder architecture (prefill phase)" width="45%" style="margin-right:10px;" />
    <img src="img/enc_dec_model_arch_decode.png" alt="Encoder/decoder architecture (decode phase)" width="45%" />
  </p>
  <figcaption style="text-align: center; margin-top: 10px;">
    <strong>Figure 1:</strong> Encoder/decoder architecture during the prefill and decode phases. Encoder layers are abstracted as gray boxes, while decoder layers are blown-up to show how self- and cross-attention utilize KV caching. The KV caches shown are the decoder self-attn cache (blue; "Self") and the encoder/decoder cross-attn cache (orange; "Cross".) Although the model architecture does not change *per se* between the prefill and decode phases, nonetheless the encoder is omitted in the decode-phase diagram because all computations on the encoder hidden states are handled by the cross-attention KV cache.
  </figcaption>
</figure>

## Encoder/decoder request processing pipeline

[This page](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html#input-processing-pipeline) introduces the vLLM input processing pipeline.

Encoder/decoder models impact the behavior at each pipeline stage.

### 1. Input data is passed to `LLMEngine` (or `AsyncLLMEngine`)

A single vLLM API call may pass one request or a set of requests to the vLLM engine; vLLM expects a decoder-only model request to have a single input prompt (this is effectively true even for decoder-only multi-modal models.) 

In contrast, there are naturally two submodules (encoder and decoder) in an encoder/decoder model, which can both accept an input prompt. The encoder prompt is typically the "primary" input associated with the model's intended workload or functionality, however the decoder prompt is commonly used to tune model behavior, especially through the use of special tokens. Whisper [^1] accepts preprocessed audio embeddings as the encoder input "prompt", yet the model's configuration settings (language, speech-recognition task, etc.) are mapped to control tokens in the decoder prompt. 

Thus, it must be possible for an encoder/decoder inference request to specify both encoder and decoder prompts, a feature which was not previously supported by vLLM.

#### Multimodal inputs

#### Supported encoder/decoder prompt formats

The encoder/decoder infrastructure PRs enable the following encoder/decoder request formats:

* Single encoder prompt

    The decoder prompt is implicitly `None`.

    * Single encoder prompt string

    ```
    "The rain in spain falls mainly on the"
    ```

    * Single encoder `TextPrompt` with prompt string and optional multi-modal inputs

    ```
    TextPrompt(
        'prompt': "The rain in spain falls mainly on the",
        'multi_modal_data': {
            ...
        }
    )
    ```

    * Single encoder `TokensPrompt` with prompt tokens and optional multi-modal inputs

    ```
    TokensPrompt(
        'prompt_tokens': [2,0,171,5,2],
        'multi_modal_data': {
            ...
        }
    )
    ```

* Explicit encoder & decoder prompts

    ```
    ExplicitEncoderDecoderPrompt(
        'encoder_prompt': <any prompt>,
        'decoder_prompt': <any prompt without multi-modal>
    )
    ```

    For example:

    ```
    ExplicitEncoderDecoderPrompt(
        'encoder_prompt': TextPrompt(
                                'prompt': "The rain in spain falls mainly on the",
                                'multi_modal_data': {
                                    ...
                                }
                            ),
        'decoder_prompt': "<BOS>"
    )
    ```

    The syntax for passing one or more prompts to `LLMEngine` or `AsyncLLMEngine` is unchanged.

### 2. Tokenize the data if necessary.

Both the encoder and the decoder prompts are tokenized, unless token ids are provided directly.

### 3. Process the inputs

Apply special processing to the decoder prompt:
* If decoder prompt is `None`, replace with default "empty" decoder prompt.
* Append decoder start token at the beginning of the tokenized decoder prompt, unless
  it is already present.

### 4. Send the processed inputs to ExecutorBase.

### 5. Distribute the inputs via WorkerBase to ModelRunnerBase.

### 6. If the data contains multi-modal data, convert it into keyword arguments using MULTIMODAL_REGISTRY.map_input.

## Memory management

### vLLM memory profiling

### How `Sequence`, `SequenceGroup`, `SequenceGroupMetadata`, and block tables are impacted by encoder/decoder

### How block manager v1 swaps sequence groups between GPU and CPU memory

#### Allocate/free/reset

Allocation is performed at the granularity of a sequence group & provisions GPU memory for (1) the decoder self-attention KV cache block table, for each decoder sequence in the sequence group and (2) the single encoder/decoder cross-attention KV cache block table in the sequence group.

```
# Sequence group allocation example
# (Note: the ordering of the physical block layout is not necessarily as shown)

# GPU memory before allocation
[M free blocks]

# GPU memory after allocation
[# blocks = len(cross-attn block table)]
[# blocks = len(seq_0 decoder self-attn block table)]
...
[# blocks = len(seq_n decoder self-attn block table)]
[M' free blocks]
```

Where $M^\prime = M - |cross.attn.blocktable| - \sum_{i}{|seq_{i}.decoder.self.attn.block.table|}$

#### Swap-in (CPU -> GPU), swap-out (GPU -> CPU)

## Engine & scheduler modifications

## Attention backend modifications

### Default encoder/decoder attention bias (or mask)

Two factors which complicate scaled dot-product (SDP) attention computation in the vLLM backend are:

1. For an $N$-sequence batch, vLLM passes the model a single token vector which is the concatenation of the $N$ sequences (without padding), and which has a total number of tokens equal to the sum of the token-counts of all $N$ sequences. The sequences must be passed to the `Attention` layers as such a single vector, **but the sequences may not attention to each other**.

2. (Encoder/decoder only) By default (i.e. unless a particular model specifies otherwise), non-causal attention is employed for encoder attention & encoder/decoder cross-attention, while causal attention is employed for decoder self-attention.

vLLM addresses both requirements by augmenting SDP attention with a *causal* or *non-causal block-diagonal attention mask*. The SDP attention computation may be augmented with a *bias* or *mask* matrix $A$:

$$
attn(Q,K,V,A) = softmax(\frac{Q K^T + A}{\sqrt{d}})V
$$


Currently, vLLM does not support *arbitrary* $A$ matrices (this is the focus of the [custom attention bias](./rfc.md#support-custom-attention-bias) workstream.) However, all vLLM attention backends support explicit (XFormers via `AttentionBias` class) or implicit (Flash-attention, Flashinfer via sequence start location arguments) configuration of a block-diagonal mask.

The SDP attention score computation $Q K^T$ yields an attention score matrix; the white regions in Figure 2 reflect the portions of the attention score matrix corresponding to inter-sequence attention (which is undesirable), while the black regions correspond to within- or intra-sequence attention (desirable.) A block-diagonal attention mask $A$ prevents inter-sequence attention, provided that it is always equal to $-\infty$ in the white regions, as shown in Figure 2.



<figure>
  <div style="display: flex; justify-content: space-between; align-items: center;">
    <img src="img/enc_attn_mask.png" alt="Block-diagonal encoder attention mask, alongside encoder attention layer Q & K" style="width:30%; margin-right: 10px;" />
    <img src="img/dec_self_attn_mask.png" alt="Block-diagonal decoder self-attention mask, alongside decoder self-attention layer Q & K" style="width:30%; margin-right: 10px;" />
    <img src="img/enc_dec_cross_attn_mask.png" alt="Block-diagonal encoder/decoder cross-attention mask, alongside cross-attention layer Q (derived from previous decoder self-attention layer output hidden states) and K (derived from encoder output hidden states.)" style="width:30%;" />
  </div>
  <figcaption style="text-align: center; margin-top: 10px;">
    <strong>Figure 2:</strong> Description of the three images showing different phases of the process.
  </figcaption>
</figure>


## BART integration

Sources/notes:

[^1]: [Whisper](https://cdn.openai.com/papers/whisper.pdf) is a multi-modal encoder/decoder speech-recognition model.