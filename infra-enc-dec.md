# vLLM encoder/decoder infrastructure overview

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
[# blocks = len(seq_n decoder self-attn block table)]=
[M' free blocks]
```

Where $M^\prime = M - |cross\_attn\_blocktable| - \sum_i{|seq_i\_decoder\_self\_attn\_block\_table|}$

#### Swap-in (CPU -> GPU), swap-out (GPU -> CPU)

## Engine & scheduler modifications

## Attention backend modifications

## BART integration

Sources/notes:

[^1]: [Whisper](https://cdn.openai.com/papers/whisper.pdf) is a multi-modal encoder/decoder speech-recognition model.