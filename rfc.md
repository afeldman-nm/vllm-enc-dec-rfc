# [RFC] Encoder/decoder models & feature compatibility

tl;dr With Encoder/decoder model support landing soon, the next steps are to (1) add frequently-asked-for models (T5, Whisper) and (2) make vLLM features (CUDAGraph, pipeline parallelism, all attn backends, ...) compatible with encoder/decoder

## Motivation

The level of interest in encoder/decoder models has resulted in a number of Issues submitted to the vLLM github repo, for example [here](https://github.com/vllm-project/vllm/issues/187) and [here](https://github.com/vllm-project/vllm/issues/180). As a result encoder/decoder support is being introduced to vLLM, requiring modifications to the backends, core & engine which were introduced over three PRs:

* [Core] block manager support for cross-attention KV cache
* [Kernel] backend support for encoder attention & cross-attention
* [Core] scheduler & engine support for encoder/decoder requests
    * Add BART model

These three PRs make encoder/decoder model inference possible, but leave more to be desired in terms of feature compatibility with encoder/decoder & the number of encoder/decoder models which are supported.

The support matrix below summarizes what models & features will be supported initially, versus what are the long-term goals:

<table>
  <tr>
    <th>Model/feature</th>
    <th>Initially available & compatible when encoder/decoder support lands?</th>
    <th>Long-term goal?</th>
  </tr>
  <tr>
    <td>Encoder/decoder infrastructure</td>
    <td><strong><u>Yes</u></strong></td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>BART</td>
    <td><strong><u>Yes</u></strong></td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>Whisper</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>T5</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>Other enc/dec models</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>Multi-modality</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>Kernels other than Xformers (esp. flash-attn, flashinfer)</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>Custom bias support</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>CUDAGraph</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>Pipeline parallelism</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>Speculative decoding</td>
    <td>No</td>
    <td><strong>Low-priority but nice-to-have; difficult.</strong></td>
  </tr>
  <tr>
    <td>Automatic prefix caching</td>
    <td>No</td>
    <td><strong>Low-priority; difficult.</strong></td>
  </tr>
  <tr>
    <td>Sliding window</td>
    <td>No</td>
    <td>No</td>
  </tr>
  <tr>
    <td>Chunked prefill</td>
    <td>No</td>
    <td>No</td>
  </tr>
  <tr>
    <td>LoRA</td>
    <td>No</td>
    <td>No</td>
  </tr>
</table>

This RFC discusses the effort to support those **"Long-term goal"** capabilities in the third column of the table above, specifically those which are not supported in the initial encoder/decoder infrastructure PRs.

## TODOs

### Add Whisper model

#### Add support for multi-modality

### Add T5 model

#### Add support for custom bias

Not directly encoder/decoder related

### Add other encoder/decoder models

* Variants of aforementioned models (BART, T5, Whisper)
* CogAgent

### Support kernels other than XFormers with encoder/decoder models

### Support CUDAGraph with encoder/decoder models

### Support pipeline-parallelism with encoder/decoder models

### Low-priority, high-effort tasks

* Speculative decoding
* Automatic prefix caching

## Overview of initial encoder/decoder infrastructure

### Encoder/decoder request processing pipeline

[This page](https://docs.vllm.ai/en/latest/dev/input_processing/input_processing_pipeline.html#input-processing-pipeline) introduces the vLLM input processing pipeline.

Encoder/decoder models impact the behavior at each pipeline stage.

#### 1. Input data is passed to `LLMEngine` (or `AsyncLLMEngine`)

A single vLLM API call may pass one request or a set of requests to the vLLM engine; vLLM expects a decoder-only model request to have a single input prompt (this is effectively true even for decoder-only multi-modal models.) 

In contrast, there are naturally two submodules (encoder and decoder) in an encoder/decoder model, which can both accept an input prompt. The encoder prompt is typically the "primary" input associated with the model's intended workload or functionality, however the decoder prompt is commonly used to tune model behavior, especially through the use of special tokens. Whisper [^1] accepts preprocessed audio embeddings as the encoder input "prompt", yet the model's configuration settings (language, speech-recognition task, etc.) are mapped to control tokens in the decoder prompt. 

Thus, it must be possible for an encoder/decoder inference request to specify both encoder and decoder prompts, a feature which was not previously supported by vLLM.

##### Multimodal inputs

##### Supported encoder/decoder prompt formats

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

#### 2. Tokenize the data if necessary.

Both the encoder and the decoder prompts are tokenized, unless token ids are provided directly.

#### 3. Process the inputs

Apply special processing to the decoder prompt:
* If decoder prompt is `None`, replace with default "empty" decoder prompt.
* Append decoder start token at the beginning of the tokenized decoder prompt, unless
  it is already present.

#### 4. Send the processed inputs to ExecutorBase.

#### 5. Distribute the inputs via WorkerBase to ModelRunnerBase.

#### 6. If the data contains multi-modal data, convert it into keyword arguments using MULTIMODAL_REGISTRY.map_input.

### Engine & scheduler modifications


#### Prompting an encoder/decoder model

### BART integration

## Implementation guides

### Guide to adding new encoder/decoder models to vLLM

### Guide to adding encoder/decoder support to existing vLLM backends

* Identify impacted AttentionMetadata fields
* Use of encoder sequence length
* Correct usage of context
* Prefill attention masks
* ModelRunner kernel enforcement

Sources/notes:

[^1]: [Whisper](https://cdn.openai.com/papers/whisper.pdf) is a multi-modal encoder/decoder speech-recognition model.