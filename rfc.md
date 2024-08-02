# [RFC] Encoder/decoder models & feature compatibility

tl;dr With Encoder/decoder model support landing soon, the next steps are to (1) add frequently-asked-for models (T5, Whisper, ...) and (2) increase the number of vLLM features (quantization, CUDAGraph, pipeline parallelism, all attn backends, ...) compatible with encoder/decoder

## Motivation

The level of interest in encoder/decoder models has resulted in a number of Issues submitted to the vLLM github repo, for example [here](https://github.com/vllm-project/vllm/issues/187) and [here](https://github.com/vllm-project/vllm/issues/180). As a result encoder/decoder support is being introduced to vLLM over three PRs which are expected to land soon:

* [Core] block manager support for cross-attention KV cache
* [Kernel] backend support for encoder attention & cross-attention
* [Core] scheduler & engine support for encoder/decoder requests
    * Add BART model

These three PRs make encoder/decoder model inference possible, but leave more to be desired in terms of feature compatibility with encoder/decoder & the number of encoder/decoder models which are supported.

## Proposed changes

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
    <td>Quantization</td>
    <td><strong><u>Untested</u></strong></td>
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

This RFC discusses the effort to support those **"Long-term goal"** capabilities in the third column of the table above, specifically those which are not supported or fully-tested in the initial encoder/decoder infrastructure PRs.

## Background

Before continuing, it will be helpful to review [the details of the new vLLM encoder/decoder infrastructure](infra-enc-dec.md). 

It will also be helpful to review [this how-to guide](how-to.md) for adding new encoder/decoder models & improving encoder/decoder feature compatibility.

## Detailed goals

### Add new models to vLLM

Please review the [how-to guide for adding new models to vLLM](how-to.md#guide-to-adding-new-encoderdecoder-models-to-vllm)

#### Add Whisper model & multi-modality



#### Add T5 model & custom bias

#### Add support for custom bias

Not directly encoder/decoder related

#### Add support for other encoder/decoder models

* Variants of aforementioned models (BART, T5, Whisper)
* CogAgent

### Support kernels other than XFormers with encoder/decoder models

### Support CUDAGraph with encoder/decoder models

### Support pipeline-parallelism with encoder/decoder models

### Low-priority, high-effort tasks

* Speculative decoding
* Automatic prefix caching

## Feedback period

## CC list

## Any other things

*No response*

Sources/notes:

[^1]: [Whisper paper](https://cdn.openai.com/papers/whisper.pdf)
[^2]: [`modeling_whisper` on huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py)