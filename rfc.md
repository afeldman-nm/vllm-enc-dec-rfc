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