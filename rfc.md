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
    <th>Initially supported w/ encoder/decoder?</th>
    <th>Support is a goal?</th>
  </tr>
  <tr>
    <td>Encoder/decoder infrastructure</td>
    <td><strong>Yes</strong></td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>BART</td>
    <td><strong>Yes</strong></td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>Whisper</td>
    <td>No</td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>T5</td>
    <td>No</td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>Other enc/dec models</td>
    <td>No</td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>Multi-modality</td>
    <td>No</td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>Kernels other than Xformers (esp. flash-attn, flashinfer)</td>
    <td>No</td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>Custom bias support</td>
    <td>No</td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>CUDAGraph</td>
    <td>No</td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>Pipeline parallelism</td>
    <td>No</td>
    <td><strong>Yes</strong></td>
  </tr>
  <tr>
    <td>Prefix caching</td>
    <td>No</td>
    <td>No</td>
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
  <tr>
    <td>Speculative decoding</td>
    <td>No</td>
    <td>No</td>
  </tr>
</table>

These three PRs enable encoder/decoder models, with the following caveats
* XFormers backend only
* The following vLLM features are **incompatible** with existing encoder/decoder  infrastructure:
    * Prefix caching
    * Sliding window
    * Chunked prefill
    * LoRA
    * CUDAGraph
    * Kernels other than XFormers
    * Speculative decoding
    * Pipeline parallelism
    * Logits soft cap (requires FlashInfer backend)
    * Multi-modal models

## Add Whisper model

## Add support for custom bias

## Add T5 model

### Add support for custom bias

## Make CUDAGraph compatible with encoder/decoder models

