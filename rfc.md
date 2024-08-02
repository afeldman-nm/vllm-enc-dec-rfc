# [RFC] Encoder/decoder models & feature compatibility

tl;dr With Encoder/decoder model support landing soon, the next steps are to (1) add frequently-asked-for models (T5, Whisper, ...) and (2) increase the number of vLLM features (quantization, CUDAGraph, pipeline parallelism, all attn backends, ...) compatible with encoder/decoder

## Motivation

The level of interest in encoder/decoder models has resulted in a number of Issues submitted to the vLLM github repo, for example [here](https://github.com/vllm-project/vllm/issues/187) and [here](https://github.com/vllm-project/vllm/issues/180). As a result encoder/decoder support is being introduced to vLLM over three PRs which are expected to land soon:

* [Core] block manager support for cross-attention KV cache
* [Kernel] backend support for encoder attention & cross-attention
* [Core] scheduler & engine support for encoder/decoder requests
    * Add BART model

These three PRs make encoder/decoder model inference possible, but leave more to be desired in terms of feature compatibility with encoder/decoder & the number of encoder/decoder models which are supported.

The ask is for the vLLM contributor community to help bring vLLM encoder/decoder support to a similar level of maturity as decoder-only, by contributing PRs which fill gaps in vLLM model and feature support.

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
    <td>custom attention bias support</td>
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

## Initial goal

Members of the vLLM contributor community identify models/features in the support matrix above, for which they will work on writing a PR.

## Detailed long-term goals

### Add new models to vLLM

Please review the [how-to guide for adding new models to vLLM](how-to.md#guide-to-adding-new-encoderdecoder-models-to-vllm)

See `tests/models/test_bart.py` for an example of an encoder/decoder model unit test. See `tests/distributed/test_basic_distributed_correctness_enc_dec.py` for an example of an encoder/decoder model test with TP > 1.

#### Add Whisper model & multi-modality

Add support for Whisper [^1], a multi-modal encoder/decoder speech recognition model.
* [Extend existing vLLM multimodality support to encoder/decoder models](#support-encoderdecoder-multimodality)
* Extend existing vLLM prompt processing pipline to support audio
* Port HuggingFace Whisper model [^2] to vLLM; an existing open PR for this workstream can be found here [^3]
* Add a Whisper test under `tests/models/`

Proposal: it makes sense to implement encoder/decoder multimodality, audio support, and Whisper in the same PR; that way, the Whisper model may be used to facilitate an end-to-end test with the other two features. 

#### Add T5 model

Note: T5 depends on [custom attention bias being supported](#support-custom-attention-bias) by at least one of the attention backends which [also supports encoder attention & cross-attention](#add-support-for-encoder-attention-and-cross-attention-to-additional-kernels); at time of writing this is not the case. Custom attention bias is required in order to support T5 [relative positional encoding.](#custom-attention-bias-and-relative-positional-encoding)

Add support for the T5 model [^4].
* Port HuggingFace T5 model [^5] to vLLM
* Add a T5 test to `tests/models/`

#### Add other encoder/decoder models

* Variants of aforementioned models (BART, T5, Whisper)
* CogAgent

### Support encoder/decoder multimodality

Extend existing vLLM multimodality support to encoder/decoder models.
* Support `multi_modal_data` field in vLLM encoder/decoder input prompts
* Support multi-modal data in encoder/decoder processing pipeline
* Add a one or more unit tests with multi-modal data & encoder/decoder models

Proposal: it makes sense to implement encoder/decoder multimodality in the same PR as adding the [Whisper model.](#add-whisper-model--multi-modality)

### Support custom attention bias

Add custom attention bias support to vLLM kernels. 

This is not directly related to encoder/decoder functionality, however custom attention bias support is required by T5 [^4] which is a frequently-requested model.

#### Custom attention bias and relative positional encoding

Custom attention bias means refers to adding an arbitrary matrix $A$ to the scaled dot-product attention scores before performing softmax, as shown below:

$$
attn(Q,K,V,A) = softmax(\frac{Q K^T + A}{\sqrt{d}})V
$$

T5 employs custom attention bias in order to implement relative positional encoding [^8], wherein pairwise positional relationships between tokens are represented by the bias matrix. The HuggingFace Transformers T5 implementation provides an example of how the relative positional encoding matrix is computed [^9].

#### Existing attention bias support

**Currently, no vLLM attention backend fully supports custom attention bias**. This is because most attention kernels employed by vLLM allow attention bias to be specified only in an indirect or "compressed" manner, i.e. through the use of a `causal=True/False` flag (causal attention mask being a type of attention bias.) The xFormers `memory_efficient_attention_forward` kernel[^7] is the exception, in that it permits an arbitrary pytorch tensor to be passed in via the `attn_bias` argument. However vLLM only employs this kernel for prefill; none of the decode-phase kernels employed by vLLM can accept an arbitrary pytorch tensor as a custom attention bias, making custom attention bias impossible to apply end-to-end for both prefill and decode. 

An overview of how attention bias is currently handled by a subset of vLLM backends:

* Prefill
  * xFormers backend: `BlockDiagonalMask` and `BlockDiagonalCausalMask` are used as compressed representations of non-causal and causal attention masks, respectively [^11]
  * Flash-attention backend: the `causal` flag controls whether a non-causal or causal attention bias is employed [^10]
  * Flashinfer backend: similar usage of a `causal` flag [^12]
* Decode
  * Autoregressive decoding is inherently causal. So the xFormers backend (which employs the PagedAttention kernel for decode) [^13] and FlashInfer [^14] simplify by omitting the `causal` flag entirely, since there is no support for a more general attention bias beyond causal or non-causal masking.
  * Flash-attention backend uses the `causal=True` setting

Additionally, the vLLM `Attention` wrapper does not currently expose a custom attention bias argument which would allow arbitrary pytorch tensors to pass into the attention kernel.

#### Adding custom attention bias support

Note: [T5](#add-t5-model) takes a dependency on custom attention bias. Custom attention bias is likely complex enough to merit its own PR.



### Add support for encoder attention and cross-attention to additional kernels

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

[^1]: [Whisper on paperswithcode](https://paperswithcode.com/paper/robust-speech-recognition-via-large-scale-1f)

[^2]: [`modeling_whisper` on HuggingFace](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py)

[^3]: [Open PR which added Whisper model](https://github.com/vllm-project/vllm/pull/5964)

[^4]: [T5 on paperswithcode](https://paperswithcode.com/method/t5)

[^5]: [`modeling_t5` on HuggingFace](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)

[^6]: [Open PR which added T5 model & paged-attn custom attention bias](https://github.com/vllm-project/vllm/pull/3117)

[^7]: [xFormers optimized operators](https://facebookresearch.github.io/xformers/components/ops.html)

[^8]: [Relative attention bias (relative positional encoding)](https://jaketae.github.io/study/relative-positional-encoding/#bridging-shaw-and-huang) in the sense of Huang, et. al.

[^9]: [Relative positional encoding implemented for T5 on HuggingFace transformers](https://github.com/huggingface/transformers/blob/c1aa0edb48217f416f4bbe6e3a9db1500284513b/src/transformers/models/t5/modeling_t5.py#L428)

[^10]: [Invocation of flash-attention for prefill in vLLM backend, using `causal` flag](https://github.com/vllm-project/vllm/blob/db35186391a2abfc6c91d703527dac20d2488107/vllm/attention/backends/flash_attn.py#L522)

[^11]: [Invocation of xFormers attention kernel for prefill in vLLM backend, using `BlockDiagonalMask` and `BlockDiagonalCausalMask`](https://github.com/vllm-project/vllm/blob/db35186391a2abfc6c91d703527dac20d2488107/vllm/attention/backends/xformers.py#L689-L738)

[^12]: [Invocation of FlashInfer attention kernel for prefill in backend, using `causal` flag](https://github.com/vllm-project/vllm/blob/db35186391a2abfc6c91d703527dac20d2488107/vllm/attention/backends/flashinfer.py#L539)

[^13]: [Invocation of PagedAttention kernel for decode in vLLM backend](https://github.com/vllm-project/vllm/blob/db35186391a2abfc6c91d703527dac20d2488107/vllm/attention/backends/xformers.py#L628)

[^14]: [Invocation of FlashInfer kernel for decode in vLLM backend](https://github.com/vllm-project/vllm/blob/db35186391a2abfc6c91d703527dac20d2488107/vllm/attention/backends/flashinfer.py#L543)