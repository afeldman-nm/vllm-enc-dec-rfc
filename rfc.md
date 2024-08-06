# [RFC] Encoder/decoder models & feature compatibility

tl;dr With Encoder/decoder model support landing soon, the next steps are to (1) add frequently-asked-for models (T5, Whisper, ...) and (2) increase the number of vLLM features (quantization, CUDAGraph, pipeline parallelism, all attn backends, ...) compatible with encoder/decoder

## Motivation

There is significant interest in vLLM supporting encoder/decoder models. [Issues 187](https://github.com/vllm-project/vllm/issues/187) and [180](https://github.com/vllm-project/vllm/issues/180), for example, request encoder/decoder model support. As a result encoder/decoder support will be introduced to vLLM via the following three PRs:

* **(Merged)** [[Core] Cross-attention KV caching and memory-management](https://github.com/vllm-project/vllm/pull/4837)
* **(Merged)** [[Kernel] Correctly invoke prefill & decode kernels for cross-attention](https://github.com/vllm-project/vllm/pull/4888)
* **(Merged)** [[Core] Subclass ModelRunner to support cross-attention & encoder sequences](https://github.com/vllm-project/vllm/pull/4942)

These three PRs make encoder/decoder model inference possible, but leave more to be desired in terms of feature compatibility with encoder/decoder & the number of encoder/decoder models which are supported.

The ask for the vLLM contributor community is to help bring vLLM encoder/decoder model support to a similar level of maturity as that of decoder-only models, by contributing PRs which fill gaps in vLLM encoder/decoder model and feature support.

## Proposed changes

The support matrix below summarizes which features & encoder/decoder models will be supported initially (by the three PRs mentioned above), versus which features & models will require community support to implement in the long term:

<table>
  <tr>
    <th>Model/feature</th>
    <th>Initially supported with encoder/decoder models?</th>
    <th>Is supporting this feature a long-term goal?</th>
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

This RFC gives an overview of those features & models which **will not be compatible with encoder/decoder initially, but which should be made compatible eventually** (i.e. **No** in the 2nd column, **Yes** in the third column in the support matrix.)

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

Steps to add support for Whisper [^1], a multi-modal encoder/decoder speech recognition model.
* [Extend existing vLLM multimodality support to encoder/decoder models](#support-encoderdecoder-multimodality)
* Extend existing vLLM prompt processing pipline to support audio
* Port HuggingFace Whisper model [^2] to vLLM; an existing open PR for this workstream can be found here [^3]
* Modify each Whisper layer, where appropriate, to support TP > 1
* Add a Whisper test under `tests/models/`

Proposal: it makes sense to implement encoder/decoder multimodality, audio support, and Whisper in the same PR; that way, the Whisper model may be used to facilitate an end-to-end test with of audio multimodality.

#### Add T5 model

Note: T5 depends on [custom attention bias being supported](#support-custom-attention-bias) by at least one of the attention backends which [also supports encoder attention & cross-attention](#add-support-for-encoder-attention-and-cross-attention-to-additional-backends); at time of writing this is not the case, since XFormers backend supports encoder/decoder models but no backend supports custom attention bias. (Custom attention bias is required in order to support T5 [relative positional encoding.](#custom-attention-bias-and-relative-positional-encoding))

Steps to add support for the T5 model [^4].
* Port HuggingFace T5 model [^5] to vLLM
  * This includes porting over the method which computes the custom attention bias matrix for T5 relative position encoding
* Modify each T5 layer, where appropriate, to support TP > 1
  * The custom attention bias computation must also support TP > 1
* Add a T5 test to `tests/models/`

#### Add other encoder/decoder models

* Variants of aforementioned models (BART, T5, Whisper)
* CogAgent

### Support encoder/decoder multimodality

Extend existing vLLM multimodality support to encoder/decoder models.
* Support `multi_modal_data` field in vLLM encoder/decoder input prompts
* Support multi-modal data in encoder/decoder processing pipeline
* Add one or more unit tests with multi-modal data & encoder/decoder models

Proposal: it makes sense to implement encoder/decoder multimodality in the same PR as adding the [Whisper model.](#add-whisper-model--multi-modality)

### Support custom attention bias

Note: [T5](#add-t5-model) takes a dependency on custom attention bias. Custom attention bias is likely complex enough to merit its own PR.

Custom attention bias is not directly related to encoder/decoder functionality, however custom attention bias support is required by [T5 which is a frequently-requested encoder/decoder model](#add-t5-model).

#### Custom attention bias and relative positional encoding

Attention bias refers to adding a matrix $A$ to the scaled dot-product (SDP) attention scores matrix before performing softmax, as shown below:

$$
attn(Q,K,V,A) = softmax(\frac{Q K^T + A}{\sqrt{d}})V
$$

Here, *custom* attention bias is understood to mean that the vLLM attention backend allows $A$ to be an arbitrary PyTorch tensor, provided the tensor dimensions are commensurate with the shape of the SDP attention scores matrix.

T5 employs custom attention bias in order to implement relative positional encoding [^8], wherein pairwise positional relationships between tokens are represented by the bias matrix. The HuggingFace Transformers T5 implementation provides an example of how the relative positional encoding matrix is computed [^9].

#### Existing attention bias support

*Currently, no vLLM attention backend fully supports custom attention bias*. This is because most attention kernels employed by vLLM allow attention bias to be specified only in an indirect or "compressed" manner, i.e. through the use of a `causal=True/False` flag (causal attention mask being a type of attention bias.) The xFormers `memory_efficient_attention_forward` kernel[^7] is the exception, in that it permits an arbitrary pytorch tensor to be passed in via the `attn_bias` argument. However vLLM only employs this kernel for prefill; none of the decode-phase kernels employed by vLLM can accept an arbitrary pytorch tensor as a custom attention bias, making custom attention bias impossible to apply end-to-end for both prefill and decode under the current vLLM implementation.

In addition to lack of kernel-level support for custom attention bias, most vLLM backends also prevent passing a custom attention bias matrix to the underlying kernel. The exception is the XFormers backend, which accepts an attention bias via `XFormersMetadata.attn_bias` attribute (however the XFormers backend only utilizes `attn_bias` in the prefill phase.)

#### Initial goals for introducing custom attention bias support

1. Focus on a particular vLLM attention backend
  * Suggestion: focus on an attention backend which also supports encoder/decoder models, in order to facilitate [running T5](#add-t5-model). At time of writing, XFormers is the only backend which supports encoder/decoder models, however there will likely be work on [supporting encoder/decoder in additional attention backends.](#add-support-for-encoder-attention-and-cross-attention-to-additional-backends)
2. Scope out the effort involved in introducing custom attention bias support to this backend
3. Some steps which will likely be involved in introducing custom attention bias support:
  * Augment attention backend's kernels to accept custom attention bias; for example, the PagedAttention kernel (for XFormers backend), the Flash-attention kernel (for the flash-attn backend), or the Flashinfer kernels (for the Flashinfer backend)
  * (Except for XFormers) add an `attn_bias` attribute to attention backend's `AttentionMetadata` subclass
  * Ensure that the attention backend passes the `attn_bias` attribute to both the prefill and decode kernels
4. Add at least two custom attention bias unit tests (for prefill & decode respectively)

#### Final goals for introducing custom attention bias support

* All vLLM attention backends should support custom attention bias, with unit tests

### Add support for encoder attention and cross-attention to additional backends

### Support CUDAGraph with encoder/decoder models

### Support pipeline-parallelism with encoder/decoder models

### Ensure full support for quantization with encoder/decoder models



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