# [RFC] Encoder/decoder models & feature compatibility

tl;dr Now that encoder/decoder model support has landed in vLLM, the next steps are to (1) add support for frequently-requested models (T5, Whisper, ...) and (2) increase the number of pre-existing vLLM features (quantization, CUDAGraph, pipeline parallelism, all attn backends, ...) which are compatible with encoder/decoder models. The ask for the vLLM contributor community is to help with this process.

## Motivation

There is significant interest in vLLM supporting encoder/decoder models. [Issues 187](https://github.com/vllm-project/vllm/issues/187) and [180](https://github.com/vllm-project/vllm/issues/180), for example, request encoder/decoder model support. As a result encoder/decoder support was recently introduced to vLLM via the following three PRs:

* **(Merged)** [[Core] Cross-attention KV caching and memory-management](https://github.com/vllm-project/vllm/pull/4837)
* **(Merged)** [[Kernel] Correctly invoke prefill & decode kernels for cross-attention](https://github.com/vllm-project/vllm/pull/4888)
* **(Merged)** [[Core] Subclass ModelRunner to support cross-attention & encoder sequences](https://github.com/vllm-project/vllm/pull/4942)

These three PRs make encoder/decoder model inference possible; however, they leave more to be desired in terms of (1) parity between vLLM's decoder-only & encoder/decoder request processing pipelines, in terms of which features are supported, and (2) the number of encoder/decoder models which are supported.

The ask for the vLLM community is to contribute PRs which help bring vLLM encoder/decoder functionality to a similar level of maturity as that of vLLM's decoder-only functionality.

## Proposed changes

The support matrix below summarizes which encoder/decoder models have already been added & which features are currently compatible with the vLLM encoder/decoder pipeline, versus which features & models will require additional PRs to implement in the long-term:

<table>
  <tr>
    <th>Model/feature</th>
    <th>Model is already available/feature is already compatible with encoder-decoder?</th>
    <th>Having this model/making this feature compatible is a long-term goal?</th>
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
    <td>Multimodality</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>Kernels other than Xformers (esp. flash-attn, flashinfer)</td>
    <td>No</td>
    <td><strong><u>Yes</u></strong></td>
  </tr>
  <tr>
    <td>Custom attention bias support</td>
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

Note that there are features (automatic prefix caching/sliding window/chunked prefill/LoRA) which are not long-term compatibility goals.

## Background

Before continuing, it will be helpful to review [the details of the new vLLM encoder/decoder infrastructure](https://github.com/afeldman-nm/vllm-enc-dec-rfc/blob/main/infra-enc-dec.md). 

It will also be helpful to review [this how-to guide](https://github.com/afeldman-nm/vllm-enc-dec-rfc/blob/main/how-to.md) for adding new encoder/decoder models & improving encoder/decoder feature compatibility.

## Initial goal

Members of the vLLM contributor community identify models/features in the support matrix above, for which they will work on writing a PR.

## Detailed long-term goals

### Quantization

The goal of this workstream is to make sure that quantization + encoder/decoder models is fully-tested, and to fill in any gaps (should they exist) in vLLM's support for quantized encoder/decoder models.

Steps to ensure that vLLM supports encoder/decoder models in combination with all existing vLLM quantization methods:

* Identify the list of quantization methods which vLLM currently supports with decoder-only models.
* Add unit tests for encoder/decoder models with all of these quantization methods.
* Determine which quantization methods are currently incompatible with vLLM encoder/decoder infrastructure.
* Scope out the effort involved in making these quantization methods compatible & submit a PR making the change.

vLLM encoder/decoder infrastructure should be compatible with most of the existing vLLM quantization methods, because the specialized quantization kernels are only employed for GEMM operations involving the learned weight matrices ($W_q$, $W_k$, etc.), whereas the encoder/decoder work really only modifies how the `Attention(q, k, v, kv_cache)` layer behaves & does not impact the learned weight matrices at all.

It is less clear whether vLLM encoder/decoder infrastructure would be incompatible with FP8. It does appear that a specialized quantized KV cache kernel is employed by the `Attention(q, k, v, kv_cache)` layer when FP8 quantization is employed.

### Support encoder/decoder multimodality

Technically, vLLM already supports multimodality for models which have an "encoder" and a "decoder", i.e. Llava. However, Llava's decoder does not utilize cross-attention & the model is basically compatible with vLLM's pre-existing decoder-only infrastructure.

But critically, for **encoder/decoder models with cross-attention** such as Whisper vLLM does not currently support multimodality of any sort. The processing pipeline does not extract or utilize multimodal data from the input prompt, and the `EncoderDecoderModelRunner` has an assert which fails if the multimodal config is not `None`. Addressing this is what is meant by "supporting encoder/decoder multimodality".

Steps to extend existing vLLM multimodality support to encoder/decoder models:
* Review [existing vLLM multimodality support in the decoder-only pipeline](https://docs.vllm.ai/en/latest/dev/multimodal/adding_multimodal_plugin.html#adding-multimodal-plugin)
* Scope out a plan for adding encoder/decoder multimodality support.
* Propose & implement one or more multimodal prompt formats for encoder/decoder models
* Integrate multimodality support into encoder/decoder processing pipeline
* Remove the assertion which fails when multimodality is enabled for an encoder/decoder model (see `assert_enc_dec_mr_supported_scenario()` in `vllm/worker/utils.py`)
* Add one or more unit tests with multimodal data

There are a number of multimodal encoder/decoder models which will benefit from this feature. One possibility is to add multimodality support & a multimodal model such as [Whisper](#add-whisper-model) in the same PR, so that Whisper may be used to facilitate an end-to-end test with multimodality.

Another possibility is to implement multimodality support in its own PR.

#### Considerations for designing multimodal encoder/decoder prompt formats

One approach to designing the vLLM multimodal encoder/decoder prompt formats, is to consider what we want the user experience to be for high-priority multimodal encoder/decoder models such as
* [Llama 3.1 multimodal](https://github.com/vllm-project/vllm/pull/7258#discussion_r1710915145)
* [Whisper](#add-whisper-model)

#### Initial proposal for multimodal encoder/decoder prompt formats

It may be helpful to review
* [The non-multimodal encoder/decoder prompt formats which are currently supported by vLLM](https://github.com/afeldman-nm/vllm-enc-dec-rfc/blob/main/infra-enc-dec.md#supported-encoderdecoder-prompt-formats): singleton prompts (raw text prompt, `TextPrompt`, `TokensPrompt`) as well as `ExplicitEncoderDecoder` prompts
* The multimodal decoder-only prompt formats which are currently supported by vLLM; search for `multi_modal_data` [here](https://github.com/vllm-project/vllm/blob/main/vllm/inputs/data.py) and also the [vLLM documentation on multimodality](https://docs.vllm.ai/en/latest/dev/multimodal/multimodal_index.html)

Generally speaking, in encoder/decoder models based on cross-attention, the non-text input modality is passed to the encoder as input. Conversely, any text prompt is typically passed to the decoder as a input prompt.

The following two encoder/decoder multimodal prompt formats are tentatively proposed:

* Singleton `TextPrompt` with `multi_modal_data` field
    * vLLM will extract the `multi_modal_data` and pass it to the encoder module
    * vLLM will extract the prompt text, tokenize it and pass the token-list to the *decoder* (note that this is the opposite of vLLM behavior for non-multimodal prompts, where the prompt text would be passed to the encoder.)

    For example passing the `TextPrompt` below to vLLM BART

    ```
    TextPrompt(
        'prompt': "The rain in spain falls mainly on the",
        'multi_modal_data': <multi modal data structure>
    )
    ```

    results in

    ```
    Encoder input: <multi modal data structure>
    Decoder prompt: "The rain in spain falls mainly on the"
    ```

* Singleton `TokensPrompt` with `multi_modal_data` field
    * vLLM will extract the `multi_modal_data` and pass it to the encoder module
    * vLLM will extract the token list and pass it unmodified to the *decoder* (note that this is the opposite of vLLM behavior for non-multimodal prompts, where the prompt tokens would be passed to the encoder.)

    For example passing the `TokensPrompt` below to vLLM BART

    ```
    TokensPrompt(
        'prompt_tokens': [2,0,171,5,2],
        'multi_modal_data': <multi modal data structure>
    )
    ```

    results in

    ```
    Encoder prompt: <multi modal data structure>
    Decoder prompt: [2,0,171,5,2]
    ```

It may also be worth considering whether or how to support
* `ExplicitEncoderDecoderPrompt`s with multimodality
* An input prompt format which encapsulates *only* multimodal encoder inputs, with no associated decoder text/tokens prompt (this would result in the decoder being passed a "default" or empty prompt.)

### Add new models to vLLM

Please review the [how-to guide for adding new models to vLLM](https://github.com/afeldman-nm/vllm-enc-dec-rfc/blob/main/how-to.md#guide-to-adding-new-encoderdecoder-models-to-vllm)

See `tests/models/test_bart.py` for an example of an encoder/decoder model unit test. See `tests/distributed/test_basic_distributed_correctness_enc_dec.py` for an example of an encoder/decoder model test with TP > 1.

#### Add Whisper model

Steps to add support for Whisper [^1], a multimodal encoder/decoder speech recognition model:
* [Extend existing vLLM multimodality support to encoder/decoder models](#support-encoderdecoder-multimodality)
* Extend existing vLLM prompt processing pipeline to support audio
* Port HuggingFace Whisper model [^2] to vLLM; an existing open PR for this workstream can be found here [^3]
* Modify each Whisper layer, where appropriate, to support TP > 1
* Add a Whisper test under `tests/models/`

Proposal: it makes sense to implement encoder/decoder multimodality, audio support, and Whisper in the same PR; that way, the Whisper model may be used to facilitate an end-to-end test with of audio multimodality.

#### Add T5 model

Note: T5 depends on [custom attention bias being supported](#support-custom-attention-bias) by at least one of the attention backends which [also supports encoder attention & cross-attention](#add-support-for-encoder-attention-and-cross-attention-to-additional-backends); at time of writing this is not the case, since XFormers backend supports encoder/decoder models but no backend supports custom attention bias. (Custom attention bias is required in order to support T5 [relative positional encoding.](#custom-attention-bias-and-relative-positional-encoding))

Steps to add support for the T5 model [^4]:
* Port HuggingFace T5 model [^5] to vLLM
  * This includes porting over the method which computes the custom attention bias matrix for T5 relative position encoding
* Modify each T5 layer, where appropriate, to support TP > 1
  * The custom attention bias computation must also support TP > 1
* Add a T5 test to `tests/models/`

#### Add other encoder/decoder models

* Variants of aforementioned models (BART, T5, Whisper)
* CogAgent

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

#### Proposed methods for supporting custom attention bias

Here the following two approaches for supporting custom attention bias in vLLM are proposed:
* **Fully-materialized bias matrix:** Modify vLLM attention backends to accept an arbitrary PyTorch tensor, passed into the backend via the `AttentionMetadata.attn_bias` field.
* **On-the-fly/fused bias matrix computation:** Enable an efficient workflow whereby vLLM developers can tweak an attention kernel to compute the custom attention bias on the fly
  * For example: rather than computing the T5 relative position encoder bias matrix once, instead the attention kernel can fuse the element-wise bias matrix formula with the $Q K^T$ and $softmax()$. The attention bias matrix is never fully materialized.
  * [FlexAttention](https://pytorch.org/blog/flexattention/#relative-position-encodings) enables fused custom attention bias computations in a FlashAttention-style kernel, using torch.compile.

    <figure>
      <p float="center">
        <img src="https://github.com/afeldman-nm/vllm-enc-dec-rfc/blob/main/img/flexattention.png" alt="FlexAttention with scoremod()" width="100%" style="margin-right:10px;" />
      </p>
      <figcaption style="text-align: center; margin-top: 10px;">
        <small>
        <strong>Figure 1:</strong> FlexAttention formula with `score_mod()` for fused custom attention bias computation (Image from <a href url="https://pytorch.org/blog/flexattention/#relative-position-encodings">FlexAttention webpage on pytorch.com</a>)
        </small>
      </figcaption>
    </figure>

It may make sense to support one or both of these methods.

Note that custom attention bias support must be added on a backend-by-backend basis, because of the kernel modifications & backend logic changes required.

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

* Remove/modify any asserts which fail if the vLLM attention backend is not XFormers
  * Currently, [the `__init__()` method of `EncoderDecoderModelRunner`](https://github.com/vllm-project/vllm/blob/b4e9528f9569d6eb8c29624771a4058fe794cb5a/vllm/worker/enc_dec_model_runner.py#L95) invokes a method `EncoderDecoderModelRunner._maybe_force_supported_attention_backend()` [here](https://github.com/vllm-project/vllm/blob/b4e9528f9569d6eb8c29624771a4058fe794cb5a/vllm/worker/enc_dec_model_runner.py#L112-L144) which (1) attempts to force encoder/decoder models to use XFormers attention backend, and (2) raises an exception if the user has overridden the attention backend to be anything other than XFormers. 

### Support CUDAGraph with encoder/decoder models

Steps to support CUDAGraph with encoder/decoder models:
* Scope out the effort require to support CUDAGraph with encoder/decoder graphs
* Write a PR for CUDAGraph + encoder/decoder
  * Remove the assertion which fails when CUDAGraph is enabled for an encoder/decoder model (see `assert_enc_dec_mr_supported_scenario()` in `vllm/worker/utils.py`)

### Support pipeline-parallelism with encoder/decoder models

Steps to support pipeline-parallelism with encoder/decoder models:
* Scope out the effort require to support pipeline-parallelism with encoder/decoder graphs
* Write a PR for pipeline-parallelism + encoder/decoder
  * Remove the assertion which fails when pipeline-parallelism is enabled for an encoder/decoder model (see `assert_enc_dec_mr_supported_scenario()` in `vllm/worker/utils.py`)

### Additional tasks

#### Low-hanging fruit: improve efficiency of the parallel cross-attention QKV computation

Cross-attention complicates the parallel GEMM computations against the $W_Q$, $W_K$, $W_V$ parameter matrices: `QKVParallelLinear.forward()` is inherited from `ColumnParallelLinear.forward(input_)`, which takes only a single `input_` argument; however, [Figure 1 of the encoder/decoder architecture overview](https://github.com/afeldman-nm/vllm-enc-dec-rfc/blob/main/infra-enc-dec.md#encoderdecoder-architecture-diagram-prefill--and-decode-phase) shows that cross-attention $W_Q$ operates on the prior self-attention output while $W_K$ and $W_V$ operate on the encoder hidden states. Furthermore, note that $W_K$ and $W_V$ are never utilized during decode because the encoder sequence is static; thus, cross-attention layers utilize all three matrices during prefill, but only $W_Q$ during decode.

For the time being, BART utilizes [the following workaround](https://github.com/vllm-project/vllm/blob/21b9c49aa37c7ba08590a99b0d4f15f86439c8f9/vllm/model_executor/models/bart.py#L359-L365):

```
def forward(
    self,
    decoder_hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: AttentionMetadata,
    encoder_hidden_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Input shape: Batch x Time x Channel"""

    # (afeldman-nm 2024/07/22) TODO:
    # Need a more efficient solution for q/k/v
    qkv_dec, _ = self.qkv_proj(decoder_hidden_states)
    q, _, _ = qkv_dec.split([self.q_size, self.kv_size, self.kv_size],
                            dim=-1)
    if encoder_hidden_states is None:
        k = None
        v = None
    else:
        qkv_enc, _ = self.qkv_proj(encoder_hidden_states)
        _, k, v = qkv_enc.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)

    attn_output = self.attn(q,
                            k,
                            v,
                            kv_cache,
                            attn_metadata,
                            attn_type=AttentionType.ENCODER_DECODER)

    output, _ = self.out_proj(attn_output)
    return output
```

The downside is that this approach performs
* 3 unnecessary GEMMs per cross-attention layer during prefill
* 2 unnecessary GEMMs per cross-attention layer during decode

The best long-term solution to this problem would be something like a `EncoderDecoderQKVParallelLinear` class with a `forward(decoder_input_,encoder_input_ = None)` method; this class would

* Compute $(decoder\; input) \times  W_Q$ and $(encoder\; input) \times [W_K W_V]$ when both arguments are not `None`
* Compute $(decoder\; input) \times  W_Q$ when `encoder_input_` is `None`

`QKVParallelLinear` exploits `ColumnParallelLinear` in order to parallelize along the head dimension; it is worth investigating whether it makes sense to preserve this strategy for `EncoderDecoderQKVParallelLinear`.

Note that self-attention layers within an encoder/decoder model are not impacted by this issue.

#### Low-priority high-effort tasks

* Speculative decoding
* Automatic prefix caching

Here it is proposed that these features are low-priority. Adding support for speculative decoder and automatic prefix caching would require a significant of effort to scope out and design the implementations.

Note that adding support for either of these features would require removing the assertions which fail when speculative decoding or automatic prefix caching are enabled for an encoder/decoder model (see `assert_enc_dec_mr_supported_scenario()` in `vllm/worker/utils.py`)

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