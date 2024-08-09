# How to add a new encoder/decoder model

**Note:** these guidelines were written for vLLM 0.5.4

First, review the vLLM project [guidelines](https://docs.vllm.ai/en/latest/models/adding_model.html) for adding a new decoder-only model.

Each section heading below links to a corresponding section on the vLLM "Adding a new model" webpage, and (with a few exceptions) the section body text discusses only the unique considerations for adding encoder/decoder models.

Note: for encoder/decoder models, we port over the `<ModelName>ForConditionalGeneration` implementation rather than the `<ModelName>ForCausalLM` implementation.

## 0. [Fork the vLLM repository](https://docs.vllm.ai/en/latest/models/adding_model.html#fork-the-vllm-repository)

Follow the instructions in the vLLM documentation.

## 1. [Bring your model code](https://docs.vllm.ai/en/latest/models/adding_model.html#bring-your-model-code)

Follow the instructions in the vLLM documentation.

Add a `.py` file for your model in `vllm/model_executor/models/`. The name of this file (without the `.py` extension) is the `module_name` you will use to [register your model](#5-register-your-model) later. For example, the BART model resides in `bart.py`.

## 2. [Rewrite the forward methods](https://docs.vllm.ai/en/latest/models/adding_model.html#rewrite-the-forward-methods)

Follow the instructions in the vLLM documentation.

The encoder/decoder `forward()` method signature differs slightly from decoder-only. For example, the change in input parameters between the [HF BART `forward()` method signature](https://github.com/huggingface/transformers/blob/16ed0640be71cd38208eed87bdcf39be29a83b5d/src/transformers/models/bart/modeling_bart.py#L1604-L1621), and the [vLLM BART `forward()` method signature](https://github.com/vllm-project/vllm/blob/e90457674380f931bb95c0350af4ad83af568d72/vllm/model_executor/models/bart.py#L845-L854) is shown below:

```
     def forward(
         self,
-        input_ids: torch.LongTensor = None,
-        attention_mask: Optional[torch.Tensor] = None,
-        decoder_input_ids: Optional[torch.LongTensor] = None,
-        decoder_attention_mask: Optional[torch.LongTensor] = None,
-        head_mask: Optional[torch.Tensor] = None,
-        decoder_head_mask: Optional[torch.Tensor] = None,
-        cross_attn_head_mask: Optional[torch.Tensor] = None,
-        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
-        past_key_values: Optional[List[torch.FloatTensor]] = None,
-        inputs_embeds: Optional[torch.FloatTensor] = None,
-        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
-        labels: Optional[torch.LongTensor] = None,
-        use_cache: Optional[bool] = None,
-        output_attentions: Optional[bool] = None,
-        output_hidden_states: Optional[bool] = None,
-        return_dict: Optional[bool] = None,
-    ) -> Union[Tuple, Seq2SeqLMOutput]:
+        input_ids: torch.Tensor,
+        positions: torch.Tensor,
+        encoder_input_ids: torch.Tensor,
+        encoder_positions: torch.Tensor,
+        kv_caches: List[torch.Tensor],
+        attn_metadata: AttentionMetadata,
+        intermediate_tensors: Optional[IntermediateTensors] = None,
+    ) -> torch.Tensor:
```

Of note, `input_ids` and `positions` are the decoder input token ids and positions, respectively, while `encoder_input_ids` and `encoder_positions` the corresponding encoder inputs. 

## 2.5 (Optional but strongly recommended) Implement the following encoder/decoder model architecture 

(There is not a corresponding section in the vLLM documentation.)

This section proposes a general encoder/decoder model architecture, starting with the top-level task-specific model class and proceeding hierarchically downward to the `Attention` layers.

This summary is at a high level of abstraction, so details like normalization, residuals, etc. are glossed over (and tend to be very model-specific anyway.)

### `<ModelName>ForConditionalGeneration`: top-level, task-specific model class

* Wraps `<ModelName>Model` & handles weight loading, logit processing & token sampling
* Members:
    * `model`: `<ModelName>Model` instance
    * `lm_head`: `ParallelLMHead` or subclass
    * `logits_processor`
    * `sampler`
* Methods other than `forward()`:
    * `compute_logits()`
    * `sample()`
    * `load_weights()`
* The `forward()` function signature is discussed in the [previous section](#2-rewrite-the-forward-methods).

### `<ModelName>Model`: core model class

* Encapsulates the encoder and decoder modules
* Members:
    * `encoder`: <ModelName>Encoder instance
    * `decoder`: <ModelName>Decoder instance
* The behavior of `<ModelName>Model.forward()` mirrors [Figure 1 in the encoder/decoder infrastructure guide](infra-enc-dec.md#encoderdecoder-architecture-diagram-prefill--and-decode-phase):
    * **Prefill:**
        * Invoke the encoder against the encoder input tokens/positions & obtain encoder output hidden states
        * Invoke the decoder against the decoder input tokens/positions & the encoder output hidden states to obtain decoder output hidden states
            * In the course of this step, each self-attention layer caches its KVs in its self-attention KV cache, and each cross-attention layer caches its KVs in its cross-attention KV cache.
            * Caching is handled implicitly by the underlying vLLM `Attention` layers & should not be explicitly handled by your model implementation.
        * Since cross-attention KVs are cached, discard the encoder output hidden states permanently
    * **Decode:**
        * Bypass the encoder entirely
        * Invoke the decoder against the decoder input tokens/positions
            * The underlying vLLM `Attention` layers in the decoder implicitly reuse the cached self-attention & cross-attention KVs
            * The self-attention KVs corresponding to the last decoded token will be cached
            * The cross-attention KV cache is read-only, since the encoder input sequence is static
* Example `forward()` function signature:

    ```
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                encoder_input_ids: torch.Tensor,
                encoder_positions: torch.Tensor, kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata) -> torch.Tensor
    ```

### `<ModelName>Encoder` and `<ModelName>Decoder`: encoder and decoder modules

* The encoder and decoder have generally similar structures, although specific models may differentiate them in subtle ways. However, one difference is that in addition to consuming decoder input tokens/positions, the decoder also consumes encoder output hidden states & passes them into each decoder layer.
* Members
    * `cache_config`
    * `quant_config`
    * `embed_tokens`: token embedding layer; instance of `VocabParallelEmbedding` or subclass
    * `embed_positions`: position embedding layer; instance of `VocabParallelEmbedding` or subclass
    * `layers`: {encoder,decoder} layer stack
    * Instances of any other layers such as `nn.LayerNorm`
* A general outline of `<ModelName>Encoder.forward()` and `<ModelName>Decoder.forward()` behavior:
    * Compute token & position embeddings
    * Evaluate the {encoder,decoder} layer stack against the embeddings to obtain {encoder,decoder} output hidden states
        * Only for decoder: pass encoder output hidden states to each decoder layer
    * Also account for normalization, etc.
* Example `forward()` function signature:
    * Encoder:
        ```
        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                    kv_caches: List[torch.Tensor],
                    attn_metadata: AttentionMetadata) -> torch.Tensor
        ```
    * Decoder:
        ```
        # Compared to encoder, has additional `encoder_hidden_states` input
        def forward(self, decoder_input_ids: torch.Tensor,
                    decoder_positions: torch.Tensor,
                    encoder_hidden_states: Optional[torch.Tensor],
                    kv_caches: List[torch.Tensor],
                    attn_metadata: AttentionMetadata) -> torch.Tensor:
        ```

### Individual encoder and decoder layers

* `<ModelName>EncoderLayer`: encoder layer class
    * `<ModelName>EncoderLayer` corresponds to [any one of the gray boxes representing encoder layers in Figure 1 *(left)* of the encoder/decoder infrastructure guide](infra-enc-dec.md#encoderdecoder-architecture-diagram-prefill--and-decode-phase)
    * Members:
        * `self_attn`: `<ModelName>EncoderAttention`
        * `activation_fn`: vLLM MLP activation function
        * `fc1` and `fc2`: MLP layers; `ColumnParallelLinear` and `RowParallelLinear` respectively
        * Instances of any other layers such as `nn.LayerNorm` which are applied by the encoder layer
    * Behavior of `<ModelName>EncoderLayer.forward()`:
        * Apply encoder self-attention to previous encoder-layer output hidden states
        * Apply MLP
        * Also account for residuals, normalization, etc.
    * Example `forward()` function signature:
        ```
        def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor,
                    attn_metadata: AttentionMetadata) -> torch.Tensor
        ```

* `<ModelName>DecoderLayer`: decoder layer class
    * Members:
        * `self_attn`: `<ModelName>DecoderSelfAttention`
        * `cross_attn` (or `encoder_attn` in BART): `<ModelName>CrossAttention`
        * `activation_fn`: vLLM MLP activation function
        * `fc1` and `fc2`: MLP layers; `ColumnParallelLinear` and `RowParallelLinear` respectively
        * Instances of any other layers such as `nn.LayerNorm` which are applied once by the encoder
    * The behavior of `<ModelName>DecoderLayer.forward()` mirrors [the blown-up decoder layer in Figure 1 *(right)* of the encoder/decoder infrastructure guide](infra-enc-dec.md#encoderdecoder-architecture-diagram-prefill--and-decode-phase):
        * Apply decoder self-attention to previous decoder-layer output hidden states
        * Apply encoder/decoder cross-attention to self-attention output hidden states & encoder output hidden states
        * Apply MLP
        * Also account for residuals, normalization, etc.
    * Example `forward()` function signature:
        ```
        def forward(
            self,
            decoder_hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            encoder_hidden_states: Optional[torch.Tensor] = None,
        ) -> torch.Tensor
        ```

### Wrapper classes for QKV computation + attention backend invocation

Note: for more context on the non-causal and causal attention masks described in this section, [review the default encoder/decoder attention masks](infra-enc-dec.md#default-encoderdecoder-attention-bias-or-mask).

* `<ModelName>EncoderAttention`
    * Members
        * `qkv_proj`: $[W_Q W_K W_V]$ as `QKVParallelLinear` instance
        * `attn`: `Attention` instance
        * `out_proj`: $W_O$ as `RowParalleLinear` instance
        * `q_size`: (heads per GPU) $\times$ (head dim)
        * `kv_size`: (KV heads per GPU) $\times$ (head dim)
    * `forward()` behavior:
        * Compute $[Q K V] = x [W_Q W_K W_V]$ using `qkv_proj(hidden_states)`
        * Invoke `Attention` backend against Q,K,V, passing in `attn_type=AttentionType.ENCODER`
            * `attn_type=AttentionType.ENCODER` causes `Attention` to
                * utilize `attn_metadata.encoder_seq_lens` as a reference for the sequence lengths of the encoder input
                * Construct a non-causal attention mask, where each diagonal block is a square matrix equal in side-length to the sequence length of the corresponding encoder hidden states
                * Forego KV caching entirely
        * Apply $W_O$ to attention output using `out_proj`, yielding result
    * Example `forward()` function signature:

        ```
        def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor,
                    attn_metadata: AttentionMetadata) -> torch.Tensor
        ```

* `<ModelName>DecoderSelfAttention`
    * Members:
        * `qkv_proj`: $[W_Q W_K W_V]$ as `QKVParallelLinear` instance
        * `attn`: `Attention` instance
        * `out_proj`: $W_O$ as `RowParalleLinear` instance
        * `q_size`: (heads per GPU) $\times$ (head dim)
        * `kv_size`: (KV heads per GPU) $\times$ (head dim)
    * `forward()` behavior:
        * Compute $[Q K V] = x [W_Q W_K W_V]$ using `qkv_proj(hidden_states)`
        * Invoke `Attention` backend against Q,K,V, passing in `attn_type=AttentionType.DECODER`
            * `attn_type=AttentionType.DECODER` causes `Attention` to
                * utilize `attn_metadata.seq_lens` as a reference for the sequence lengths of the decoder input
                * Construct a causal attention mask, where each diagonal block is a square matrix equal in side-length to the sequence length of the corresponding decoder hidden states
                * Cache self-attention KVs during prefill; cache new KVs & reuse old ones during decode
        * Apply $W_O$ to attention output using `out_proj`, yielding result
    * Example `forward()` function signature:

        ```
        def forward(self, hidden_states: torch.Tensor, kv_cache: torch.Tensor,
                    attn_metadata: AttentionMetadata) -> torch.Tensor
        ```

* `<ModelName>CrossAttention`
    * The QKV computation here is currently inefficient, for reasons described [later](#parallelizing-cross-attention-W_Q-W_K-W_V-parameter-matrices). Addressing this is a near-term workstream.
    * Members
        * `qkv_proj`: $[W_Q W_K W_V]$ as `QKVParallelLinear` instance
        * `attn`: `Attention` instance
        * `out_proj`: $W_O$ as `RowParalleLinear` instance
        * `q_size`: (heads per GPU) $\times$ (head dim)
        * `kv_size`: (KV heads per GPU) $\times$ (head dim)
    * `forward()` behavior:
        * Compute $[Q_{dec} K_{dec} V_{dec}] = x [W_Q W_K W_V]$ using `qkv_proj(decoder_hidden_states)`
        * Keep $Q_{dec}$, discard $K_{dec}$, $V_{dec}$
        * Compute $K_{enc}$ and $V_{enc}$
            * **Prefill:** compute $[Q_{enc} K_{enc} V_{enc}] = x [W_Q W_K W_V]$ using `qkv_proj(encoder_hidden_states)`; discard $Q_{enc}$
            * **Decode:** $K_{enc} = V_{enc} =$ `None`
        * Invoke `Attention` backend against $Q_{dec}$ , $K_{enc}$ , $V_{enc}$ , passing in `attn_type=AttentionType.ENCODER_DECODER`
            * `attn_type=AttentionType.ENCODER_DECODER` causes `Attention` to
                * utilize `attn_metadata.seq_lens` as a reference for the sequence lengths of the corresponding decoder hidden states, and `attn_metadata.encoder_seq_lens` as a reference for the sequence lengths of the corresponding encoder hidden states
                * Construct a non-causal attention mask, where each diagonal block is a rectangular matrix with dimensions (decoder seq len) $\times$ (encoder seq len)
                * Cache cross-attention KVs during prefill; reuse old KVs during decode
    * Example `forward()` function signature:

        ```
        def forward(
            self,
            decoder_hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            encoder_hidden_states: Optional[torch.Tensor] = None,
        ) -> torch.Tensor
        ```

## 3. [(Optional but strongly recommended) Implement tensor parallelism and quantization support](https://docs.vllm.ai/en/latest/models/adding_model.html#optional-implement-tensor-parallelism-and-quantization-support)

Follow the instructions in the vLLM documentation.

Recall that vLLM parallelizes QKV computation & `Attention.forward()` along the head-index dimension (i.e. per-head computations are distributed among GPUs.) Review the `__init__()` code in `BartEncoderAttention`, `BartDecoderSelfAttention`, and `BartCrossAttention` for guidance on how to use `tp_world_size` to compute the size of the attention computation (`num_heads`, `num_kv_heads`, etc.) on a single GPU.

### Parallelizing cross-attention $W_Q$ $W_K$ $W_V$ parameter matrices <a href="#user-content-parallelizing-cross-attention-W_Q-W_K-W_V-parameter-matrices" id="parallelizing-cross-attention-W_Q-W_K-W_V-parameter-matrices">#</a>

Cross-attention complicates the parallel GEMM computations against the $W_Q$, $W_K$, $W_V$ parameter matrices, because the Q/K/V computation must operate on both the previous-layer decoder hidden states and also encoder output hidden states; however, `QKVParallelLinear.forward()` is designed to operate on only a single input.

A near-term goal is to add a `CrossAttentionQKVParallelLinear` class which supports two inputs.

Until that fix becomes available, the following workaround was [employed in BART](https://github.com/vllm-project/vllm/blob/21b9c49aa37c7ba08590a99b0d4f15f86439c8f9/vllm/model_executor/models/bart.py#L359-L365) to parallelize the Q/K/V computation:

```
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
```

As this is not efficient, it is a near-term goal to find a better approach.

## [4. Implement the weight loading logic](https://docs.vllm.ai/en/latest/models/adding_model.html#implement-the-weight-loading-logic)

Follow the instructions in the vLLM documentation.

Encoder/decoder weight loader logic belongs in `<ModelName>ForConditionalGeneration` (as opposed to `<ModelName>ForCausalLM`.) 

## [5. Register your model](https://docs.vllm.ai/en/latest/models/adding_model.html#register-your-model)

This step differs from the vLLM documentation. For encoder/decoder models, register the `ForConditionalGeneration` class to the `_CONDITIONAL_GENERATION_MODELS` in [vllm/model_executor/models/\_\_init\_\_.py](https://github.com/vllm-project/vllm/blob/757ac70a64b5a643b68281c0b65f72f847cedbd6/vllm/model_executor/models/__init__.py#L86-L89)

The registry is a dictionary; the structure of a registry entry is:

```
"model_arch": ("module_name","model_cls_name")
```

where
* `module_name` is the model's Python module name in vLLM, i.e. the filename of the model in `vLLM/model_executor/models/` omitting the `.py` extension
    * For vLLM BART, this is `bart`
* `model_cls_name` is the name of the model's `ForConditionalGeneration` class within its vLLM Python module
    * For vLLM BART, this is `BartForConditionalGeneration`
* `model_arch` is the value in the `architectures` field of the model's `config.json` file on HF
    * HF BART `model_arch` examples:
        * [`facebook/bart-large`](https://huggingface.co/facebook/bart-large/blob/cb48c1365bd826bd521f650dc2e0940aee54720c/config.json#L6-L8): `model_arch` is `BartModel`
        * [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn/blob/37f520fa929c961707657b28798b30c003dd100b/config.json#L6-L8): `model_arch` is `BartForConditionalGeneration`
        * BART is an example of a model which has a different `model_arch` for different variants. If this is the case, you need to add a registry entry for each `model_arch` that you want to support, even if the `model_cls_name` and `module_name` are the same.
        * If the HF model you are porting to vLLM has multiple entries under the `architectures` field of `config.json`, then more in-depth study will be required in order to determine how to correctly register it with vLLM

For example, the BART model registration comprises the following two entries:

```
"BartModel": ("bart", "BartForConditionalGeneration"),
"BartForConditionalGeneration": ("bart", "BartForConditionalGeneration"),
```

## [6. Out-of-tree model integration](https://docs.vllm.ai/en/latest/models/adding_model.html#out-of-tree-model-integration)

Out-of-tree model integration has not been tested with encoder/decoder models.

## Example: add BART model

As an example, the [latest vLLM BART model integration may be found here](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/bart.py).

## Final note: feature dependencies

Some encoder/decoder models depend on other vLLM encoder/decoder workstreams:
* Multimodal encoder/decoder models with cross-attention, such as [Whisper](https://github.com/vllm-project/vllm/issues/7366#add-whisper-model), depend on [vLLM support for multimodal encoder/decoder models](https://github.com/vllm-project/vllm/issues/7366#support-encoderdecoder-multimodality).
* Models which rely on custom attention bias - as is the case for T5 - depend on [vLLM support for custom attention bias](https://github.com/vllm-project/vllm/issues/7366#support-custom-attention-bias).
