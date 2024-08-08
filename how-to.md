# How to add a new encoder/decoder model

First, review the vLLM project [guidelines](https://docs.vllm.ai/en/latest/models/adding_model.html) for adding a new decoder-only model.

Each section heading below links to a corresponding section on the vLLM "Adding a new model" webpage, and the section body text discusses only the unique considerations for adding encoder/decoder models.

Note: for encoder/decoder models, we port over the `<ModelName>ForConditionalGeneration` implementation rather than the `<ModelName>ForCausalLM` implementation.

## 0. [Fork the vLLM repository](https://docs.vllm.ai/en/latest/models/adding_model.html#fork-the-vllm-repository)

Follow the instructions in the vLLM documentation.

## 1. [Bring your model code](https://docs.vllm.ai/en/latest/models/adding_model.html#bring-your-model-code)

Follow the instructions in the vLLM documentation.

## 2. [Rewrite the forward methods](https://docs.vllm.ai/en/latest/models/adding_model.html#rewrite-the-forward-methods)

Follow the instructions in the vLLM documentation.

The encoder/decoder `forward()` method signature differs slightly from decoder-only. For example, the change in input parameters between the [HF BART `forward()` method signature](https://github.com/huggingface/transformers/blob/16ed0640be71cd38208eed87bdcf39be29a83b5d/src/transformers/models/bart/modeling_bart.py#L1604-L1621), and the vLLM BART `forward()` method signature is shown below:

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

Of note, `input_ids` and `positions` are respectively the decoder input token ids and the decoder input positions, while `encoder_input_ids` and `encoder_positions` are as the name would suggestion the encoder inputs. 

## 3. [(Optional but strongly recommended) Implement tensor parallelism and quantization support](https://docs.vllm.ai/en/latest/models/adding_model.html#optional-implement-tensor-parallelism-and-quantization-support)

Follow the instructions in the vLLM documentation.

Cross-attention complicates the parallel GEMM computations against the $W_Q$, $W_K$, $W_V$ parameter matrices, [for reasons described in the encoder/decoder RFC](rfc.md#low-hanging-fruit-improve-cross-attention-parameter-matrix-parallelism); essentially the Q/K/V computation must operate on both previous-layer decoder hidden states and also encoder output hidden states. The same section of the RFC proposes a near-term workstream to address the issue. 

In the mean time, the following workaround was imployed in BART to parallelize the Q/K/V computation:

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

## [4. Implement the weight loading logic](https://docs.vllm.ai/en/latest/models/adding_model.html#implement-the-weight-loading-logic)

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

## [Example: add BART model]

As an example, the [vLLM BART model integration may be found here](https://github.com/vllm-project/vllm/blob/21b9c49aa37c7ba08590a99b0d4f15f86439c8f9/vllm/model_executor/models/bart.py).

## Final note: feature dependencies

Most of the [encoder/decoder feature support TODOs](rfc.md) do not prevent specific models from being added to vLLM. However some models depend on other vLLM encoder/decoder workstreams:
* Multimodal encoder/decoder models with cross-attention, such as [Whisper](rfc.md#add-whisper-model), depend on [vLLM support for multimodal encoder/decoder models](rfc.md#support-encoderdecoder-multimodality).
* Models which rely on custom attention bias - as is the case for T5 - depend on [vLLM support for custom attention bias](rfc.md#support-custom-attention-bias).