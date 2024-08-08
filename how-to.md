# How to add a new encoder/decoder model

First, review the vLLM project [guidelines](https://docs.vllm.ai/en/latest/models/adding_model.html) for adding a new decoder-only model.

Each section heading below links to a corresponding section on the vLLM "Adding a new model" webpage, and the section body text discusses only the unique considerations for adding encoder/decoder models.

Note: for encoder/decoder models, we port over the `<ModelName>ForConditionalGeneration` implementation rather than the `<ModelName>ForCausalLM` implementation.

## 0. [Fork the vLLM repository](https://docs.vllm.ai/en/latest/models/adding_model.html#fork-the-vllm-repository)

## 1. [Bring your model code](https://docs.vllm.ai/en/latest/models/adding_model.html#bring-your-model-code)

## 2. [Rewrite the forward methods](https://docs.vllm.ai/en/latest/models/adding_model.html#rewrite-the-forward-methods)

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

## 3. (Optional but strongly recommended) Implement tensor parallelism and quantization support

## 4. Implement the weight loading logic

## 5. Register your model

## Feature dependencies

Most of the [encoder/decoder feature support TODOs](rfc.md) do not prevent specific models from being added to vLLM. However some models depend on other vLLM encoder/decoder workstreams:
* Multimodal encoder/decoder models with cross-attention, such as [Whisper](rfc.md#add-whisper-model), depend on [vLLM support for multimodal encoder/decoder models](rfc.md#support-encoderdecoder-multimodality).
* Models which rely on custom attention bias - as is the case for T5 - depend on [vLLM support for custom attention bias](rfc.md#support-custom-attention-bias).