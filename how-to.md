# How-to guides for encoder/decoder models & feature compatibility

## Guide to adding new encoder/decoder models to vLLM

## Guide to adding encoder/decoder support to existing vLLM backends

* Identify impacted AttentionMetadata fields
* Use of encoder sequence length
* Correct usage of context
* Prefill attention masks
* ModelRunner kernel enforcement

Sources/notes:

[^1]: [Whisper](https://cdn.openai.com/papers/whisper.pdf) is a multi-modal encoder/decoder speech-recognition model.