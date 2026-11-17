# License

The tiny-voca-es dataset (audio, metadata and annotations) is released under the
Creative Commons Attribution 4.0 International License (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

## Attribution

Please credit "tiny-voca-es dataset" and link to the dataset page,
https://huggingface.co/datasets/marcosmaceo/tiny-voca-es. The method follows
the tiny-voca dataset by Muhammed Nazeem, https://huggingface.co/datasets/n4ze3m/tiny-voca.

## Third-party material

- Text with `source_license` `cc0-1.0` comes from the Mozilla Common Voice
  Spanish sentence files (CC0), https://github.com/common-voice/common-voice:
  `sentence-collector.txt` (origin `common_voice_sentence_collector`) and
  `wiki.es.txt` (origin `common_voice_wiki`). The `source_id` and `source_url`
  columns identify each sentence.
- Text with `source_license` `project-own` was written for this project:
  templates (origin `tiny_voca_templates`) and questions and exclamations
  (origins `mistral_questions`, `mistral_exclamations`) generated with the
  `mistral-large-latest` model through the Mistral API and filtered by the
  project's scripts.
- The audio was generated with Qwen3-TTS 1.7B Base and the two voices with
  Qwen3-TTS 1.7B VoiceDesign, both released under Apache-2.0. The voices are not
  recordings or imitations of real people.
- Whisper (MIT) screened the clips, and `facebook/wav2vec2-lv-60-espeak-cv-ft`
  (Apache-2.0) aligned them; neither is part of the data.
