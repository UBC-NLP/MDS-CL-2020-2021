General Speech Knowledge Tracker

### We know:
How speech different codecs (wrapper layer: wav, mp3 etc.) 
Sampling freq. For each file
AV -> codecs (mp4 / mp3 + mp4) 

Wav2Vec2, self-supervised ASR model, speech to text.
XLSR -> multilingual pre-trained wav2vec2 

Wav2Vec2 very good performance with minimal fine-tuning on librispeech, even under other-test

Librosa/ TorchAudio/ SoundFile can  read in audio files.

### We need to know:
How do we process non-English languages?  (XLSR multilingual model fine tuned on a specific language).
What formats are supported by Wav2vec2 (wav, flac)
How to deal with audio in background? (Ideally don't have audio in the background)

How much data to fine-tune XLSR ?   (Varies depending on language! Check the huggingface model library to see variation of WER for different models trained on common voice).

Sources of data:
openslr.org 
https://commonvoice.mozilla.org/en


How to incorporate a Language Model (for fixing spelling errors on CTC outputs)?  Fairseq/Flashlight + pre-trained language model potentially the best bet. Hugging face doesn't support it.

How to load data into Hugging Face. You need to jerry rig the dataset to sort of work... here is an example, but there may be other good examples to look at in the hugging face model library:

```
 from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
 from datasets import load_dataset
 import soundfile as sf
 import torch

 # load model and tokenizer
 tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
 model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

 # define function to read in sound file
 def map_to_array(batch):
     speech, _ = sf.read(batch["file"])
     batch["speech"] = speech
     return batch

 # load dummy dataset and read soundfiles
 ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
 ds = ds.map(map_to_array)

 # tokenize
 input_values = tokenizer(ds["speech"][:2], return_tensors="pt", padding="longest").input_values  # Batch size 1

```
How long to train?  (Limit it to 1 day, WER may not be the best but this is the computing situation)

How to load a shared dataset (shared on google drive) in colab?  (In google drive right click on folder that is shared and "link in my drive" and then save the link in the directory you want. When you mount your google drive in colab you can then navigate to the directory where you saved it to then access any files in that folder)



### Q/A

is there a model we could use to convert audio straight to translated text? (without doing MT in between)  —> end-to-end SLT 

Why is CTC decoding/inclusion of language model complex: https://medium.com/corti-ai/ctc-networks-and-language-models-prefix-beam-search-explained-c11d1ee23306

is forced alignment part of speech processing? CTC can deal with this.


