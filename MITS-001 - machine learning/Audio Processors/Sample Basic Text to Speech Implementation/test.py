# IMPORT Libraries
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from IPython.display import Audio


# III. Loading the Neural Network Components
# Before we can process text, we need to load the pre-trained weights—essentially the "knowledge" the model gained during its training phase.

# processor = SpeechT5Processor...
# Think of this as the translator. As we know, machines can't read letters, and so is Neural networks. They only understand numbers. The processor converts your text into a format (tokens) the model recognizes.

# model = SpeechT5ForTextToSpeech...
# This is the core Neural Network. It contains the Transformer layers that map text sequences to acoustic features.

# vocoder = SpeechT5HifiGan...:
# This is a specialized neural network called a GAN (Generative Adversarial Network). Its only job is to take the "blurry" acoustic map from the model and turn it into high-fidelity audio waves that is easier to interpret.

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# IV. Prepare the Input Text
# The first line in the next cell is the text (a string) we want to convert. You can later change this to have your own input.

# The second line in the next cell converts the string into a Tensor (a multi-dimensional array). The string literal "pt" stands for PyTorch, the mathematical framework powering the model.

input_text = "Hello! This is a simple demo of a text to speech program using Neural Network. Deep learning is quite fascinating, isn't it?"
inputs = processor(text=input_text, return_tensors="pt")

# V. Load 'Speaker Embeddings'

# We use a specific xvector from the SpeechT5 example set to avoid the script loading error
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True, revision="refs/convert/parquet")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


# VI. Generate the Speech (Inference)
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# VII. Outpute the result
sf.write("output.wav", speech.numpy(), samplerate=16000)
Audio("output.wav")