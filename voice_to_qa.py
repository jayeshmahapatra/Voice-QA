import gradio as gr
import re
import os
import torch

#Speech to text
import whisper

#QA
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

#TTS
import tempfile
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from typing import Optional


device = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper: Speech-to-text
model = whisper.load_model("base", device = device)
model_med = whisper.load_model("small", device = device)

#Roberta Q&A
model_name = "deepset/tinyroberta-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device = 0)

#TTS
tts_manager = ModelManager()
MAX_TXT_LEN = 100



print(model.device)

# Whisper - speech-to-text
def whisper_stt(audio):
  print("Inside Whisper TTS")
  # load audio and pad/trim it to fit 30 seconds
  audio = whisper.load_audio(audio)
  audio = whisper.pad_or_trim(audio)
  
  # make log-Mel spectrogram and move to the same device as the model
  mel = whisper.log_mel_spectrogram(audio).to(model.device)
  
  # detect the spoken language
  _, probs = model.detect_language(mel)
  lang = max(probs, key=probs.get)
  print(f"Detected language: {max(probs, key=probs.get)}")
  
  # decode the audio
  options_transc = whisper.DecodingOptions(fp16 = False, language=lang, task='transcribe') #lang
  options_transl = whisper.DecodingOptions(fp16 = False, language='en', task='translate') #lang
  result_transc = whisper.decode(model_med, mel, options_transc)
  result_transl = whisper.decode(model_med, mel, options_transl)
  
  # print the recognized text
  print(f"transcript is : {result_transc.text}")
  print(f"translation is : {result_transl.text}")

  return result_transc.text, result_transl.text, lang

# Coqui - Text-to-Speech
def tts(text: str, model_name: str):
    if len(text) > MAX_TXT_LEN:
        text = text[:MAX_TXT_LEN]
        print(f"Input text was cutoff since it went over the {MAX_TXT_LEN} character limit.")
    print(text, model_name)
    # download model
    model_path, config_path, model_item = tts_manager.download_model(f"tts_models/{model_name}")
    vocoder_name: Optional[str] = model_item["default_vocoder"]
    # download vocoder
    vocoder_path = None
    vocoder_config_path = None
    if vocoder_name is not None:
        vocoder_path, vocoder_config_path, _ = tts_manager.download_model(vocoder_name)
    # init synthesizer
    synthesizer = Synthesizer(
        model_path, config_path, None, None, vocoder_path, vocoder_config_path,
    )

    # synthesize
    if synthesizer is None:
        raise NameError("model not found")
    wavs = synthesizer.tts(text)

    # return output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        synthesizer.save_wav(wavs, fp)
        return fp.name

def engine(audio, context):
  # Get voice query to text
  transcribe, translation, lang = whisper_stt(audio)
  
  # Get Query answer
  answer = get_query_result(translation, context)

  answer_speech = tts(answer, model_name= 'en/ljspeech/tacotron2-DDC_ph')

  return translation, answer, answer_speech


def get_query_result(query, context):

  QA_input = {
    'question': query,
    'context': context
  }
  answer = nlp(QA_input)['answer']

  return answer


demo = gr.Blocks()

with demo:
  gr.Markdown("<h1><center>Voice to QA</center></h1>")
  gr.Markdown(
        """<center> An app to ask voice queries about a text article.</center>
        """
    )
  gr.Markdown(
        """Model pipeline consisting of - <br>- [**Whisper**](https://github.com/openai/whisper)for Speech-to-text, <br>- [**Roberta Base QA**](https://huggingface.co/deepset/roberta-base-squad2) for Question Answering, and <br>- [**CoquiTTS**](https://github.com/coqui-ai/TTS) for Text-To-Speech.
        <br> Just type/paste your text in the context field, and then ask voice questions.""")
  with gr.Column():
    with gr.Row():
      with gr.Column():
        in_audio = gr.Audio(source="microphone",  type="filepath", label='Record your voice query here in English, Spanish or French for best results-')
        in_context = gr.Textbox(label="Context")
      b1 = gr.Button("Generate Answer")

      with gr.Column():
        out_query = gr.Textbox('Your Query (Transcribed)')
        out_audio = gr.Audio(label = 'Voice response')
        out_textbox = gr.Textbox(label="Answer")

      b1.click(engine, inputs=[in_audio, in_context], outputs=[out_query, out_textbox, out_audio])
      
  #with gr.Row(): 
  #  gr.Markdown("![visitor badge](https://visitor-badge.glitch.me/badge?page_id=ysharma_Voice-to-Youtube)")

demo.launch(enable_queue=True, debug=True)