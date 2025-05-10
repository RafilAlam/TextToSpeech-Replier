import os
os.environ["COQUI_TOS_AGREED"] = "1"
import torch
import time


from pygame import mixer

import whisper
from pyannote.audio import Pipeline
from utils import words_per_segment

# ////////////////////////////////////////////// #
#                NLP COMPUTATION                 #

import ollama

messageHistory = [
  {
    'role': 'user',
    'content': 'You must always reply with a message, even if there is nothing to say.',
  },
  {'role': 'assistant',
   'content': "Ok."
  },
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
  {'role': 'assistant',
   'content': "Because god made it blue."
  },
  {
    'role': 'user',
    'content': 'Okay, so who are you?',
  },
  {'role': 'assistant',
   'content': "I'm Prime, just a little guy."
  },
]

# ////////////////////////////////////////////// #
#               STT - AND - DIARIZATION          #

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

pipeline.to(torch.device("cuda"))

model = whisper.load_model("medium.en")

def GenResponse(fileName):
        diarization_result = pipeline(fileName, min_speakers=1, max_speakers=3)
        transcription_result = model.transcribe(fileName, word_timestamps=True)

        final_result = words_per_segment(transcription_result, diarization_result)

        for _, segment in final_result.items():
                content = f'{segment["speaker"]}\t{segment["text"]}'
                print("Content: ", content)
                messageHistory.append({
                        'role': 'user',
                        'content': content,
                })
        response = ollama.chat(model='llama2-uncensored', messages=messageHistory)
        messageHistory.append({
               'role': 'assistant',
               'content': response['message']['content']
        })
        print('response:', response['message']['content'])
        return response['message']['content']

# ///////////////////////////////////////////////////////////////////////////////// #

from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS('tts_models/en/jenny/jenny').to(device)
def GenTTS(input_text):
        tts.tts_to_file(text=input_text if input_text.strip() else 'I did not quite get that, could you repeat?', file_path="output.wav")
        return 'output.wav'

while True:
        mixer.init(devicename='CABLE Input (VB-Audio Virtual Cable)')
        input("Enter to Begin Recording")

        from subprocess import Popen
        #time.sleep(8)
        mixer.music.load(GenTTS(GenResponse('Input.wav')))
        mixer.music.play()
        Popen('python capture.py')
        input("Enter to End")
        mixer.music.stop()
        mixer.quit()