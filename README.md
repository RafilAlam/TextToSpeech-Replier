# 🧠📢 Real-Time Screen-to-TTS AI Agent  
🎙️ **Listen In, Think Fast, Talk Back**  
> A quirky yet powerful AI system that listens to your screen, interprets it using an LLM, and talks back using lifelike TTS — simulating a vocal AI companion who responds in real-time to any audible content.

---

## 🎥 Demo  


https://github.com/user-attachments/assets/b8321fb8-599f-47e0-bd2e-50d7ddf57a47


---

## 🛠️ Features
- 🎧 **Screen Audio Capture** with VB-Audio Cable  
- 🧍‍♂️ **Speaker Diarization** using `pyannote`  
- ✍️ **Speech Transcription** powered by OpenAI's `whisper`  
- 💬 **Conversational LLM (LLaMA2)** response generation via `ollama`  
- 🔊 **Natural TTS Output** using 🐸Coqui TTS ("Jenny" voice model)  
- 🔁 **Real-Time Loop**: Record → Transcribe → Converse → Speak  

---

## 📦 Tech Stack
**Language**: Python
- `whisper` (STT)
- `pyannote.audio` (Speaker Diarization)
- `ollama` (LLM backend)
- `TTS` by Coqui (Text-to-Speech)
- `pygame.mixer` (Audio playback)
- `VB-Cable` (Virtual audio device)
