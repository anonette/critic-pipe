# Book Critic

### Have an AI-powered critical discussion about any book or article

Book Critic is an interactive AI application that allows you to engage in a critical discussion about any book or article. It features a virtual critic (modeled after Evgeny Morozov) who provides sharp, insightful analysis and engages in meaningful dialogue about the content you provide.

Key features:
- Upload any book or article for analysis
- Engage in real-time voice conversations with the AI critic
- Receive thought-provoking critiques and analysis
- Experience natural voice interactions using advanced text-to-speech

The application is built using [Daily](https://www.daily.co/) for real-time media transport and [Cartesia](https://cartesia.ai) for text-to-speech. Everything is orchestrated together (VAD -> STT -> LLM -> TTS) using [Pipecat](https://www.pipecat.ai/).

## Setup

1. Clone the repository
2. Copy `env.example` to a `.env` file and add API keys
3. Install the required packages: `pip install -r requirements.txt`
4. Run `python3 bookCritic.py` from your command line.
5. While the app is running, go to the `https://<yourdomain>.daily.co/<room_url>` set in `DAILY_SAMPLE_ROOM_URL` and start your critical discussion!
