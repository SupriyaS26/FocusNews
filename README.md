# FocusNews
A real-time Indian news summarization system with gaze tracking and attention-based T5 summaries.
## Features

- RSS-based Indian news scraping
- Noise removal using StaDyNoT
- Gaze tracking with WebGazer.js + XGBoost correction
- Saliency scoring via spaCy
- Attention score = gaze + saliency fusion
- T5 summarization with attention-aware output
- Flask backend and live UI with dynamic expansion

## How to run

1. 'python server.py'
2. 'python pipeline.py'
3. Open 'http://localhost:5500/ui.html' in browser
