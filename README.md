Start-up Hackathon at NAAMII 2024. 

---

# Post Simulation and Sentiment Analysis

## Overview

This project allows influencers, politicians, or any high-profile individuals to simulate their posts' reactions (likes, comments, shares) and sentiment before publishing them on social media. The system uses machine learning (ML) to analyze the post's content and provides insights into potential public responses.

### Features:
- **Text Input**: Users can input their posts.
- **Sentiment Analysis**: Provides a sentiment score (Positive, Negative, Neutral) for the post.
- **Metrics**: Displays simulated metrics like likes, comments, and shares.
- **Mock Comments**: Generates mock comments to preview how the post might be received.
- **Clear Button**: Clears the entered text for a fresh input.

## Architecture

### Frontend:
- **HTML/CSS/JavaScript**: The user interface is built with HTML, CSS, and JavaScript 
- **Dynamic Content**: The frontend communicates with the backend via API calls (AJAX/fetch) and dynamically updates the UI.

### Backend:
- **Node.js**: The main server is built with Node.js, handling frontend requests and routing.
- **Python Backend**: A separate Python service using Flask handles sentiment analysis and other ML tasks.
- **Communication**: Node.js communicates with the Python backend to perform sentiment analysis via API requests.

### Communication Flow:
1. **User Input**: The user enters a post on the front end.
2. **Node.js Backend**: The post is sent to the Node.js server.
3. **Python Backend**: Node.js forwards the post to a Python Flask server where sentiment analysis and metrics are calculated.
4. **Response**: The Python server sends the sentiment analysis, metrics, and other details to the Node.js backend.
5. **Frontend Update**: The Node.js server sends the data back to the front end, where it's displayed.

### Dependencies:
- **Node.js** (Express, Axios, Body-Parser)
- **Python** (Flask)
- **Machine Learning Model**: Sentiment analysis, metrics prediction (simulated or custom models)
