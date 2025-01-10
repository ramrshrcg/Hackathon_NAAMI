import subprocess
import sys

from flask import Flask, request, jsonify
import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Load models and data
embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
model = torch.load('gnn_model_full.pth')
data = torch.load("graph_data.pt")

class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super(HeteroGNN, self).__init__()
        self.conv1 = HeteroConv({
            ('post', 'has_comment', 'comment'): SAGEConv((-1, -1), hidden_channels),
            ('comment', 'authored_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('post', 'self_loop', 'post'): SAGEConv((-1, -1), hidden_channels),  # Self-loop
            ('comment', 'self_loop', 'comment'): SAGEConv((-1, -1), hidden_channels),  # Self-loop
            ('user', 'self_loop', 'user'): SAGEConv((-1, -1), hidden_channels),  # Self-loop
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('post', 'has_comment', 'comment'): SAGEConv((-1, -1), hidden_channels),
            ('comment', 'authored_by', 'user'): SAGEConv((-1, -1), hidden_channels),
            ('post', 'self_loop', 'post'): SAGEConv((-1, -1), hidden_channels),  # Self-loop
            ('comment', 'self_loop', 'comment'): SAGEConv((-1, -1), hidden_channels),  # Self-loop
            ('user', 'self_loop', 'user'): SAGEConv((-1, -1), hidden_channels),  # Self-loop
        }, aggr='mean')

        # Decoders for link prediction and sentiment prediction
        self.link_pred = Linear(hidden_channels * 2, 1)  # Binary classification
        self.sentiment_pred = Linear(hidden_channels * 2, out_channels)  # Multi-class classification

    def forward(self, x_dict, edge_index_dict):
        # GNN Layers
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)

        return x_dict

    def decode(self, z_src, z_dst, edge_type):
        # Concatenate embeddings for link prediction
        z = torch.cat([z_src, z_dst], dim=-1)
        if edge_type == 'link_pred':
            return self.link_pred(z).sigmoid()  # Link prediction
        elif edge_type == 'sentiment':
            return self.sentiment_pred(z)  # Sentiment classification

def generate_embedding(text):
    return embedding_model.encode(text)

def predict_new_post(post_text, model, data, text_embedding_model, embedding_size):
    # Step 1: Generate text embedding for the new post
    post_embedding = torch.tensor(text_embedding_model(post_text), dtype=torch.float)

    # Step 2: Pad the embedding if needed
    if post_embedding.size(0) < embedding_size:
        padding = torch.zeros(embedding_size - post_embedding.size(0))
        post_embedding = torch.cat([post_embedding, padding], dim=0)
    elif post_embedding.size(0) > embedding_size:
        raise ValueError(f"Embedding size mismatch: Expected {embedding_size}, got {post_embedding.size(0)}")

    # Step 3: Add the new embedding to the graph
    data['post'].x = torch.cat([data['post'].x, post_embedding.unsqueeze(0)], dim=0)

    # Step 4: Forward pass through the model
    model.eval()
    with torch.no_grad():
        x_dict = model(data.x_dict, data.edge_index_dict)

        # Step 5: Extract the new post's embedding
        held_out_post_idx = data['post'].x.size(0) - 1
        z_post = x_dict['post'][held_out_post_idx]

        # Step 6: Compute link predictions
        z_users = x_dict['user']
        link_preds = model.decode(z_users, z_post.repeat(z_users.size(0), 1), edge_type='link_pred')

        # Step 7: Compute sentiment predictions
        sentiment_preds = model.decode(z_users, z_post.repeat(z_users.size(0), 1), edge_type='sentiment')

    # Step 8: Filter users with high link prediction probabilities
    predicted_users = torch.where(link_preds.squeeze() > 0.5)[0]
    predicted_sentiments = sentiment_preds[predicted_users]

    # Convert predictions to readable formats
    predicted_users = predicted_users.tolist()
    predicted_sentiments = predicted_sentiments.argmax(dim=1).tolist()  # Sentiment as class labels

    return predicted_users, predicted_sentiments

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        post_text = input_data.get('post_text')

        if not post_text:
            return jsonify({'error': 'Post text is required'}), 400

        # Handle language processing and predictions
        processed_text = post_text  # Assuming language handling is already included
        predicted_users, predicted_sentiments = predict_new_post(
            post_text=processed_text,
            model=model,
            data=data,
            text_embedding_model=generate_embedding,
            embedding_size=data['post'].x.size(1)
        )

        total_comments = len(predicted_users)

        # Map sentiments to human-readable labels
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment_labels = [sentiment_map[sentiment] for sentiment in predicted_sentiments]

        return jsonify({
            'total_comments': total_comments,
            'sentiments': sentiment_labels
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
