import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import axios from "axios";
import path from "path";
import { fileURLToPath } from "url";

import getRandomResponse from "./randomjson.js";

const app = express();
const PORT = 3000;

// Resolve __dirname in ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.urlencoded({ extended: true })); // Parse form data
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(express.static(path.join(__dirname, "public")));

// Route to handle form submission
app.post("/submit", async (req, res) => {
  const userPost = req.body.post;

  if (!userPost || userPost.trim() === "") {
    return res.status(400).json({ error: "Post cannot be empty!" });
  }

  try {
    // Communicate with the Python API
    const response = await axios.post("http://localhost:5000/predict", {
      post_text: userPost,
    });

    // Format the response back to the client
    const { predicted_users, sentiments } = response.data;
    res.status(200).json({
      total_comments: predicted_users.length,
      sentiment_distribution: sentiments,
    });
  } catch (error) {
    console.error("Error communicating with prediction API:", error);
    res
      .status(500)
      .json({ error: "Failed to get predictions from the model." });
  }
});
// Start the server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
