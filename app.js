const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json());

app.post("/run-agent", async (req, res) => {
    try {
        const { query, max_results } = req.body;

        const response = await axios.post("http://127.0.0.1:5001/query_papers", {
            query: query,
            max_results: max_results || 2
        });

        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => console.log("Express server running on port 3000"));
