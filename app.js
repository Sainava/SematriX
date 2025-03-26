const express = require("express");
const axios = require("axios");
const path = require("path");

const app = express();
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views")); 
app.use(express.static("public"));
app.use(express.json());
app.use(express.urlencoded({ extended: true })); 

//Routes
app.get('/', (req, res) => {
    res.render('index'); // 
});

app.get('/about', (req, res) => {
    res.render('about_us'); // 
});

app.get('/contact', (req, res) => {
    res.render('contact_us'); // 
});


app.get("/research", (req, res) => {
    res.render("research", { papers: null, error: null });
});


// GET request - Show empty research page
app.get("/research", (req, res) => {
    res.render("research", { papers: null, error: null });
});

// POST request - Fetch data from Flask API & pass it to research.ejs
app.post("/research", async (req, res) => {
    try {
        console.log("Received request:", req.body);  // Debugging Line

        const { query, max_results } = req.body;
        if (!query) {
            console.log("No query provided.");
            return res.render("research", { papers: null, error: "Please enter a search term." });
        }

        console.log(`Fetching papers for: ${query}, Max Results: ${max_results || 5}`);

        const response = await axios.post("http://127.0.0.1:5001/query_papers", {
            query,
            max_results: max_results || 5
        });

        console.log("API Response Received:", response.data);

        res.render("research", { papers: response.data.response, error: null });

    } catch (error) {
        console.error("Error fetching papers:", error.message);
        res.render("research", { papers: null, error: "Failed to fetch data from API." });
    }
});


app.listen(3000, () => console.log("Server running on http://localhost:3000"));
