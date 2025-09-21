const express=require("express");
const path=require("path");
const open=require("open");
const app=express();
const PORT=3000;
app.use("/static", express.static(path.join(__dirname, "static")));
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "templates", "index.html"));
});

app.listen(PORT, "0.0.0.0", () => {
    console.log(`Servidor rodando em http://0.0.0.0:${PORT}`);
    open('https://localhost:${PORT}');
});