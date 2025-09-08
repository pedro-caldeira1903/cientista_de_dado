const express=require("express");
const axios=require("axios");
const path=require("path");

const app = express();
const PORT = 5000;

app.use("/static", express.static(path.join(__dirname, "static")));
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "templates", "index.html"));
});

async function abrirNavegador() {
  try {
    const response = await axios.get("https://api.ipify.org");
    const ipPublico = response.data;
    const open = (await import("open")).default; 
    await open(`http://${ipPublico}:${PORT}`);
  } catch (err) {
    console.error("Erro ao obter IP público:", err);
  }
}

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Servidor rodando em http://0.0.0.0:${PORT}`);
  abrirNavegador();
});
