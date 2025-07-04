from flask import Flask, request, render_template, jsonify
from pedrobot import responder

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('ex01.html')

@app.route('/mensagem', methods=['POST'])
def mensagem():
    dados = request.get_json()
    texto = responder(dados['mensagem'])
    return jsonify({'resposta': texto})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)