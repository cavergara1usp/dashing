import webview
import threading
from app import app # Importa o app Dash

def run_dash():
    app.run_server(port=8050, debug=False, use_reloader=False)

# Inicia o servidor Dash em thread separada
threading.Thread(target=run_dash, daemon=True).start()

# Cria janela com webview apontando para o app local
webview.create_window("Segurança Pública SP", "http://localhost:8050")

# Inicia o loop do webview
webview.start()
