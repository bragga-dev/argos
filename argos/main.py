"""
ARGOS — Sistema de Visão Computacional para Análise Metalográfica
Ponto de entrada da aplicação.
"""

import sys
from PyQt6.QtWidgets import QApplication
from app.main_window import MainWindow


def main():
    # Cria a aplicação PyQt6
    app = QApplication(sys.argv)
    app.setApplicationName("ARGOS")
    app.setApplicationVersion("0.1.0")

    # Abre a janela principal
    window = MainWindow()
    window.show()

    # Mantém o loop da aplicação rodando
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
