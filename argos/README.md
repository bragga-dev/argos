# ARGOS — Sistema de Análise Metalográfica
**Fase 1 — Fundação**

---

## Instalação

### 1. Pré-requisitos
- Python 3.11 ou superior
- pip (gerenciador de pacotes)

### 2. Criar ambiente virtual (recomendado)
```bash
# Cria o ambiente virtual
python -m venv venv

# Ativa o ambiente (Windows)
venv\Scripts\activate

# Ativa o ambiente (Linux/macOS)
source venv/bin/activate
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. Rodar o ARGOS
```bash
python main.py
```

---

## Como usar (Fase 1)

### Fluxo básico:
1. **Abrir Imagem** → `Ctrl+O` ou toolbar
   - Formatos suportados: TIFF, PNG, JPEG
   - Preencha os metadados no diálogo que aparece

2. **Calibrar Escala** → toolbar "⚖️ Calibrar Escala"
   - Método 1: Meça a barra de escala da imagem em pixels
   - Método 2: Insira o valor µm/pixel diretamente
   - ⚠️ OBRIGATÓRIO para medições em µm reais

3. **Aplicar filtros** → painel esquerdo
   - Grayscale: converte para escala de cinza
   - Brilho/Contraste: ajuste de histograma
   - CLAHE: contraste local (muito útil em metalografia)
   - Gaussiano: suavização geral
   - Mediana: remove ruído granular (preserva bordas)
   - Bilateral: suaviza sem borrar contornos
   - Canny/Sobel: detecta bordas

4. **Reset** → desfaz TODOS os filtros, volta ao original

---

## Estrutura do Projeto

```
argos/
├── main.py                      # Ponto de entrada
├── requirements.txt             # Dependências
│
├── app/                         # Interface gráfica (PyQt6)
│   ├── main_window.py           # Janela principal
│   ├── dialogs/
│   │   ├── metadata_dialog.py   # Preenchimento de metadados
│   │   └── scale_dialog.py      # Calibração de escala
│   └── viewers/
│       └── image_viewer.py      # Widget de exibição de imagem
│
├── acquisition/                 # Camada 1 — Aquisição
│   ├── image_loader.py          # Carregamento de imagens
│   ├── scale_manager.py         # Calibração µm/pixel
│   └── metadata.py              # Dados da amostra
│
├── preprocessing/               # Camada 2 — Preparação
│   ├── image_processor.py       # Orquestrador de filtros
│   └── filters/
│       ├── gaussian_blur.py     # Filtro gaussiano
│       ├── median.py            # Filtro de mediana
│       ├── bilateral.py         # Filtro bilateral
│       └── clahe.py             # Equalização adaptativa
│
├── analysis/                    # Camada 3 — Análise (Fase 2)
├── interpretation/              # Camada 4 — Interpretação (Fase 3)
└── reports/                     # Relatórios PDF/Excel (Fase 3)
```

---

## Roadmap

- ✅ **Fase 1** — Fundação (você está aqui)
- ⬜ **Fase 2** — Threshold, Watershed, medições em µm
- ⬜ **Fase 3** — Normas ASTM E112/E45, relatórios PDF
- ⬜ **Fase 4** — U-Net para segmentação assistida por IA
