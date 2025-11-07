====================================================================================================

### PT-BR

# Annotation Tool

Ferramenta interativa para validar detecoes das classes listadas em `TARGET_CLASSES` usando um modelo YOLO treinado (`../best.pt`) e registrar resultados em um dataset COCO localizado em `output_dataset/` (imagens aprovadas + `annotations.coco.json`).

## Requisitos
- Python 3.9+
- Dependencias instaladas no ambiente ativo:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `pillow`
  - `tqdm` (opcional, mas recomendada se usar o pipeline de extracao de frames)

> Observacao: `tkinter` ja acompanha a instalacao oficial do Python em Windows e macOS. No Linux pode ser necessario instalar `python3-tk` via gerenciador de pacotes.

## Estrutura esperada


```
tracker/
├── best.pt
├── videos/
│   └── ... video.mp4
└── testa_tracking/
    ├── main.py
    ├── README.md
    └── output_dataset/
        ├── images/
        │   └── ... frames_validados.jpg
        └── annotations.coco.json
```
- Não precisa ser exatamente como está acima, no lugar do video.mp4 pode ser uma pasta dataset com as imagens dentro

```
tracker/
├── best.pt
├── dataset/
│   └── ... images[i].jpg
└── testa_tracking/
    ├── main.py
    ├── README.md
    └── output_dataset/
        ├── images/
        │   └── ... images[i].jpg
        └── annotations.coco.json
```


## Configuracao de classes
- Abra `main.py` e ajuste a lista `TARGET_CLASSES` com os rótulos exatos disponibilizados em `model.pt`. É possível validar quantas classes forem necessárias; cada item da lista gera automaticamente uma entrada em `categories`.
- A constante `DEFAULT_MANUAL_CLASS` define em qual classe as caixas desenhadas manualmente serão salvas. Por padrão ela assume o primeiro item de `TARGET_CLASSES`, mas pode ser alterada livremente.
- O limiar `CONF_THRESHOLD` (padrão 0.60) continua controlando quais deteccoes automáticas são exibidas/salvas.

## Como usar
1. Ajuste `SOURCE_PATH` em `main.py`:
   - Se apontar para um arquivo de video (`.mp4`, `.avi`, etc.), o app processa frame a frame.
   - Se apontar para um diretorio contendo imagens suportadas (`.jpg/.png/...`), o app percorre todos os arquivos e trata cada imagem como um item individual.
   - Ambas as formas funcionam; basta fornecer o caminho desejado (absoluto ou relativo).
2. Certifique-se de que `model.pt` esteja no diretorio pai (`../model.pt` a partir de `testa_tracking`).
3. Ative o ambiente virtual e execute:
   ```bash
   python main.py
   ```
4. Para cada frame/imagem processado(a) com deteccoes (confidencia >= `CONF_THRESHOLD` para qualquer classe de `TARGET_CLASSES`), uma janela mostrara as caixas e a confianca. Os registros validados sao gravados em `output_dataset/images/`.

## Controles
- `Validar (Enter)`: mantem as deteccoes exibidas, grava o frame em `output_dataset/images/` e atualiza `annotations.coco.json`.
- `Rejeitar (Espaco)`: ignora o frame atual e segue para o proximo.
- `Sair (Esc)`: encerra o processo; se houver anotacoes em memoria, sao persistidas antes de fechar.
- Botao `Modo anotacao (K)` ou tecla `k`: alterna o modo de anotacao manual (inicia ativado). Quando ativo, clique e arraste dentro da janela para criar novas bounding boxes; elas sao adicionadas ao JSON ao validar o frame atual.
- Botao `Remover anotacao`: ativa o modo de exclusao. Clique sobre uma bounding box (manual ou da rede) para remove-la do frame atual; ela deixara de aparecer e nao sera salva no JSON.

## Saida COCO
O arquivo `annotations.coco.json` (salvo em `output_dataset/`) e atualizado continuamente com a estrutura:
- `categories`: lista contendo todas as classes configuradas em `TARGET_CLASSES`, cada uma com um `id` sequencial.
- `images`: um item por frame validado (`file_name`, `width`, `height`, `id`).
- `annotations`: cada deteccao aprovada com `bbox` no formato `[x, y, largura, altura]`, `score`, `image_id` e `category_id`.

Os campos `image_id` e `annotation_id` sao gerados sequencialmente e começam em 1. Cada frame aprovado gera um arquivo `VIDEO_FRAME_xxxxx.jpg` em `output_dataset/images/`, alinhado com o registro correspondente no JSON.

## Dicas
- Ajuste `CONF_THRESHOLD`, `TARGET_CLASSES` e `DEFAULT_MANUAL_CLASS` no topo do `main.py` para alinhar o comportamento com o modelo carregado.
- Caso o video seja longo, considere interromper com `Esc`; o progresso ate o momento sera mantido no JSON.
- Para refinar anotacoes, utilize o modo manual (`k`) para complementar deteccoes que a rede nao encontrou.
- Se precisar descartar uma caixa incorreta antes de validar, ative o modo `Remover anotacao` e clique diretamente sobre ela.


Desenvolvido por Roberto Neto - 06/11/2025

O modelo "model.pt" foi retirado do hugging face: https://huggingface.co/mozilla-ai/swimming-pool-detector/tree/main

===================================================================================================================================

# ENG

# Annotation Tool

Interactive tool for validating detections of the classes listed in `TARGET_CLASSES` using a trained YOLO model (`../best.pt`) and recording results into a COCO dataset located in `output_dataset/` (approved images + `annotations.coco.json`).

## Requirements
- Python 3.9+
- Dependencies installed in the active environment:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `pillow`
  - `tqdm` (optional but recommended if using the frame extraction pipeline)

> **Note:** `tkinter` comes pre-installed with official Python distributions on Windows and macOS.  
> On Linux, you may need to install it manually via your package manager (`sudo apt install python3-tk`).

## Expected Structure

```
tracker/
├── best.pt
├── videos/
│   └── ... video.mp4
└── testa_tracking/
    ├── main.py
    ├── README.md
    └── output_dataset/
        ├── images/
        │   └── ... validated_frames.jpg
        └── annotations.coco.json
```

The structure doesn’t need to match exactly.  
Instead of `video.mp4`, you can use a `dataset` folder containing images:

```
tracker/
├── best.pt
├── dataset/
│   └── ... images[i].jpg
└── testa_tracking/
    ├── main.py
    ├── README.md
    └── output_dataset/
        ├── images/
        │   └── ... images[i].jpg
        └── annotations.coco.json
```

## Class Configuration
- Open `main.py` and adjust the `TARGET_CLASSES` list with the exact labels from your `model.pt`.  
  You can validate as many classes as needed; each item automatically generates a corresponding `categories` entry.
- The `DEFAULT_MANUAL_CLASS` constant defines the class used for manually drawn boxes.  
  By default, it takes the first item from `TARGET_CLASSES`, but you can change it freely.
- The `CONF_THRESHOLD` value (default: 0.60) controls which detections are displayed/saved.

## How to Use
1. Set `SOURCE_PATH` in `main.py`:
   - If it points to a video file (`.mp4`, `.avi`, etc.), the app processes it frame by frame.
   - If it points to a directory containing supported image files (`.jpg/.png/...`), it processes each file individually.
   - Both modes work; just provide the desired (absolute or relative) path.
2. Ensure that `model.pt` is located in the parent directory (`../model.pt` from `testa_tracking`).
3. Activate your virtual environment and run:
   ```bash
   python main.py
   ```
4. For each processed frame/image with detections (confidence ≥ `CONF_THRESHOLD` for any class in `TARGET_CLASSES`),  
   a window will show the bounding boxes and confidence values.  
   Validated results are saved in `output_dataset/images/`.

## Controls
| Action | Key/Button | Description |
|--------|-------------|-------------|
| **Validate** | Enter | Saves detections and updates JSON |
| **Reject** | Space | Skips the current frame |
| **Exit** | Esc | Stops and saves progress |
| **Annotation Mode** | K | Toggle manual box drawing mode |
| **Remove Annotation** | Button | Click on a box to delete it |

When **Annotation Mode** is active, click and drag inside the window to draw new bounding boxes.  
They are added to the JSON upon validation.

When **Remove Annotation** mode is active, click on any bounding box (manual or model-generated) to remove it from the current frame; it will not appear in the JSON.

## COCO Output
The `annotations.coco.json` file (in `output_dataset/`) is continuously updated with:
- **`categories`**: list of all configured classes from `TARGET_CLASSES`, each with a sequential `id`.
- **`images`**: one entry per validated frame (`file_name`, `width`, `height`, `id`).
- **`annotations`**: each approved detection with a `bbox` in `[x, y, width, height]` format, `score`, `image_id`, and `category_id`.

The `image_id` and `annotation_id` fields are sequential and start at 1.  
Each approved frame generates a `VIDEO_FRAME_xxxxx.jpg` file in `output_dataset/images/`, aligned with its corresponding JSON entry.

## Tips
- Adjust `CONF_THRESHOLD`, `TARGET_CLASSES`, and `DEFAULT_MANUAL_CLASS` at the top of `main.py` to match your model configuration.
- For long videos, you can safely stop the process with **Esc**; progress will be saved.
- Use manual annotation mode (**K**) to add missing detections not found by the model.
- If you need to remove a wrong box before validation, enable **Remove Annotation** and click on it directly.

---

**Developed by Roberto Neto – 11/06/2025**

The YOLO model (`model.pt`) was obtained from Hugging Face:  
👉 [https://huggingface.co/mozilla-ai/swimming-pool-detector/tree/main](https://huggingface.co/mozilla-ai/swimming-pool-detector/tree/main)

