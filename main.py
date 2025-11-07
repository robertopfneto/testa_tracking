"""Ferramenta interativa para validar detecoes YOLO e gerar anotacoes COCO."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

# Setup de caminhos e parametros
SOURCE_PATH = Path(
    "dataset"
)
# Configuracoes de deteccao e classes alvo
CONF_THRESHOLD = 0.40
TARGET_CLASSES = ["swimming_pool"]
DEFAULT_MANUAL_CLASS = TARGET_CLASSES[0] if TARGET_CLASSES else None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "model.pt"
LEGACY_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "best.pt"
WEIGHTS_PATH = DEFAULT_WEIGHTS_PATH if DEFAULT_WEIGHTS_PATH.exists() else LEGACY_WEIGHTS_PATH
OUTPUT_DIR = Path(__file__).resolve().parent / "output_dataset"
OUTPUT_IMAGES_DIR = OUTPUT_DIR / "images"
ANNOTATIONS_PATH = OUTPUT_DIR / "annotations.coco.json"





@dataclass
class Detection:
    bbox_xyxy: np.ndarray
    confidence: float
    category_id: int
    label: str


class AnnotationTool:
    """Controla a interface de validacao e a geracao do arquivo COCO."""

    def __init__(self):
        if not SOURCE_PATH.exists():
            raise FileNotFoundError(f"Origem nao encontrada: {SOURCE_PATH}")
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                "Pesos nao encontrados. Coloque o arquivo em "
                f"{DEFAULT_WEIGHTS_PATH} ou mantenha o nome antigo em {LEGACY_WEIGHTS_PATH}"
            )

        OUTPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_IMAGES_DIR.mkdir(exist_ok=True)

        self.model = YOLO(str(WEIGHTS_PATH))
        self.cap: Optional[cv2.VideoCapture] = None
        self.image_paths: List[Path] = []
        self.media_mode = ""
        self.source_name = ""
        self.total_items: Optional[int] = None
        self.current_source_path: Optional[Path] = None
        self.current_output_name = ""
        self.frame_index = 0
        self.current_image_idx = -1
        self.image_id = 1
        self.annotation_id = 1

        self.current_frame = None
        self.current_detections: List[Detection] = []
        self.tk_image = None
        self.last_frame_shape: Optional[Tuple[int, int]] = None

        self.annotation_mode = True
        self.remove_mode = False
        self.manual_boxes: List[Tuple[int, int, int, int]] = []
        self.drawing_start: Optional[Tuple[int, int]] = None
        self.drawing_start_display: Optional[Tuple[int, int]] = None
        self.drawing_rect_id: Optional[int] = None
        self.canvas_image_id: Optional[int] = None
        self.display_scale = 1.0

        if not TARGET_CLASSES:
            raise ValueError("TARGET_CLASSES deve conter ao menos uma classe.")

        self.categories = [
            {"id": idx + 1, "name": class_name} for idx, class_name in enumerate(TARGET_CLASSES)
        ]
        self.class_to_category_id: Dict[str, int] = {
            item["name"]: item["id"] for item in self.categories
        }

        if DEFAULT_MANUAL_CLASS not in self.class_to_category_id:
            raise ValueError("DEFAULT_MANUAL_CLASS precisa existir em TARGET_CLASSES.")
        self.manual_category_id = self.class_to_category_id[DEFAULT_MANUAL_CLASS]

        self.images = []
        self.annotations = []

        self.window = tk.Tk()
        self.window.title("Validador de deteccoes")
        self.window.protocol("WM_DELETE_WINDOW", self.on_quit)

        controls_frame = tk.Frame(self.window)
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        self.info_var = tk.StringVar(value="Carregando...")
        self.info_label = tk.Label(controls_frame, textvariable=self.info_var, font=("Arial", 12))
        self.info_label.pack(side=tk.TOP, pady=(0, 10))

        buttons_frame = tk.Frame(controls_frame)
        buttons_frame.pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Canvas(self.window, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.accept_button = tk.Button(buttons_frame, text="Validar (Enter)", command=self.on_accept, width=18)
        self.accept_button.grid(row=0, column=0, padx=5)

        self.reject_button = tk.Button(buttons_frame, text="Rejeitar (Espaco)", command=self.on_reject, width=18)
        self.reject_button.grid(row=0, column=1, padx=5)

        self.quit_button = tk.Button(buttons_frame, text="Sair (Esc)", command=self.on_quit, width=18)
        self.quit_button.grid(row=0, column=2, padx=5)

        self.annotation_button = tk.Button(
            buttons_frame, text="Modo anotacao ON (K)", command=self.toggle_annotation_mode, width=22
        )
        self.annotation_button.grid(row=0, column=3, padx=5)

        self.remove_button = tk.Button(
            buttons_frame, text="Remover anotacao OFF", command=self.toggle_remove_mode, width=22
        )
        self.remove_button.grid(row=0, column=4, padx=5)

        navigation_frame = tk.Frame(controls_frame)
        navigation_frame.pack(side=tk.TOP, pady=(5, 0))

        self.prev_button = tk.Button(navigation_frame, text="< Anterior", command=self.on_prev_image, width=12)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(navigation_frame, text="Proximo >", command=self.on_next_image, width=12)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.window.bind("<Return>", lambda event: self.on_accept())
        self.window.bind("<space>", lambda event: self.on_reject())
        self.window.bind("<Escape>", lambda event: self.on_quit())
        self.window.bind("k", lambda event: self.toggle_annotation_mode())
        self.window.bind("K", lambda event: self.toggle_annotation_mode())
        self.window.bind("<Left>", lambda event: self.on_prev_image())
        self.window.bind("<Right>", lambda event: self.on_next_image())

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.media_mode, self.source_name = self._init_media()
        self.update_navigation_buttons()
        self.load_next_frame()

    def _init_media(self) -> Tuple[str, str]:
        """Configura origem e retorna modo ativo e nome base para saida."""
        if SOURCE_PATH.is_dir():
            image_paths = sorted(
                [p for p in SOURCE_PATH.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            )
            if not image_paths:
                raise ValueError(f"Nenhuma imagem suportada encontrada em {SOURCE_PATH}")
            self.image_paths = image_paths
            self.total_items = len(image_paths)
            return "images", SOURCE_PATH.name

        if SOURCE_PATH.is_file():
            cap = cv2.VideoCapture(str(SOURCE_PATH))
            if not cap.isOpened():
                raise RuntimeError(f"Falha ao abrir video: {SOURCE_PATH}")
            self.cap = cap
            return "video", SOURCE_PATH.stem

        raise ValueError(f"Origem desconhecida: {SOURCE_PATH}")

    def load_next_frame(self):
        """Carrega o proximo frame ou imagem e atualiza a tela."""
        if self.media_mode == "video":
            assert self.cap is not None
            ret, frame = self.cap.read()
            if not ret:
                self.finish_processing("Video finalizado.")
                return
            self.frame_index += 1
            self.current_source_path = SOURCE_PATH
            self.current_output_name = f"{self.source_name}_frame_{self.frame_index:05d}.jpg"
            self._apply_loaded_frame(frame)
            return

        if self.media_mode == "images":
            self._load_next_image_sequential()
            return

        else:
            raise RuntimeError(f"Modo de midia desconhecido: {self.media_mode}")

    def _load_next_image_sequential(self) -> bool:
        """Carrega a proxima imagem disponivel seguindo a ordem da pasta."""
        next_idx = self.current_image_idx + 1
        while next_idx < len(self.image_paths):
            candidate = self.image_paths[next_idx]
            frame = cv2.imread(str(candidate))
            if frame is None:
                print(f"[AVISO] Falha ao carregar imagem: {candidate}")
                next_idx += 1
                continue
            self._set_image_frame(next_idx, frame, candidate)
            return True

        self.finish_processing("Processamento de imagens finalizado.")
        return False

    def _set_image_frame(self, image_idx: int, frame, source_path: Path):
        """Atualiza estado interno para a imagem indicada."""
        self.current_image_idx = image_idx
        self.frame_index = image_idx + 1
        self.current_source_path = source_path
        self.current_output_name = source_path.name
        self._apply_loaded_frame(frame)

    def _apply_loaded_frame(self, frame):
        """Aplica rotinas comuns apos carregar um frame/imagem."""
        self.current_frame = frame
        self.current_detections = self.run_model(frame)
        self.manual_boxes = []
        self.annotation_mode = True
        self.remove_mode = False
        self.drawing_start = None
        self.drawing_start_display = None
        if self.drawing_rect_id is not None:
            self.canvas.delete(self.drawing_rect_id)
            self.drawing_rect_id = None
        self.update_annotation_button()
        self.update_remove_button()
        self.update_navigation_buttons()
        self.update_display()

    def load_image_by_index(self, target_idx: int) -> bool:
        """Carrega uma imagem especifica usando o indice absoluto."""
        if self.media_mode != "images":
            return False
        if target_idx < 0 or target_idx >= len(self.image_paths):
            print("[INFO] Nao ha imagem nesse indice para exibir.")
            return False
        candidate = self.image_paths[target_idx]
        frame = cv2.imread(str(candidate))
        if frame is None:
            print(f"[AVISO] Falha ao carregar imagem: {candidate}")
            return False
        self._set_image_frame(target_idx, frame, candidate)
        return True

    def run_model(self, frame) -> List[Detection]:
        """Executa o modelo YOLO e filtra detecoes da classe alvo."""
        height, width = frame.shape[:2]
        detections: List[Detection] = []
        results = self.model(frame, verbose=False)
        if not results:
            return detections

        result = results[0]
        names = result.names

        def resolve_label(cls_idx: int) -> str:
            if isinstance(names, dict):
                return names.get(cls_idx, str(cls_idx))
            if isinstance(names, (list, tuple)) and 0 <= cls_idx < len(names):
                return names[cls_idx]
            return str(cls_idx)

        for box in result.boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = resolve_label(cls_id)
            if conf < CONF_THRESHOLD or label not in self.class_to_category_id:
                continue
            category_id = self.class_to_category_id[label]
            xyxy = box.xyxy.cpu().numpy()[0]
            xyxy[0::2] = np.clip(xyxy[0::2], 0, width - 1)
            xyxy[1::2] = np.clip(xyxy[1::2], 0, height - 1)
            detections.append(
                Detection(bbox_xyxy=xyxy, confidence=conf, category_id=category_id, label=label)
            )

        return detections

    def draw_detections(self, frame, detections: List[Detection]):
        """Desenha as caixas detectadas no frame visivel."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.label} {det.confidence * 100:.1f}%"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def update_display(self):
        """Renderiza o frame com deteccoes e anotacoes manuais."""
        if self.current_frame is None:
            return

        annotated = self.draw_detections(self.current_frame.copy(), self.current_detections)
        height, width = annotated.shape[:2]
        self.last_frame_shape = (width, height)
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        max_width = max(1, int(screen_width * 0.9))
        max_height = max(1, int(screen_height * 0.9))

        scale = min(1.0, max_width / width, max_height / height)
        if scale < 1.0:
            display_width = max(1, int(width * scale))
            display_height = max(1, int(height * scale))
            display_image = pil_image.resize((display_width, display_height), resample=RESAMPLE_LANCZOS)
        else:
            display_width, display_height = width, height
            display_image = pil_image

        self.display_scale = scale
        self.tk_image = ImageTk.PhotoImage(image=display_image)

        self.canvas.delete("all")
        self.canvas.config(width=display_width, height=display_height)
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        for (x1, y1, x2, y2) in self.manual_boxes:
            dx1, dy1 = self.to_display_coords(x1, y1)
            dx2, dy2 = self.to_display_coords(x2, y2)
            self.canvas.create_rectangle(dx1, dy1, dx2, dy2, outline="yellow", width=2)

        self.update_status()

    def to_original_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Converte coordenadas do canvas para o tamanho original do frame."""
        if self.display_scale <= 0:
            return int(round(x)), int(round(y))
        inv_scale = 1.0 / self.display_scale
        return int(round(x * inv_scale)), int(round(y * inv_scale))

    def to_display_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Converte coordenadas do frame original para o canvas exibido."""
        scale = self.display_scale
        return int(round(x * scale)), int(round(y * scale))

    def build_status_message(self) -> str:
        """Gera mensagem de status para a barra de informacoes."""
        label = "Imagem" if self.media_mode == "images" else "Frame"
        total = f"/{self.total_items}" if self.total_items is not None else ""
        base = (
            f"{label} {self.frame_index}{total} | Deteccoes validas (> {CONF_THRESHOLD*100:.0f}%): "
            f"{len(self.current_detections)}"
        )
        if self.last_frame_shape:
            width, height = self.last_frame_shape
            base += f" | Resolucao: {width}x{height}"
        base += f" | Modo anotacao: {'ON' if self.annotation_mode else 'OFF'}"
        base += f" | Remover anotacao: {'ON' if self.remove_mode else 'OFF'}"
        if self.manual_boxes:
            base += f" | BBoxes manuais: {len(self.manual_boxes)}"
        return base

    def update_status(self):
        """Atualiza o texto de status exibido na interface."""
        self.info_var.set(self.build_status_message())
        self.update_annotation_button()
        self.update_remove_button()

    def update_annotation_button(self):
        """Atualiza o texto do botao de modo de anotacao."""
        if hasattr(self, "annotation_button"):
            estado = "ON" if self.annotation_mode else "OFF"
            self.annotation_button.config(text=f"Modo anotacao {estado} (K)")

    def update_remove_button(self):
        """Atualiza o texto do botao de remocao."""
        if hasattr(self, "remove_button"):
            estado = "ON" if self.remove_mode else "OFF"
            self.remove_button.config(text=f"Remover anotacao {estado}")

    def update_navigation_buttons(self):
        """Ativa/desativa botoes de navegacao conforme o contexto."""
        if not hasattr(self, "prev_button") or not hasattr(self, "next_button"):
            return
        if self.media_mode != "images" or not self.image_paths:
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            return
        if self.current_image_idx <= 0:
            self.prev_button.config(state=tk.DISABLED)
        else:
            self.prev_button.config(state=tk.NORMAL)
        if self.current_image_idx >= len(self.image_paths) - 1:
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL)

    def toggle_annotation_mode(self):
        """Alterna o modo de anotacao manual ativado pelo atalho 'k'."""
        if self.current_frame is None:
            return

        self.annotation_mode = not self.annotation_mode
        if self.annotation_mode and self.remove_mode:
            self.remove_mode = False

        if not self.annotation_mode:
            if self.drawing_rect_id is not None:
                self.canvas.delete(self.drawing_rect_id)
                self.drawing_rect_id = None
            self.drawing_start = None
            self.drawing_start_display = None
        estado_msg = "ativado" if self.annotation_mode else "desativado"
        print(f"[INFO] Modo anotacao manual {estado_msg}. Clique e arraste para desenhar caixas.")
        self.update_status()

    def toggle_remove_mode(self):
        """Alterna o modo de remocao de anotacoes."""
        if self.current_frame is None:
            return

        self.remove_mode = not self.remove_mode
        if self.remove_mode:
            if self.annotation_mode:
                self.annotation_mode = False
            if self.drawing_rect_id is not None:
                self.canvas.delete(self.drawing_rect_id)
                self.drawing_rect_id = None
            self.drawing_start = None
            self.drawing_start_display = None
        else:
            if not self.annotation_mode:
                self.annotation_mode = True
            self.drawing_start = None
            self.drawing_start_display = None
        estado_msg = "ativado" if self.remove_mode else "desativado"
        print(f"[INFO] Modo remover anotacao {estado_msg}. Clique sobre uma caixa para remove-la.")
        self.update_status()

    def on_prev_image(self):
        """Navega para a imagem anterior quando em modo de imagens."""
        if self.media_mode != "images":
            return
        if self.current_image_idx <= 0:
            print("[INFO] Ja esta na primeira imagem.")
            return
        self.load_image_by_index(self.current_image_idx - 1)

    def on_next_image(self):
        """Navega para a proxima imagem quando em modo de imagens."""
        if self.media_mode != "images":
            return
        if not self.image_paths:
            return
        target_idx = self.current_image_idx + 1 if self.current_image_idx >= 0 else 0
        if target_idx >= len(self.image_paths):
            print("[INFO] Ja esta na ultima imagem.")
            return
        self.load_image_by_index(target_idx)

    def on_mouse_down(self, event):
        """Inicia o desenho de uma caixa manual quando em modo anotacao."""
        if self.current_frame is None:
            return

        if self.remove_mode:
            self.remove_annotation_at(event.x, event.y)
            return

        if not self.annotation_mode:
            return

        self.drawing_start = self.to_original_coords(event.x, event.y)
        self.drawing_start_display = (event.x, event.y)
        self.drawing_rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="yellow",
            width=2,
            dash=(4, 2),
        )

    def on_mouse_drag(self, event):
        """Atualiza a caixa em desenho."""
        if self.remove_mode or not self.annotation_mode or self.drawing_start is None:
            return

        if self.drawing_start_display is None:
            self.drawing_start_display = (event.x, event.y)
        start_dx, start_dy = self.drawing_start_display

        if self.drawing_rect_id is None:
            self.drawing_rect_id = self.canvas.create_rectangle(
                start_dx,
                start_dy,
                event.x,
                event.y,
                outline="yellow",
                width=2,
                dash=(4, 2),
            )
        self.canvas.coords(
            self.drawing_rect_id,
            start_dx,
            start_dy,
            event.x,
            event.y,
        )

    def on_mouse_up(self, event):
        """Finaliza a caixa manual e a armazena para o frame atual."""
        if self.remove_mode or not self.annotation_mode or self.drawing_start is None:
            return

        start_x, start_y = self.drawing_start
        end_x, end_y = self.to_original_coords(event.x, event.y)

        if self.drawing_rect_id is not None:
            self.canvas.delete(self.drawing_rect_id)
            self.drawing_rect_id = None

        self.drawing_start = None
        self.drawing_start_display = None

        if self.last_frame_shape is None:
            return

        width, height = self.last_frame_shape
        x1, x2 = sorted((max(0, min(width - 1, start_x)), max(0, min(width - 1, end_x))))
        y1, y2 = sorted((max(0, min(height - 1, start_y)), max(0, min(height - 1, end_y))))

        if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
            return

        self.manual_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        self.update_display()

    def remove_annotation_at(self, x: int, y: int) -> bool:
        """Remove caixa que contenha o ponto (x, y) se existir."""
        orig_x, orig_y = self.to_original_coords(x, y)
        # Prioriza caixas manuais mais recentes
        for idx in range(len(self.manual_boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.manual_boxes[idx]
            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                del self.manual_boxes[idx]
                print("[INFO] Caixa manual removida.")
                self.update_display()
                return True

        # Em seguida avalia deteccoes do modelo
        for idx in range(len(self.current_detections) - 1, -1, -1):
            det = self.current_detections[idx]
            x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                del self.current_detections[idx]
                print("[INFO] Deteccao removida.")
                self.update_display()
                return True

        print("[INFO] Nenhuma caixa encontrada para remover.")
        return False

    def on_accept(self):
        """Persistir anotacoes quando o usuario aprova o frame."""
        if self.current_frame is None:
            return
        detections_to_save = list(self.current_detections)
        for box in self.manual_boxes:
            detections_to_save.append(
                Detection(
                    bbox_xyxy=np.array(box, dtype=np.float32),
                    confidence=1.0,
                    category_id=self.manual_category_id,
                    label=DEFAULT_MANUAL_CLASS,
                )
            )
        if detections_to_save:
            self.store_annotations(detections_to_save)
            self.write_annotations()
        self.load_next_frame()

    def on_reject(self):
        """Ignora o frame atual e avanca para o proximo."""
        self.load_next_frame()

    def on_quit(self):
        """Encerra o processo de anotacao."""
        self.finish_processing("Processo encerrado pelo usuario.")

    def store_annotations(self, detections: List[Detection]):
        """Adiciona as detecoes aprovadas na estrutura COCO."""
        height, width = self.current_frame.shape[:2]
        file_name = self.current_output_name or f"{self.source_name}_frame_{self.frame_index:05d}.jpg"
        image_path = OUTPUT_IMAGES_DIR / file_name
        if not cv2.imwrite(str(image_path), self.current_frame):
            raise RuntimeError(f"Falha ao salvar frame em {image_path}")

        image_info = {
            "id": self.image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        }
        self.images.append(image_info)

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            w = x2 - x1
            h = y2 - y1
            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": det.category_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [],
                "score": float(det.confidence),
            }
            self.annotations.append(annotation)
            self.annotation_id += 1

        self.image_id += 1

    def write_annotations(self):
        """Grava o arquivo annotations.coco.json com as anotacoes atuais."""
        data = {
            "info": {
                "description": "Validacao manual de deteccoes",
                "version": "1.0",
                "media_source": str(SOURCE_PATH),
            },
            "licenses": [],
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
        }
        with open(ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Anotacoes atualizadas em {ANNOTATIONS_PATH}")

    def finish_processing(self, message: str):
        """Libera recursos e encerra a interface."""
        if self.cap is not None:
            self.cap.release()
        self.window.unbind("<Return>")
        self.window.unbind("<space>")
        self.window.unbind("<Escape>")
        self.accept_button.config(state=tk.DISABLED)
        self.reject_button.config(state=tk.DISABLED)
        self.quit_button.config(state=tk.DISABLED)
        if self.images or self.annotations:
            self.write_annotations()
        self.info_var.set(message)
        self.window.after(1500, self.window.destroy)

    def run(self):
        """Inicia o loop principal da interface Tkinter."""
        self.window.mainloop()


def main():
    try:
        tool = AnnotationTool()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    tool.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
