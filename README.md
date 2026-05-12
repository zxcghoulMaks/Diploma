# Обробка відео

Невеликий Python-проєкт для обробки відео через OpenCV. Програма відкриває відеофайл або камеру, застосовує фільтри, показує результат у вікні та паралельно зберігає оброблене відео у вихідний файл.

## Що вміє

- читати відео з файлу або камери;
- застосовувати `median` і `gaussian` фільтри;
- виконувати object tracking для знайдених об'єктів у відеопотоці;
- розпізнавати людей у відеопотоці та виділяти їх рамками;
- підтримує архітектуру, яку можна далі розширити для розпізнавання собак;
- показувати результат у вікні OpenCV;
- перемикати повноекранний режим клавішею `F`;
- завершувати роботу клавішею `Esc`;
- зберігати результат у `output/`.

## Структура

- `main.py` — проста точка входу;
- `video_processor/` — основна логіка застосунку;
- `app_config.toml` — конфігурація під час запуску;
- `pyproject.toml` — конфіг проєкту та залежності.

## Вимоги

- Python `3.11+`
- встановлений `pip`

## Встановлення залежностей

```bash
pip install -e .
```

Або без режиму редагування:

```bash
pip install .
```

## Налаштування

Основні параметри лежать у `app_config.toml`.

Приклад:

```toml
[video]
source = "videos/video1.mp4"
output = "output/result.avi"
fps = 0.0
codec = "MJPG"

[filters]
use_median = false
use_gaussian = false
median_kernel = 5
gaussian_kernel = [5, 5]

[detection]
enabled_object_classes = ["person"]
model_path = "models/yolov8n.onnx"
model_input_size = [640, 640]
processing_stride = 2
detection_interval = 3
confidence_threshold = 0.30
nms_threshold = 0.3
min_size = [8, 24]
track_iou_threshold = 0.2
track_max_missed_cycles = 3
box_color = [0, 255, 0]
box_thickness = 2
font_scale = 0.7
font_path = "C:/Windows/Fonts/arial.ttf"

[detection.labels]
person = "Людина"
dog = "Собака"

[window]
processed_window_name = "Оброблене відео"
normal_max_height = 700
fullscreen_toggle_key = "f"
```

`source` може бути:

- шляхом до відеофайлу;
- індексом камери, наприклад `0`.
- `fps = 0.0` означає автоматично взяти FPS з відеоджерела, щоб вихідний файл не був уповільненим.
- `codec = "MJPG"` дає якісніший результат запису, ніж більш агресивне стиснення.

Параметри секції `detection`:

- `enabled_object_classes` — список класів об'єктів, для яких треба увімкнути детекцію й трекінг;
- `model_path` — шлях до ONNX-файлу моделі YOLO;
- `model_input_size` — вхідний розмір кадру для YOLO, зазвичай `640 x 640`;
- `processing_stride` — як часто виконувати повне оновлення трекінгу; `1` означає кожен кадр, `2` означає кожен другий кадр;
- `detection_interval` — як часто запускати важку детекцію; між цими кадрами використовуються останні знайдені рамки;
- `confidence_threshold` — мінімальна впевненість YOLO, після якої об'єкт потрапляє в результат;
- `nms_threshold` — агресивність об'єднання дубльованих рамок;
- `min_size` — мінімальний розмір знайденого об'єкта; для далеких людей у кадрі значення краще тримати невеликим, наприклад `8 x 24`;
- `track_iou_threshold` — мінімальний IoU для прив'язки нової детекції до вже існуючого треку;
- `track_max_missed_cycles` — скільки циклів трек можна тримати без нової детекції;
- `box_color`, `box_thickness`, `font_scale`, `font_path` — вигляд підпису та рамок;
- `detection.labels` — читабельні підписи для кожного класу, наприклад `person` або майбутнього `dog`.

## Розширення під собак

Поточна версія вже має спільний шар object tracking та працює через одну YOLO-модель.
Щоб увімкнути собак, достатньо:

- використати COCO-сумісну YOLO-модель, яка вміє клас `dog`;
- додати `"dog"` у `enabled_object_classes`.

## Запуск

Найпростіший варіант:

```bash
python main.py
```

Або як модуль:

```bash
python -m video_processor
```

Або через команду запуску після `pip install -e .`:

```bash
video-processor
```

## Запуск з іншим конфігом

```bash
python -m video_processor --config custom_config.toml
```

## Керування

- `F` — повноекранний режим
- `Esc` — вихід

## Куди зберігається результат

За замовчуванням оброблене відео записується в:

```text
output/result.avi
```
