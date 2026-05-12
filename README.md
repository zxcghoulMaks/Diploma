# Обробка відео

Налаштовуваний Python-застосунок для обробки відео через OpenCV. Програма читає відеофайл або камеру, застосовує фільтри, виконує детекцію та трекінг об'єктів через YOLO ONNX, показує результат у вікні OpenCV і паралельно записує оброблене відео у файл.

## Можливості

- читання відео з файлу або камери;
- фільтри `median` і `gaussian`;
- детекція людей через COCO-сумісну YOLO-модель;
- трекінг об'єктів між запусками детекції;
- підписи кирилицею поверх кадру;
- підсумковий лічильник знайдених об'єктів у верхній частині кадру;
- показ результату у вікні OpenCV;
- перемикання повноекранного режиму клавішею `F`;
- збереження результату у `output/`.

## Структура проєкту

- `main.py` - мінімальна точка входу.
- `video_processor/app.py` - головний цикл обробки відео.
- `video_processor/config.py` - читання і валідація `TOML`-конфігурації.
- `video_processor/yolo_detection.py` - детекція об'єктів через YOLO ONNX.
- `video_processor/object_tracking.py` - трекінг, поєднання детекцій і відмалювання рамок.
- `video_processor/text_rendering.py` - рендеринг тексту через Pillow.
- `video_processor/video_io.py` - відкриття джерела, читання FPS і створення запису.
- `app_config.toml` - конфігурація за замовчуванням.
- `models/yolov8n.onnx` - модель YOLO, яку використовує прикладова конфігурація.

## Вимоги

- Python `3.11+`
- `pip`

## Встановлення

```bash
pip install -e .
```

Або без editable-режиму:

```bash
pip install .
```

## Конфігурація

Застосунок читає параметри з `app_config.toml`. Шляхи до `output` і `model_path` резолвляться відносно каталогу самого конфіг-файлу.

Приклад актуальної конфігурації:

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

Ключові параметри:

- `video.source` - шлях до відеофайлу або індекс камери, наприклад `0`.
- `video.output` - шлях до збереженого результату.
- `video.fps = 0.0` - використовувати FPS джерела; якщо його не вдається коректно прочитати, застосунок переходить на `25 FPS`.
- `filters.use_median` і `filters.use_gaussian` - вмикають відповідні фільтри.
- `detection.enabled_object_classes` - список класів для детекції й трекінгу; у поточному сценарії проєкт використовується насамперед для `person`, а порожній список вимикає блок детекції повністю.
- `detection.model_path` - шлях до `ONNX`-моделі.
- `detection.processing_stride` - як часто оновлювати трекінг на кадрах показу.
- `detection.detection_interval` - як часто запускати важчу YOLO-детекцію.
- `detection.labels` - читабельні назви класів для відображення.
- `window.fullscreen_toggle_key` - одна клавіша для перемикання fullscreen.

## Запуск

Найпростіший варіант:

```bash
python main.py
```

Запуск як модуля:

```bash
python -m video_processor
```

Після `pip install -e .` доступна консольна команда:

```bash
video-processor
```

Запуск з іншим конфігом:

```bash
python -m video_processor --config custom_config.toml
```

## Керування

- `F` - повноекранний режим
- `Esc` - вихід

## Модель

За замовчуванням застосунок очікує COCO-сумісну `ONNX`-модель за шляхом:

```text
models/yolov8n.onnx
```

Якщо потрібно використовувати іншу модель, змініть `detection.model_path` у конфігу.

Типова конфігурація і поточний опис проєкту орієнтовані на клас `person`. У конфігурації вже є заготовки для `dog` у `detection.labels`, але це не описується як окрема завершена можливість проєкту.
