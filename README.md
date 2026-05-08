# Video Processor

Невеликий Python-проєкт для обробки відео через OpenCV. Програма відкриває відеофайл або камеру, застосовує фільтри, показує результат у вікні та паралельно зберігає оброблене відео у вихідний файл.

## Що вміє

- читати відео з файлу або камери;
- застосовувати `median` і `gaussian` фільтри;
- показувати результат у вікні OpenCV;
- перемикати fullscreen клавішею `F`;
- завершувати роботу клавішею `Esc`;
- зберігати результат у `output/`.

## Структура

- `main.py` — проста точка входу;
- `video_processor/` — основна логіка застосунку;
- `app_config.toml` — runtime-конфігурація;
- `pyproject.toml` — конфіг проєкту та залежності.

## Вимоги

- Python `3.11+`
- встановлений `pip`

## Встановлення залежностей

```bash
pip install -e .
```

Або без editable-режиму:

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
fps = 20.0
codec = "XVID"

[filters]
use_median = true
use_gaussian = false
median_kernel = 5
gaussian_kernel = [5, 5]

[window]
processed_window_name = "Processed Video"
normal_max_height = 700
fullscreen_toggle_key = "f"
```

`source` може бути:

- шляхом до відеофайлу;
- індексом камери, наприклад `0`.

## Запуск

Найпростіший варіант:

```bash
python main.py
```

Або як модуль:

```bash
python -m video_processor
```

Або через script entrypoint після `pip install -e .`:

```bash
video-processor
```

## Запуск з іншим конфігом

```bash
python -m video_processor --config custom_config.toml
```

## Керування

- `F` — fullscreen
- `Esc` — вихід

## Куди зберігається результат

За замовчуванням оброблене відео записується в:

```text
output/result.avi
```
