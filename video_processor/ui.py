from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


VIDEO_FILE_TYPES = (
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
    ("All files", "*.*"),
)
OUTPUT_FILE_TYPES = (
    ("AVI files", "*.avi"),
    ("MP4 files", "*.mp4"),
    ("All files", "*.*"),
)


@dataclass(frozen=True)
class VideoSelection:
    source: Path
    output: Path


def prompt_video_selection(default_source: str | int, default_output: Path) -> VideoSelection | None:
    root = tk.Tk()
    root.title("Video Processor")
    root.resizable(False, False)
    root.geometry("720x220")

    selection: VideoSelection | None = None
    selected_source = _normalize_source(default_source)
    selected_output = _derive_output_path(selected_source, default_output)
    output_was_edited = False

    source_var = tk.StringVar(value=str(selected_source) if selected_source else "")
    output_var = tk.StringVar(value=str(selected_output))

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(0, weight=1)

    ttk.Label(frame, text="Choose the video file to process.").grid(
        row=0,
        column=0,
        columnspan=2,
        sticky="w",
        pady=(0, 12),
    )

    ttk.Label(frame, text="Input video").grid(row=1, column=0, sticky="w")
    ttk.Entry(frame, textvariable=source_var, width=78).grid(row=2, column=0, sticky="ew", padx=(0, 8))
    ttk.Button(frame, text="Browse...", command=lambda: browse_source()).grid(row=2, column=1, sticky="ew")

    ttk.Label(frame, text="Output file").grid(row=3, column=0, sticky="w", pady=(12, 0))
    ttk.Entry(frame, textvariable=output_var, width=78).grid(row=4, column=0, sticky="ew", padx=(0, 8))
    ttk.Button(frame, text="Save as...", command=lambda: browse_output()).grid(row=4, column=1, sticky="ew")

    buttons = ttk.Frame(frame)
    buttons.grid(row=5, column=0, columnspan=2, sticky="e", pady=(18, 0))
    ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right")
    ttk.Button(buttons, text="Start", command=lambda: start()).pack(side="right", padx=(0, 8))

    def browse_source() -> None:
        nonlocal selected_source, selected_output, output_was_edited

        initial_dir = selected_source.parent if selected_source else default_output.parent
        chosen_path = filedialog.askopenfilename(
            title="Choose video",
            filetypes=VIDEO_FILE_TYPES,
            initialdir=str(initial_dir),
        )
        if not chosen_path:
            return

        selected_source = Path(chosen_path).resolve()
        source_var.set(str(selected_source))
        if not output_was_edited:
            selected_output = _derive_output_path(selected_source, default_output)
            output_var.set(str(selected_output))

    def browse_output() -> None:
        nonlocal selected_output, output_was_edited

        initial_file = Path(output_var.get().strip()) if output_var.get().strip() else default_output
        chosen_path = filedialog.asksaveasfilename(
            title="Choose output file",
            defaultextension=default_output.suffix or ".avi",
            filetypes=OUTPUT_FILE_TYPES,
            initialdir=str(initial_file.parent),
            initialfile=initial_file.name,
        )
        if not chosen_path:
            return

        selected_output = Path(chosen_path).resolve()
        output_var.set(str(selected_output))
        output_was_edited = True

    def start() -> None:
        nonlocal selection

        raw_source = Path(source_var.get().strip()) if source_var.get().strip() else None
        raw_output = Path(output_var.get().strip()) if output_var.get().strip() else None

        if raw_source is None:
            messagebox.showerror("Missing input", "Choose a video file first.", parent=root)
            return
        if not raw_source.exists():
            messagebox.showerror("Missing input", "The selected video file does not exist.", parent=root)
            return
        if raw_output is None:
            messagebox.showerror("Missing output", "Choose where to save the processed video.", parent=root)
            return

        selection = VideoSelection(source=raw_source.resolve(), output=raw_output.resolve())
        root.destroy()

    root.mainloop()
    return selection


def _normalize_source(source: str | int) -> Path | None:
    if isinstance(source, int):
        return None

    stripped_source = source.strip()
    if not stripped_source:
        return None

    return Path(stripped_source).resolve()


def _derive_output_path(source: Path | None, default_output: Path) -> Path:
    if source is None:
        return default_output.resolve()

    output_suffix = default_output.suffix or ".avi"
    return (default_output.parent / f"{source.stem}_processed{output_suffix}").resolve()
