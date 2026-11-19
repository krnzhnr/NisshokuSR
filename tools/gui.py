#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Простая GUI для тестовых прогонов NisshokuSR.

Запускает build\\nisshoku.exe (или dist\\NisshokuSR\\nisshoku.exe)
дочерним процессом и показывает его вывод. Все настройки редактируются,
по умолчанию выставлен профиль quality-split.
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE = Path(__file__).resolve().parent.parent

CONTAINER_CHOICES = (("mp4", "MP4"), ("mkv", "MKV"), ("raw", "Raw"))

# --out-ext для raw: расширение выбирается по кодеку.
RAW_OUT_EXT = {"hevc": "h265", "h265": "h265", "h264": "h264", "av1": "av1"}

# Значения профилей из src/Cli.cpp: ApplyProfile().
PROFILES = {
    "fast": {
        "codec": "hevc", "bitdepth": "8", "preset": "p1", "tune": "hq",
        "rc": "cqp", "qp": "18", "bitrate": "25000", "gop": "250",
        "bframes": "0", "refs": "0", "lookahead": "0", "bref": "disabled",
        "split": "disable", "scale": "2",
    },
    "balanced": {
        "codec": "hevc", "bitdepth": "10", "preset": "p7", "tune": "hq",
        "rc": "cqp", "qp": "18", "bitrate": "25000", "gop": "250",
        "bframes": "0", "refs": "0", "lookahead": "0", "bref": "disabled",
        "split": "disable", "scale": "2",
    },
    "quality": {
        "codec": "hevc", "bitdepth": "10", "preset": "p7", "tune": "hq",
        "rc": "cqp", "qp": "18", "bitrate": "25000", "gop": "250",
        "bframes": "5", "refs": "8", "lookahead": "16", "bref": "middle",
        "split": "disable", "scale": "2",
    },
    "quality-split": {
        "codec": "hevc", "bitdepth": "10", "preset": "p7", "tune": "hq",
        "rc": "cqp", "qp": "18", "bitrate": "25000", "gop": "250",
        "bframes": "5", "refs": "8", "lookahead": "16", "bref": "middle",
        "split": "forced_2", "scale": "2",
    },
}


def find_exe() -> Path | None:
    env = os.environ.get("NISSHOKU_EXE")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(BASE / "build" / "nisshoku.exe")
    candidates.append(BASE / "dist" / "NisshokuSR" / "nisshoku.exe")
    for path in candidates:
        if path.is_file():
            return path
    return None


def find_engine() -> str:
    for path in (BASE / "testdata" / "animejanai_fp16.engine",
                 BASE / "dist" / "NisshokuSR" / "engines" / "animejanai_fp16.engine"):
        if path.is_file():
            return str(path)
    return ""


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.proc: subprocess.Popen | None = None
        self.queue: queue.Queue = queue.Queue()
        self.cancelled = False
        self._dry_run = False
        self._batch_summary: str | None = None

        self.inputs: list[str] = []
        self.onnx_var = tk.StringVar()
        self.engine_var = tk.StringVar(value=find_engine())
        self.output_dir_var = tk.StringVar()
        self._out_dir_auto = True
        self.container_var = tk.StringVar(value="mp4")
        self.profile_var = tk.StringVar(value="quality-split")
        self.codec_var = tk.StringVar()
        self.bitdepth_var = tk.StringVar()
        self.preset_var = tk.StringVar()
        self.tune_var = tk.StringVar()
        self.rc_var = tk.StringVar()
        self.qp_var = tk.StringVar()
        self.bitrate_var = tk.StringVar()
        self.gop_var = tk.StringVar()
        self.bframes_var = tk.StringVar()
        self.refs_var = tk.StringVar()
        self.lookahead_var = tk.StringVar()
        self.bref_var = tk.StringVar()
        self.split_var = tk.StringVar()
        self.scale_var = tk.StringVar()
        self.frames_var = tk.StringVar(value="0")
        self.audio_var = tk.StringVar(value="copy")
        self.subs_var = tk.StringVar(value="copy")
        self.gray_var = tk.BooleanVar(value=False)
        self.rebuild_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Готов")
        self.progress_var = tk.StringVar(value="—")

        self._build_ui()
        self._apply_profile()
        self._on_container()
        self._suggest_defaults()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------

    def _build_ui(self) -> None:
        self.root.title("NisshokuSR — тестовый запуск")
        self.root.geometry("1000x880")
        self.root.minsize(860, 720)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(6, weight=1)

        files = ttk.LabelFrame(self.root, text="Входы и модель", padding=8)
        files.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        files.columnconfigure(1, weight=1)

        inputs = ttk.Frame(files)
        inputs.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(0, 6))
        inputs.columnconfigure(0, weight=1)
        ttk.Label(inputs, text="Входные файлы и папки:").grid(row=0, column=0, sticky="w")
        self.input_list = tk.Listbox(inputs, height=6, selectmode="extended",
                                     activestyle="none", font=("Consolas", 9))
        self.input_list.grid(row=1, column=0, sticky="nsew")
        input_scroll = ttk.Scrollbar(inputs, orient="vertical",
                                     command=self.input_list.yview)
        input_scroll.grid(row=1, column=1, sticky="ns")
        self.input_list.configure(yscrollcommand=input_scroll.set)
        input_buttons = ttk.Frame(inputs)
        input_buttons.grid(row=1, column=2, sticky="ns", padx=(8, 0))
        for title, command in (("Добавить файлы…", self._browse_inputs),
                               ("Добавить папку…", self._browse_input_dir),
                               ("Убрать", self._remove_selected_inputs),
                               ("Очистить", self._clear_inputs)):
            ttk.Button(input_buttons, text=title, command=command).pack(fill="x",
                                                                        pady=(0, 4))

        self._file_row(files, 1, "ONNX модель", self.onnx_var, self._browse_onnx)
        self._file_row(files, 2, "TensorRT engine (необязательно)", self.engine_var,
                       self._browse_engine)
        out_entry = self._file_row(files, 3, "Папка выхода", self.output_dir_var,
                                   self._browse_output_dir)
        out_entry.bind("<KeyRelease>", lambda _e: self._mark_out_dir_manual())

        head = ttk.LabelFrame(self.root, text="Контейнер и профиль", padding=8)
        head.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        head.columnconfigure(4, weight=1)

        ttk.Label(head, text="Контейнер:").grid(row=0, column=0, sticky="w")
        for col, (value, title) in enumerate(CONTAINER_CHOICES, start=1):
            ttk.Radiobutton(head, text=title, value=value, variable=self.container_var,
                            command=self._on_container).grid(row=0, column=col, sticky="w",
                                                             padx=(0, 8))
        ttk.Label(head, text="Профиль:").grid(row=0, column=5, sticky="e", padx=(16, 4))
        self.profile_combo = ttk.Combobox(head, textvariable=self.profile_var, width=14,
                                          state="readonly", values=list(PROFILES))
        self.profile_combo.grid(row=0, column=6, sticky="w")
        self.profile_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_profile())

        ttk.Label(head, text="MP4 не хранит ASS/PGS-субтитры; Raw — видеопоток "
                             "без аудио/субтитров, расширение по кодеку.",
                  foreground="#666").grid(row=1, column=0, columnspan=7, sticky="w",
                                          pady=(6, 0))

        enc = ttk.LabelFrame(self.root, text="Кодирование", padding=8)
        enc.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        for col in (1, 3):
            enc.columnconfigure(col, weight=1)

        fields = [
            ("Кодек", self._combo(enc, self.codec_var, ("hevc", "h264", "av1"),
                                  self._on_codec)),
            ("Битность", self._combo(enc, self.bitdepth_var, ("8", "10"))),
            ("Пресет", self._combo(enc, self.preset_var,
                                   ("p1", "p2", "p3", "p4", "p5", "p6", "p7"))),
            ("Tune", self._combo(enc, self.tune_var, ("hq", "ll", "ull"))),
            ("Rate control", self._combo(enc, self.rc_var, ("cqp", "cbr"))),
            ("QP", self._entry(enc, self.qp_var)),
            ("Битрейт, kbps", self._entry(enc, self.bitrate_var)),
            ("GOP", self._entry(enc, self.gop_var)),
            ("B-кадры", self._entry(enc, self.bframes_var)),
            ("Ref / DPB (0=auto)", self._entry(enc, self.refs_var)),
            ("Lookahead", self._entry(enc, self.lookahead_var)),
            ("B-ref", self._combo(enc, self.bref_var, ("middle", "disabled", "auto"))),
            ("Split enc", self._combo(enc, self.split_var,
                                      ("disable", "auto", "auto_forced",
                                       "forced_2", "forced_3"))),
            ("Масштаб", self._entry(enc, self.scale_var)),
        ]
        for i, (label, widget) in enumerate(fields):
            row, pair = divmod(i, 2)
            col = pair * 2
            ttk.Label(enc, text=label + ":").grid(row=row, column=col, sticky="w",
                                                  padx=(0, 6), pady=2)
            widget.grid(row=row, column=col + 1, sticky="ew", padx=(0, 18), pady=2)

        misc = ttk.LabelFrame(self.root, text="Прочее", padding=8)
        misc.grid(row=3, column=0, sticky="ew", padx=10, pady=4)

        ttk.Label(misc, text="Кадров (0 = все):").grid(row=0, column=0, sticky="w")
        ttk.Entry(misc, textvariable=self.frames_var, width=10).grid(row=0, column=1,
                                                                    sticky="w", padx=(0, 16))
        ttk.Label(misc, text="Аудио:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(misc, textvariable=self.audio_var, values=("copy", "none"), width=8,
                     state="readonly").grid(row=0, column=3, sticky="w", padx=(0, 16))
        ttk.Label(misc, text="Субтитры:").grid(row=0, column=4, sticky="w")
        self.subs_combo = ttk.Combobox(misc, textvariable=self.subs_var,
                                       values=("copy", "none"), width=8, state="readonly")
        self.subs_combo.grid(row=0, column=5, sticky="w", padx=(0, 16))
        ttk.Checkbutton(misc, text="Серый кадр (--gray)", variable=self.gray_var)\
            .grid(row=0, column=6, sticky="w")
        ttk.Checkbutton(misc, text="Пересобрать engine (--rebuild-engine)",
                        variable=self.rebuild_var).grid(row=1, column=0, columnspan=4,
                                                        sticky="w", pady=(4, 0))

        bar = ttk.Frame(self.root)
        bar.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 6))
        self.run_btn = ttk.Button(bar, text="Запустить", command=lambda: self._start(False))
        self.run_btn.pack(side="left")
        self.dry_btn = ttk.Button(bar, text="Dry-run", command=lambda: self._start(True))
        self.dry_btn.pack(side="left", padx=(6, 0))
        self.cancel_btn = ttk.Button(bar, text="Остановить", command=self._cancel,
                                     state="disabled")
        self.cancel_btn.pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Открыть папку выхода",
                   command=self._open_output_dir).pack(side="left", padx=(6, 0))

        prog = ttk.Frame(self.root)
        prog.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 4))
        prog.columnconfigure(1, weight=1)
        ttk.Label(prog, text="Прогресс:").grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100.0)
        self.progress.grid(row=0, column=1, sticky="ew", padx=(6, 8))
        self.progress_text = ttk.Label(prog, textvariable=self.progress_var, anchor="w")
        self.progress_text.grid(row=0, column=2, sticky="w")

        box = ttk.Frame(self.root)
        box.grid(row=6, column=0, sticky="nsew", padx=10, pady=4)
        box.columnconfigure(0, weight=1)
        box.rowconfigure(0, weight=1)
        self.console = tk.Text(box, height=16, wrap="none", state="disabled",
                               font=("Consolas", 9), background="#101418",
                               foreground="#d6dde4", insertbackground="#d6dde4")
        self.console.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(box, orient="vertical", command=self.console.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(box, orient="horizontal", command=self.console.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.console.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.console.tag_configure("cmd", foreground="#7ec8e3")
        self.console.tag_configure("err", foreground="#ff8a80")

        status = ttk.Label(self.root, textvariable=self.status_var, relief="sunken",
                           anchor="w", padding=(6, 2))
        status.grid(row=7, column=0, sticky="ew")

        exe = find_exe()
        self._log(f"nisshoku: {exe if exe else 'НЕ НАЙДЕН (соберите build или dist)'}\n")
        self._log("Профиль по умолчанию: quality-split (HEVC 10-bit, split forced_2).\n\n")

    def _file_row(self, parent, row: int, label: str, var: tk.StringVar, command):
        ttk.Label(parent, text=label + ":").grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=(6, 6), pady=2)
        ttk.Button(parent, text="Обзор…", command=command).grid(row=row, column=2, pady=2)
        return entry

    def _combo(self, parent, var: tk.StringVar, values, callback=None):
        combo = ttk.Combobox(parent, textvariable=var, values=values, width=12,
                             state="readonly")
        if callback is not None:
            combo.bind("<<ComboboxSelected>>", lambda _e: callback())
        return combo

    def _entry(self, parent, var: tk.StringVar):
        return ttk.Entry(parent, textvariable=var, width=12)

    # ---------- defaults ----------

    def _suggest_defaults(self) -> None:
        candidate = BASE / "testdata" / "bench_src.mkv"
        if candidate.is_file():
            self._add_inputs([candidate])

    def _apply_profile(self) -> None:
        values = PROFILES.get(self.profile_var.get())
        if not values:
            return
        mapping = {
            "codec": self.codec_var, "bitdepth": self.bitdepth_var,
            "preset": self.preset_var, "tune": self.tune_var, "rc": self.rc_var,
            "qp": self.qp_var, "bitrate": self.bitrate_var, "gop": self.gop_var,
            "bframes": self.bframes_var, "refs": self.refs_var,
            "lookahead": self.lookahead_var, "bref": self.bref_var,
            "split": self.split_var, "scale": self.scale_var,
        }
        for key, var in mapping.items():
            var.set(values[key])

    # ---------- input list ----------

    def _add_inputs(self, paths: list[Path]) -> None:
        known = {os.path.normcase(item) for item in self.inputs}
        added = False
        for path in paths:
            text = str(path)
            key = os.path.normcase(text)
            if key in known:
                continue
            known.add(key)
            self.inputs.append(text)
            self.input_list.insert("end", text)
            added = True
        if added and self._out_dir_auto:
            self._suggest_output_dir()

    def _remove_selected_inputs(self) -> None:
        selection = self.input_list.curselection()
        if not selection:
            return
        for index in reversed(selection):
            self.input_list.delete(index)
            del self.inputs[index]
        if self._out_dir_auto:
            if self.inputs:
                self._suggest_output_dir()
            else:
                self.output_dir_var.set("")

    def _clear_inputs(self) -> None:
        self.input_list.delete(0, "end")
        self.inputs.clear()
        if self._out_dir_auto:
            self.output_dir_var.set("")

    def _default_output_dir(self) -> Path | None:
        """Папка выхода по умолчанию: <папка первого входа>\\upscaled."""
        if not self.inputs:
            return None
        first = self._path(self.inputs[0])
        base = first if first.is_dir() else first.parent
        return base / "upscaled"

    def _output_dir(self) -> Path | None:
        text = self.output_dir_var.get().strip()
        if text:
            return self._path(text)
        return self._default_output_dir()

    def _suggest_output_dir(self) -> None:
        folder = self._default_output_dir()
        if folder is not None:
            self.output_dir_var.set(str(folder))

    def _mark_out_dir_manual(self) -> None:
        self._out_dir_auto = False

    # ---------- browse ----------

    def _browse_inputs(self) -> None:
        picked = filedialog.askopenfilenames(
            title="Входные файлы",
            filetypes=[("Видео", "*.mkv *.mp4 *.mov *.ts *.m2ts *.webm *.avi"),
                       ("Все файлы", "*.*")])
        if not picked:
            return
        if isinstance(picked, str):
            picked = self.root.tk.splitlist(picked)
        self._add_inputs([Path(path) for path in picked])

    def _browse_input_dir(self) -> None:
        path = filedialog.askdirectory(title="Папка с видео")
        if path:
            self._add_inputs([Path(path)])

    def _browse_onnx(self) -> None:
        path = filedialog.askopenfilename(
            title="ONNX-модель",
            filetypes=[("ONNX модель", "*.onnx"), ("Все файлы", "*.*")])
        if path:
            self.onnx_var.set(path)

    def _browse_engine(self) -> None:
        path = filedialog.askopenfilename(
            title="TensorRT engine",
            filetypes=[("TensorRT engine", "*.engine *.plan"), ("Все файлы", "*.*")])
        if path:
            self.engine_var.set(path)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Папка выхода")
        if path:
            self.output_dir_var.set(path)
            self._out_dir_auto = False

    # ---------- container/extension ----------

    def _out_ext(self) -> str:
        """--out-ext: контейнер или расширение raw по кодеку."""
        container = self.container_var.get()
        if container != "raw":
            return container
        return RAW_OUT_EXT[self.codec_var.get()]

    def _on_container(self) -> None:
        if self.container_var.get() == "mp4":
            if self.subs_var.get() != "none":
                self._subs_backup = self.subs_var.get()
            self.subs_var.set("none")
            self.subs_combo.configure(state="disabled")
        else:
            self.subs_combo.configure(state="readonly")
            backup = getattr(self, "_subs_backup", None)
            if backup:
                self.subs_var.set(backup)
                self._subs_backup = None

    def _on_codec(self) -> None:
        if self.codec_var.get() == "h264" and self.bitdepth_var.get() == "10":
            self.bitdepth_var.set("8")

    # ---------- process ----------

    def _path(self, text: str) -> Path:
        path = Path(text.strip()).expanduser()
        if not path.is_absolute():
            path = BASE / path
        return path

    def _int_value(self, title: str, var: tk.StringVar, lo: int, hi: int) -> int:
        raw = var.get().strip()
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{title}: нужно целое число")
        if not lo <= value <= hi:
            raise ValueError(f"{title}: допустимо {lo}..{hi}")
        return value

    def _build_command(self, dry_run: bool) -> tuple[list[str], Path]:
        exe = find_exe()
        if exe is None:
            raise ValueError("nisshoku.exe не найден: соберите build или dist")
        if not self.inputs:
            raise ValueError("добавьте хотя бы один входной файл или папку")

        inputs: list[Path] = []
        for item in self.inputs:
            path = self._path(item)
            if not path.exists():
                raise ValueError(f"вход не найден: {path}")
            inputs.append(path)

        engine_text = self.engine_var.get().strip()
        onnx_text = self.onnx_var.get().strip()
        if not engine_text and not onnx_text:
            raise ValueError("укажите ONNX-модель или TensorRT engine (нужен хотя бы один)")
        engine_path = self._path(engine_text) if engine_text else None
        onnx_path = self._path(onnx_text) if onnx_text else None
        if engine_path is not None and not engine_path.is_file():
            raise ValueError(f"engine не найден: {engine_path}")
        if onnx_path is not None and not onnx_path.is_file():
            raise ValueError(f"ONNX-модель не найдена: {onnx_path}")

        container = self.container_var.get()
        codec = self.codec_var.get()
        if codec == "av1" and container in ("mp4", "mkv"):
            raise ValueError("AV1 в MP4/MKV не поддерживается: выберите Raw или HEVC/H.264")
        # Расширение raw всегда выводится из кодека, поэтому проверяем только кодек.
        if container == "raw" and codec not in RAW_OUT_EXT:
            raise ValueError(f"для Raw неизвестен кодек: {codec}")

        out_dir = self._output_dir()
        if out_dir is None:
            raise ValueError("не задана папка выхода")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"не удалось создать папку выхода: {exc}")
        self.output_dir_var.set(str(out_dir))

        subs = "none" if container == "mp4" else self.subs_var.get()

        cmd = [str(exe), "--profile", self.profile_var.get()]
        for path in inputs:
            cmd += ["-i", str(path)]
        cmd += ["--out-dir", str(out_dir), "--out-ext", self._out_ext()]
        if onnx_path is not None:
            cmd += ["--onnx", str(onnx_path)]
        if engine_path is not None:
            cmd += ["-e", str(engine_path)]
        cmd += [
            "--codec-out", codec,
            "--bit-depth", self.bitdepth_var.get(),
            "--preset", self.preset_var.get(),
            "--tune", self.tune_var.get(),
            "--rc", self.rc_var.get(),
            "--qp", str(self._int_value("QP", self.qp_var, 0, 51)),
            "--bitrate", str(self._int_value("Битрейт", self.bitrate_var, 1, 1000000)),
            "--gop", str(self._int_value("GOP", self.gop_var, 1, 10000)),
            "--bframes", str(self._int_value("B-кадры", self.bframes_var, 0, 5)),
            "--ref", str(self._int_value("Ref / DPB", self.refs_var, 0, 16)),
            "--lookahead", str(self._int_value("Lookahead", self.lookahead_var, 0, 31)),
            "--bref-mode", self.bref_var.get(),
            "--split-enc", self.split_var.get(),
            "--scale", str(self._int_value("Масштаб", self.scale_var, 1, 8)),
            "--frames", str(self._int_value("Кадры", self.frames_var, 0, 1000000000)),
            "--audio", self.audio_var.get(),
            "--subs", subs,
        ]
        if self.gray_var.get():
            cmd.append("--gray")
        if self.rebuild_var.get():
            if onnx_path is None:
                raise ValueError("--rebuild-engine требует ONNX-модель")
            cmd.append("--rebuild-engine")
        if dry_run:
            cmd.append("--dry-run")
        return cmd, out_dir

    def _child_env(self) -> dict:
        """PATH как в scripts/run.ps1: dev-exe ищет DLL рядом или в third_party."""
        env = os.environ.copy()
        tp = Path(env.get("NISSHOKU_THIRD_PARTY", BASE.parent / "third_party"))
        trt = Path(env.get("TRT_ROOT", tp / "TensorRT-10.8"))
        ffmpeg = Path(env.get("FFMPEG_ROOT",
                              tp / "ffmpeg-shared" /
                              "ffmpeg-n8.1-latest-win64-lgpl-shared-8.1"))
        cuda = Path(env.get("CUDA_PATH",
                            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"))
        dirs = [trt / "lib", trt / "bin", ffmpeg / "bin", cuda / "bin"]
        prefix = [str(path) for path in dirs if path.is_dir()]
        if prefix:
            env["PATH"] = os.pathsep.join(prefix + [env.get("PATH", "")])
        if (ffmpeg / "bin").is_dir():
            env["FFMPEG_ROOT"] = str(ffmpeg)
        if trt.is_dir():
            env["TRT_ROOT"] = str(trt)
        return env

    def _start(self, dry_run: bool) -> None:
        if self.proc is not None:
            return
        try:
            cmd, _ = self._build_command(dry_run)
        except ValueError as exc:
            messagebox.showerror("Ошибка", str(exc))
            return

        self._clear_console()
        self._log("> " + subprocess.list2cmdline(cmd) + "\n\n", tag="cmd")
        self.cancelled = False
        self._dry_run = dry_run
        self._batch_summary = None
        self.status_var.set("Выполняется…")
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress["value"] = 0.0
        self.progress_var.set("dry-run" if dry_run else "0.0%")
        self._set_running(True)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(BASE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=creationflags, env=self._child_env())
        except OSError as exc:
            self._set_running(False)
            self.status_var.set("Ошибка запуска")
            self.progress_var.set("ошибка")
            messagebox.showerror("Ошибка", f"не удалось запустить процесс:\n{exc}")
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.root.after(100, self._poll)

    def _reader(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    self.queue.put(line)
        finally:
            self.queue.put(None)

    def _poll(self) -> None:
        done = False
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                done = True
                break
            if not self._handle_progress(item):
                self._note_status(item)
                if "[mux] remux" in item:
                    self.progress_var.set("Финализация контейнера…")
                self._log(item)
        if done:
            proc = self.proc
            code = proc.wait() if proc is not None else -1
            self.proc = None
            self._set_running(False)
            self.progress.stop()
            if self.cancelled:
                self.status_var.set("Остановлено пользователем")
                self.progress_var.set("остановлено")
                self._log("\n[gui] процесс остановлен; временный .tmp.ts может остаться.\n",
                          tag="err")
            elif code == 0:
                summary = self._batch_summary
                self.status_var.set(f"[batch] итог: {summary}" if summary else "Готово")
                self.progress.configure(mode="determinate")
                if self._dry_run:
                    self.progress["value"] = 0.0
                    self.progress_var.set("dry-run")
                else:
                    self.progress["value"] = 100.0
                    self.progress_var.set("100%")
                self._log("\n[gui] процесс завершился успешно.\n")
            else:
                summary = self._batch_summary
                if summary:
                    self.status_var.set(f"Ошибка: {summary}")
                else:
                    self.status_var.set(f"Ошибка, код {code}")
                self.progress_var.set("ошибка")
                self._log(f"\n[gui] процесс завершился с кодом {code}.\n", tag="err")
        else:
            self.root.after(100, self._poll)

    def _note_status(self, line: str) -> None:
        """Запоминает итог пакета и стадию сборки engine для строки статуса."""
        if line.startswith("[batch] итог:"):
            self._batch_summary = line.split(":", 1)[1].strip()
        elif line.startswith("[engine] сборка"):
            self.status_var.set("Сборка TensorRT engine…")
        elif line.startswith(("[engine] собран", "[engine] кэш engine")):
            self.status_var.set("Выполняется…")

    def _handle_progress(self, line: str) -> bool:
        """Разбирает [progress] из CLI и обновляет полосу/статус."""
        if self._dry_run or not line.startswith("[progress]"):
            return False
        fields = {}
        for token in line.split()[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
        frames = fields.get("frames", "?")
        total = fields.get("total", "?")
        pct = fields.get("pct", "?")
        fps = fields.get("fps", "?")
        speed = fields.get("speed", "?")
        eta = fields.get("eta", "?")
        file_no = fields.get("file")
        files_total = fields.get("files")

        if total != "?":
            if str(self.progress.cget("mode")) != "determinate":
                self.progress.stop()
                self.progress.configure(mode="determinate")
            try:
                self.progress["value"] = max(0.0, min(100.0, float(pct)))
            except ValueError:
                pass
        else:
            if str(self.progress.cget("mode")) != "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start(12)

        parts = []
        if file_no and files_total:
            parts.append(f"файл {file_no}/{files_total}")
        parts.append(f"{frames}/{total}")
        parts.append(f"{pct}%" if pct != "?" else "?%")
        if fps != "?":
            parts.append(f"{fps} fps")
        if speed != "?":
            parts.append(speed)
        if eta != "?":
            parts.append("ETA " + self._fmt_seconds(eta))
        self.progress_var.set("  ".join(parts))
        return True

    @staticmethod
    def _fmt_seconds(value: str) -> str:
        try:
            sec = max(0, int(float(value)))
        except ValueError:
            return "?"
        return f"{sec // 60:02d}:{sec % 60:02d}"

    def _cancel(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.cancelled = True
            self.status_var.set("Остановка…")
            self.proc.terminate()

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.run_btn.configure(state=state)
        self.dry_btn.configure(state=state)
        self.cancel_btn.configure(state="normal" if running else "disabled")

    def _clear_console(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def _log(self, text: str, tag: str | None = None) -> None:
        self.console.configure(state="normal")
        self.console.insert("end", text, (tag,) if tag else ())
        self.console.see("end")
        self.console.configure(state="disabled")

    def _open_output_dir(self) -> None:
        folder = self._output_dir()
        if folder is None:
            messagebox.showinfo("Папка выхода", "Сначала добавьте входной файл или папку.")
            return
        if not folder.is_dir():
            messagebox.showinfo("Папка выхода", f"Папка ещё не создана:\n{folder}")
            return
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("Ошибка", f"не удалось открыть папку:\n{exc}")

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("Выход", "Процесс ещё выполняется. Остановить и выйти?"):
                return
            self.cancelled = True
            self.proc.terminate()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
