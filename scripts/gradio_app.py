"""
CLI entry-point for the OCR Benchmark Dashboard.
Reads results produced by BenchmarkRunner using the centralized configuration.

Usage:
    uv run scripts/gradio_app.py
    uv run scripts/gradio_app.py --host 0.0.0.0 --port 8080 --share
    uv run scripts/gradio_app.py --config custom_config.yaml
"""

import argparse
import glob
import json
import math
import os

import gradio as gr
import pandas as pd

from ocr_core.config import BenchmarkConfig, load_config

# ── Helpers ─────────────────────────────────────────────────


def _safe_ls(path):
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


def _fmt(v, spec=".4f", suffix=""):
    if v is None or (isinstance(v, (int, float)) and math.isnan(v)):
        return "N/A"
    try:
        return f"{v:{spec}}{suffix}"
    except (ValueError, TypeError):
        return str(v)


def _dd(choices, value=None):
    return gr.update(choices=choices, value=value)


def _slider(minimum=1, maximum=1, value=1):
    return gr.update(minimum=minimum, maximum=maximum, value=value, step=1)


# ── Dashboard Class ─────────────────────────────────────────


class Dashboard:
    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        # Resolve paths relative to the project root (one level up from /scripts)
        self._root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # os.path.join safely handles both relative and absolute paths
        self.input_dir = os.path.join(self._root, cfg.data.input_dir)
        self.processed_dir = os.path.join(self._root, cfg.data.processed_dir)
        self.results_dir = os.path.join(self._root, cfg.data.results_dir)

    def _list_test_sets(self):
        return [
            d
            for d in _safe_ls(self.input_dir)
            if os.path.isdir(os.path.join(self.input_dir, d))
        ]

    def _list_files(self, ts):
        d = os.path.join(self.input_dir, ts) if ts else ""
        return (
            [f for f in _safe_ls(d) if os.path.isfile(os.path.join(d, f))] if d else []
        )

    def _parse_results(self, ts):
        d = os.path.join(self.results_dir, ts)
        suffix = "_results.json"
        for f in _safe_ls(d):
            if not f.endswith(suffix):
                continue
            path = os.path.join(d, f)
            model, device = None, None
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                model = data.get("model_name")
                device = data.get("device")
            except (json.JSONDecodeError, OSError):
                pass
            if not model or not device:
                stem = f[: -len(suffix)]
                idx = stem.rfind("_")
                if idx > 0:
                    model, device = stem[:idx], stem[idx + 1 :]
            if model and device:
                yield model, device

    def _list_models(self, ts):
        return sorted({m for m, _ in self._parse_results(ts)})

    def _list_devices(self, ts, model):
        return sorted({d for m, d in self._parse_results(ts) if m == model})

    def _load_result(self, ts, model, device):
        p = os.path.join(self.results_dir, ts, f"{model}_{device}_results.json")
        if not os.path.isfile(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _page_image(self, ts, fname, idx=0):
        stem = os.path.splitext(fname)[0]
        d = os.path.join(self.processed_dir, ts, stem)
        pngs = sorted(glob.glob(os.path.join(d, "*.png"))) if os.path.isdir(d) else []
        return pngs[idx] if idx < len(pngs) else None

    def _page_count(self, ts, fname):
        stem = os.path.splitext(fname)[0]
        d = os.path.join(self.processed_dir, ts, stem)
        return len(glob.glob(os.path.join(d, "*.png"))) if os.path.isdir(d) else 0

    def load_summary(self):
        csv = os.path.join(self.results_dir, "summary.csv")
        if not os.path.isfile(csv):
            return pd.DataFrame()
        df = pd.read_csv(csv)
        return df.drop_duplicates(subset=["Model", "Test Set", "Device"], keep="last")

    def _cascade(self, ts=None, model=None):
        tss = self._list_test_sets()
        ts = ts or (tss[0] if tss else None)
        files = self._list_files(ts) if ts else []
        models = self._list_models(ts) if ts else []
        mdl = model or (models[0] if models else None)
        devices = self._list_devices(ts, mdl) if ts and mdl else []
        first = files[0] if files else None
        n = max(1, self._page_count(ts, first)) if ts and first else 1
        return ts, files, models, mdl, devices, first, n

    def init_explorer(self):
        ts, files, models, mdl, devices, first, n = self._cascade()
        return (
            _dd(self._list_test_sets(), ts),
            _dd(files, first),
            _dd(models, mdl),
            _dd(devices, devices[0] if devices else None),
            _slider(1, n, 1),
        )

    def on_ts(self, ts):
        _, files, models, mdl, devices, first, n = self._cascade(ts)
        return (
            _dd(files, first),
            _dd(models, mdl),
            _dd(devices, devices[0] if devices else None),
            _slider(1, n, 1),
        )

    def on_model(self, ts, m):
        if not ts or not m:
            return _dd([], None)
        devs = self._list_devices(ts, m)
        return _dd(devs, devs[0] if devs else None)

    def on_file(self, ts, f):
        if not ts or not f:
            return _slider()
        n = self._page_count(ts, f)
        return _slider(1, max(1, n), 1)

    def load_explorer(self, ts, fname, model, device, page):
        empty = (None, "", "", pd.DataFrame(), "", "")
        if not all([ts, fname, model, device]):
            return *empty[:-1], "⚠️ Select all fields"

        data = self._load_result(ts, model, device)
        if not data:
            return *empty[:-1], f"⚠️ No results for {model}/{device}"

        file_metrics = [m for m in data.get("metrics", []) if m["file"] == fname]
        if not file_metrics:
            return *empty[:-1], f"⚠️ No data for {fname}"

        page_num = max(1, int(page))
        img = self._page_image(ts, fname, page_num - 1)
        pm = next((m for m in file_metrics if m.get("page") == page_num), None)
        if pm is None:
            return *empty[:-1], f"⚠️ No data for page {page_num} of {fname}"

        score_keys = set()
        for m in file_metrics:
            score_keys.update(m.get("scores", {}).keys())
        score_keys = sorted(score_keys)

        rows = []
        for m in file_metrics:
            row = {
                "Page": m.get("page", "?"),
                "Time (s)": _fmt(m.get("prediction_time_seconds"), ".3f"),
            }
            for sk in score_keys:
                row[sk.upper()] = _fmt(m.get("scores", {}).get(sk))
            rows.append(row)

        regions_md = ""
        pred_regions = pm.get("predicted_regions", [])
        if pred_regions:
            regions_md = (
                "\n### 🗺️ Detected Regions\n\n"
                "| # | Category | Text (preview) | BBox |\n"
                "|---|----------|----------------|------|\n"
            )
            for i, r in enumerate(pred_regions):
                cat = r.get("category", "?")
                txt_raw = r.get("text", "").replace("\n", " ↵ ").replace("|", "&#124;")
                txt_raw = txt_raw.replace("<", "&lt;").replace(">", "&gt;")
                txt = (txt_raw[:60] + "...") if len(txt_raw) > 60 else txt_raw
                bbox = r.get("bbox", {})
                if isinstance(bbox, dict):
                    x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
                    x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
                    bbox_str = f"({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})"
                else:
                    bbox_str = str(bbox)
                regions_md += f"| {i+1} | {cat} | {txt} | {bbox_str} |\n"

        summaries = data.get("metric_summaries", {})
        timing = data.get("timing_summary", {})

        md = f"### 📋 Benchmark: {data.get('model_name', 'N/A')}\n\n"
        md += "| Metric | Mean | 95% CI |\n|--------|------|--------|\n"
        for name, s in summaries.items():
            mean = s["mean"]
            ci_lower = s.get("ci_lower", 0)
            ci_upper = s.get("ci_upper", 0)
            md += f"| **{name.upper()}** | {mean:.4f} | "
            md += f"[{ci_lower:.4f}, {ci_upper:.4f}] |\n"
        if timing:
            speed = timing.get("mean_s_per_page", 0)
            ci_lower = timing.get("ci_lower", 0)
            ci_upper = timing.get("ci_upper", 0)
            md += f"| **Speed (s/page)** | {speed:.4f} | "
            md += f"[{ci_lower:.4f}, {ci_upper:.4f}] |\n"

        measured = data.get("measured_runs", "?")
        warmup = data.get("warmup_runs", 0)
        md += f"\n- Runs: {measured} measured + {warmup} warmup\n"
        md += f"- Pages: {data.get('total_pages_processed', '?')}\n"
        res = data.get("resources", {})
        md += f"- Peak RAM: {_fmt(res.get('peak_ram_mb'), '.0f', ' MB')}\n"
        md += f"- Peak VRAM: {_fmt(res.get('peak_vram_mb'), '.0f', ' MB')}\n"

        deg = data.get("degradation_results", [])
        if deg:
            md += "\n### 🔧 Degradation Robustness\n\n"
            keys = sorted({k for d in deg for k in d.get("scores", {})})
            md += "| Degradation |" + "".join(f" {k.upper()} |" for k in keys) + "\n"
            md += "|-------------|" + "".join("------|" for _ in keys) + "\n"
            for d in deg:
                row = f"| {d['label']} |"
                for k in keys:
                    row += f" {_fmt(d['scores'].get(k))} |"
                md += row + "\n"

        md += regions_md

        return (
            img,
            pm.get("predicted_text", ""),
            pm.get("ground_truth_text") or "",
            pd.DataFrame(rows),
            md,
            f"✅ Page {page_num}",
        )

    def init_compare(self):
        ts, files, models, mdl, devices, first, _ = self._cascade()
        return (
            _dd(self._list_test_sets(), ts),
            _dd(files, first),
            _dd(models, mdl),
            _dd(models, mdl),
            _dd(devices, devices[0] if devices else None),
            _dd(devices, devices[0] if devices else None),
        )

    def on_cmp_ts(self, ts):
        _, files, models, mdl, devices, first, _ = self._cascade(ts)
        return (
            _dd(files, first),
            _dd(models, mdl),
            _dd(models, mdl),
            _dd(devices, devices[0] if devices else None),
            _dd(devices, devices[0] if devices else None),
        )

    def on_cmp_model(self, ts, m):
        if not ts or not m:
            return _dd([], None)
        devs = self._list_devices(ts, m)
        return _dd(devs, devs[0] if devs else None)

    def load_compare(self, ts, fname, ma, da, mb, db):
        def _one(model, dev):
            if not all([ts, fname, model, dev]):
                return "", ""
            data = self._load_result(ts, model, dev)
            if not data:
                return "", "*No data*"
            file_entries = [
                entry for entry in data.get("metrics", []) if entry["file"] == fname
            ]
            pred = "\n\n".join(
                f"── Page {entry.get('page', '?')} ──\n{entry['predicted_text']}"
                for entry in file_entries
                if entry.get("predicted_text")
            )

            summaries = data.get("metric_summaries", {})
            info = " | ".join(
                f"**{n.upper()}:** {s['mean']:.4f}" for n, s in summaries.items()
            )
            res = data.get("resources", {})
            info += f" | **RAM:** {_fmt(res.get('peak_ram_mb'), '.0f', ' MB')}"
            info += f" | **VRAM:** {_fmt(res.get('peak_vram_mb'), '.0f', ' MB')}"
            return pred, info

        pa, ia = _one(ma, da)
        pb, ib = _one(mb, db)
        return pa, ia, pb, ib

    def build_ui(self) -> gr.Blocks:
        with gr.Blocks(title="OCR Benchmark Dashboard") as demo:
            gr.Markdown(
                "# 🔍 OCR Benchmarking Dashboard\n"
                "Evaluate and compare OCR models with comprehensive metrics."
            )

            with gr.Tab("📊 Summary"):
                gr.Markdown("### All benchmark runs")
                refresh = gr.Button("🔄 Refresh")
                summary_df = gr.Dataframe(
                    value=pd.DataFrame(), interactive=False, wrap=True
                )
                refresh.click(fn=self.load_summary, outputs=summary_df)

            with gr.Tab("🔎 Explorer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        e_ts = gr.Dropdown(
                            label="Test Set", choices=[], interactive=True
                        )
                        e_file = gr.Dropdown(
                            label="Input File", choices=[], interactive=True
                        )
                        e_model = gr.Dropdown(
                            label="Model", choices=[], interactive=True
                        )
                        e_dev = gr.Dropdown(
                            label="Device", choices=[], interactive=True
                        )
                        e_page = gr.Slider(
                            label="Page",
                            minimum=1,
                            maximum=1,
                            step=1,
                            value=1,
                            interactive=True,
                        )
                        e_btn = gr.Button("Load", variant="primary")
                        e_status = gr.Textbox(
                            label="Status", interactive=False, lines=2
                        )
                    with gr.Column(scale=2):
                        e_img = gr.Image(
                            label="Page Image", type="filepath", height=420
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        e_md = gr.Markdown(
                            "*Select a test set above, then click Load.*"
                        )
                    with gr.Column(scale=1):
                        e_df = gr.Dataframe(label="Per-Page Metrics", interactive=False)
                with gr.Row():
                    with gr.Column():
                        e_pred = gr.Textbox(
                            label="Predicted Text", interactive=False, lines=14
                        )
                    with gr.Column():
                        e_gt = gr.Textbox(
                            label="Ground Truth", interactive=False, lines=14
                        )

                e_ts.change(
                    fn=self.on_ts,
                    inputs=[e_ts],
                    outputs=[e_file, e_model, e_dev, e_page],
                )
                e_model.change(
                    fn=self.on_model, inputs=[e_ts, e_model], outputs=[e_dev]
                )
                e_file.change(fn=self.on_file, inputs=[e_ts, e_file], outputs=[e_page])
                _in = [e_ts, e_file, e_model, e_dev, e_page]
                _out = [e_img, e_pred, e_gt, e_df, e_md, e_status]
                e_btn.click(fn=self.load_explorer, inputs=_in, outputs=_out)
                e_dev.change(fn=self.load_explorer, inputs=_in, outputs=_out)
                e_page.change(fn=self.load_explorer, inputs=_in, outputs=_out)

            with gr.Tab("⚖️ Compare"):
                gr.Markdown("### Side-by-side comparison")
                with gr.Row():
                    c_ts = gr.Dropdown(label="Test Set", choices=[], interactive=True)
                    c_file = gr.Dropdown(
                        label="Input File", choices=[], interactive=True
                    )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Model A")
                        c_ma = gr.Dropdown(label="Model", choices=[], interactive=True)
                        c_da = gr.Dropdown(label="Device", choices=[], interactive=True)
                    with gr.Column():
                        gr.Markdown("#### Model B")
                        c_mb = gr.Dropdown(label="Model", choices=[], interactive=True)
                        c_db = gr.Dropdown(label="Device", choices=[], interactive=True)
                c_btn = gr.Button("Compare", variant="primary")
                with gr.Row():
                    with gr.Column():
                        c_pa = gr.Textbox(
                            label="Model A — Predicted", lines=14, interactive=False
                        )
                        c_ia = gr.Markdown()
                    with gr.Column():
                        c_pb = gr.Textbox(
                            label="Model B — Predicted", lines=14, interactive=False
                        )
                        c_ib = gr.Markdown()

                c_ts.change(
                    fn=self.on_cmp_ts,
                    inputs=[c_ts],
                    outputs=[c_file, c_ma, c_mb, c_da, c_db],
                )
                c_ma.change(fn=self.on_cmp_model, inputs=[c_ts, c_ma], outputs=[c_da])
                c_mb.change(fn=self.on_cmp_model, inputs=[c_ts, c_mb], outputs=[c_db])
                c_btn.click(
                    fn=self.load_compare,
                    inputs=[c_ts, c_file, c_ma, c_da, c_mb, c_db],
                    outputs=[c_pa, c_ia, c_pb, c_ib],
                )

            demo.load(
                fn=self.init_explorer, outputs=[e_ts, e_file, e_model, e_dev, e_page]
            )
            demo.load(
                fn=self.init_compare, outputs=[c_ts, c_file, c_ma, c_mb, c_da, c_db]
            )
            demo.load(fn=self.load_summary, outputs=summary_df)

        return demo


# ── Main ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Launch OCR Benchmark Gradio Dashboard"
    )
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host IP to bind to (use 0.0.0.0 for external access)",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Initialize UI
    dashboard = Dashboard(cfg)
    app = dashboard.build_ui()

    # Launch
    app.launch(
        theme=gr.themes.Soft(),
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
