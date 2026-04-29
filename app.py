"""Flask GUI for the RT Scheduler project."""

import io
import os
import base64
import json
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, send_file, abort)

import scheduler_analysis
import experiments as exp_module
import generate_gantt_charts as gantt_module
import visualizations

app = Flask(__name__)
app.secret_key = "rt-scheduler-gui-secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
TASKSETS_DIR = os.path.join(BASE_DIR, "task_sets")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _load_bundled_tasksets():
    sets = []
    for category in ("schedulable", "unschedulable"):
        cat_dir = os.path.join(TASKSETS_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if fname.endswith(".csv"):
                full = os.path.join(cat_dir, fname)
                sets.append({
                    "name": fname,
                    "label": f"[{category}] {fname}",
                    "path": full,
                })
    return sets


def _figure_label(fname: str) -> str:
    labels = {
        "fig1_tc1_dm_gantt.png": "Fig 1: TC1 DM Gantt",
        "fig2_tc1_edf_gantt.png": "Fig 2: TC1 EDF Gantt",
        "fig3_tc2_comparison_gantt.png": "Fig 3: TC2 DM vs EDF",
        "fig4_fraction_schedulable.png": "Fig 4: Fraction Schedulable",
        "fig5_wcrt_comparison.png": "Fig 5: WCRT Comparison",
        "fig6_analytical_vs_observed.png": "Fig 6: Analytical vs Observed",
        "fig7_preemptions.png": "Fig 7: Preemptions",
        "fig8_tc5_rt_boxplot.png": "Fig 8: RT Boxplots",
        "fig9_arj_u07_u08_u09.png": "Fig 9: ARJ per Task",
    }
    return labels.get(fname, fname)


def _get_stats():
    n_tasksets = 0
    for cat in ("schedulable", "unschedulable"):
        d = os.path.join(TASKSETS_DIR, cat)
        if os.path.isdir(d):
            n_tasksets += len([f for f in os.listdir(d) if f.endswith(".csv")])

    n_figures = 0
    if os.path.isdir(FIGURES_DIR):
        n_figures = len([f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")])

    n_sweep_rows = 0
    sweep_csv = os.path.join(DATA_DIR, "utilization_sweep_results.csv")
    if os.path.exists(sweep_csv):
        try:
            n_sweep_rows = len(pd.read_csv(sweep_csv))
        except Exception:
            pass

    n_data_files = 0
    if os.path.isdir(DATA_DIR):
        n_data_files = len([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])

    return {
        "n_tasksets": n_tasksets,
        "n_figures": n_figures,
        "n_sweep_rows": n_sweep_rows,
        "n_data_files": n_data_files,
    }


def _run_analysis(tasks: pd.DataFrame, num_sim_runs: int, num_hyperperiods: int, seed: int):
    results = scheduler_analysis.analyze_task_set(
        tasks,
        num_sim_runs=num_sim_runs,
        num_hyperperiods=num_hyperperiods,
        seed=seed,
        verbose=False,
    )

    df = results["results_df"]
    table_rows = df.to_dict(orient="records")

    # Inline WCRT plot
    plot_wcrt = None
    try:
        fig = visualizations.plot_wcrt_comparison(df)
        plot_wcrt = _fig_to_b64(fig)
    except Exception:
        pass

    # Inline analytical vs observed plot
    plot_obs = None
    try:
        fig = visualizations.plot_analytical_vs_observed(df)
        plot_obs = _fig_to_b64(fig)
    except Exception:
        pass

    results["table_rows"] = table_rows
    results["plot_wcrt"] = plot_wcrt
    results["plot_obs"] = plot_obs
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        stats=_get_stats(),
        bundled_tasksets=_load_bundled_tasksets(),
    )


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    bundled = _load_bundled_tasksets()
    preset = request.args.get("preset")
    results = None

    if request.method == "POST":
        mode = request.form.get("input_mode", "upload")
        num_sim_runs = int(request.form.get("num_sim_runs", 20))
        num_hyperperiods = int(request.form.get("num_hyperperiods", 1))
        seed = int(request.form.get("seed", 42))

        try:
            if mode == "upload":
                f = request.files.get("csv_file")
                if not f or f.filename == "":
                    flash("No file uploaded.", "error")
                    return render_template("analyze.html", bundled_tasksets=bundled, preset=preset)
                tasks = pd.read_csv(f)

            elif mode == "manual":
                raw = request.form.get("manual_tasks_json") or request.form.get("manual_tasks", "[]")
                task_list = json.loads(raw)
                if not task_list:
                    flash("No tasks entered.", "error")
                    return render_template("analyze.html", bundled_tasksets=bundled, preset=preset)
                tasks = pd.DataFrame(task_list)

            elif mode == "preset":
                path = request.form.get("preset_path", "")
                if not path or not os.path.exists(path):
                    flash("Task set file not found.", "error")
                    return render_template("analyze.html", bundled_tasksets=bundled, preset=preset)
                tasks = pd.read_csv(path)

            else:
                flash("Unknown input mode.", "error")
                return render_template("analyze.html", bundled_tasksets=bundled, preset=preset)

            tasks = scheduler_analysis.normalize_task_columns(tasks)
            results = _run_analysis(tasks, num_sim_runs, num_hyperperiods, seed)

        except Exception as e:
            flash(f"Analysis error: {e}", "error")

    return render_template("analyze.html", bundled_tasksets=bundled, preset=preset, results=results)


@app.route("/figures")
def figures():
    figs = []
    if os.path.isdir(FIGURES_DIR):
        for fname in sorted(os.listdir(FIGURES_DIR)):
            if fname.endswith(".png"):
                figs.append({"filename": fname, "label": _figure_label(fname)})
    return render_template("figures.html", figures=figs)


@app.route("/figures/img/<filename>")
def figures_img(filename):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/figures/download/<filename>")
def figures_download(filename):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/png", as_attachment=True, download_name=filename)


@app.route("/figures/regenerate", methods=["POST"])
def figures_regenerate():
    try:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        gantt_module.main()
        flash("Gantt charts regenerated.", "success")
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect(url_for("figures"))


@app.route("/experiments", methods=["GET", "POST"])
def experiments():
    sweep_results = None
    sweep_plot = None
    existing_summary = None

    summary_csv = os.path.join(DATA_DIR, "fraction_schedulable_summary.csv")
    if os.path.exists(summary_csv):
        try:
            existing_summary = pd.read_csv(summary_csv).to_dict(orient="records")
        except Exception:
            pass

    if request.method == "POST":
        try:
            raw_levels = request.form.get("util_levels", "0.5,0.7,0.9,1.0")
            util_levels = [float(x.strip()) for x in raw_levels.split(",") if x.strip()]
            samples = int(request.form.get("samples_per_level", 100))
            n_tasks_raw = request.form.get("n_tasks", "varied")
            n_tasks = None if n_tasks_raw == "varied" else int(n_tasks_raw)

            sweep_df = exp_module.run_utilization_sweep(
                utilization_levels=util_levels,
                samples_per_level=samples,
                n_tasks=n_tasks,
                tasksets_dir=os.path.join(TASKSETS_DIR, "generated", "sweep"),
                output_dir=DATA_DIR,
                verbose=False,
            )
            summary_df = exp_module.compute_fraction_schedulable(sweep_df)
            summary_df.to_csv(summary_csv, index=False)
            sweep_results = summary_df.to_dict(orient="records")
            existing_summary = sweep_results

            fig = visualizations.plot_fraction_schedulable(summary_df)
            sweep_plot = _fig_to_b64(fig)
            os.makedirs(FIGURES_DIR, exist_ok=True)
            summary_df_for_plot = summary_df.copy()
            visualizations.plot_fraction_schedulable(
                summary_df_for_plot,
                os.path.join(FIGURES_DIR, "fig4_fraction_schedulable.png")
            )
            flash("Sweep complete.", "success")
        except Exception as e:
            flash(f"Sweep error: {e}", "error")

    return render_template(
        "experiments.html",
        sweep_results=sweep_results,
        sweep_plot=sweep_plot,
        existing_summary=existing_summary,
    )


@app.route("/overload", methods=["GET", "POST"])
def overload():
    results = None
    if request.method == "POST":
        try:
            overload_u = float(request.form.get("overload_u", 1.05))
            n_tasksets = int(request.form.get("n_tasksets", 50))
            n_tasks = int(request.form.get("n_tasks", 6))
            sim_hyperperiods = int(request.form.get("sim_hyperperiods", 4))

            df = exp_module.run_overload_deadline_miss_analysis(
                overload_utilization=overload_u,
                n_tasksets=n_tasksets,
                n_tasks=n_tasks,
                sim_hyperperiods=sim_hyperperiods,
                output_dir=DATA_DIR,
                verbose=False,
            )
            if not df.empty:
                summary = df.groupby("priority_rank")[
                    ["dm_miss_fraction", "edf_miss_fraction"]
                ].mean().reset_index()
                results = summary.to_dict(orient="records")
            else:
                flash("No results generated.", "warning")
        except Exception as e:
            flash(f"Error: {e}", "error")
    return render_template("overload.html", results=results)


@app.route("/gantt", methods=["GET"])
def gantt():
    saved = []
    if os.path.isdir(FIGURES_DIR):
        for fname in ["fig1_tc1_dm_gantt.png", "fig2_tc1_edf_gantt.png", "fig3_tc2_comparison_gantt.png"]:
            path = os.path.join(FIGURES_DIR, fname)
            if os.path.exists(path):
                saved.append({"filename": fname, "label": _figure_label(fname)})
    return render_template("gantt.html", gantt_plots=None, saved_gantt_figs=saved)


@app.route("/gantt/builtin", methods=["POST"])
def gantt_builtin():
    try:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        gantt_module.main()
        flash("TC1 and TC2 Gantt charts generated.", "success")
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect(url_for("gantt"))


@app.route("/gantt/custom", methods=["POST"])
def gantt_custom():
    gantt_plots = []
    saved = []
    if os.path.isdir(FIGURES_DIR):
        for fname in ["fig1_tc1_dm_gantt.png", "fig2_tc1_edf_gantt.png", "fig3_tc2_comparison_gantt.png"]:
            if os.path.exists(os.path.join(FIGURES_DIR, fname)):
                saved.append({"filename": fname, "label": _figure_label(fname)})

    try:
        raw = request.form.get("manual_tasks", "[]")
        task_list = json.loads(raw)
        if not task_list:
            flash("No tasks provided.", "error")
            return render_template("gantt.html", gantt_plots=None, saved_gantt_figs=saved)

        tasks_df = pd.DataFrame(task_list)
        tasks_df["WCET"] = pd.to_numeric(tasks_df["WCET"])
        tasks_df["Period"] = pd.to_numeric(tasks_df["Period"])
        tasks_df["Deadline"] = pd.to_numeric(tasks_df["Deadline"])
        if "BCET" not in tasks_df.columns:
            tasks_df["BCET"] = tasks_df["WCET"]
        tasks_df = tasks_df.reset_index(drop=True)

        policy = request.form.get("policy", "both")
        time_limit_raw = int(request.form.get("time_limit", 0))
        time_limit = time_limit_raw if time_limit_raw > 0 else None

        H = scheduler_analysis.compute_hyperperiod(tasks_df["Period"].tolist())
        tl = time_limit or H
        policies = ["DM", "EDF"] if policy == "both" else [policy]

        for pol in policies:
            events = gantt_module.simulate_schedule_gantt(tasks_df, policy=pol, time_limit=tl)
            fig, ax = plt.subplots(figsize=(11, max(3, len(tasks_df) * 1.2)))
            gantt_module.draw_gantt(ax, events, tasks_df, f"Custom Task Set: {pol} (H={H})", tl)
            fig.tight_layout()
            gantt_plots.append({"label": f"{pol} Schedule", "data": _fig_to_b64(fig)})

    except Exception as e:
        flash(f"Gantt error: {e}", "error")

    return render_template("gantt.html", gantt_plots=gantt_plots, saved_gantt_figs=saved)


@app.route("/data")
def data_files():
    files = []
    if os.path.isdir(DATA_DIR):
        for fname in sorted(os.listdir(DATA_DIR)):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(DATA_DIR, fname)
            size_kb = round(os.path.getsize(fpath) / 1024, 1)
            preview = None
            rows = 0
            try:
                df = pd.read_csv(fpath)
                rows = len(df)
                preview_df = df.head(5)
                preview = {
                    "columns": list(preview_df.columns),
                    "rows": preview_df.values.tolist(),
                }
            except Exception:
                pass
            files.append({"name": fname, "size_kb": size_kb, "rows": rows, "preview": preview})
    return render_template("data_files.html", files=files)


@app.route("/data/download/<filename>")
def data_download(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path) or not filename.endswith(".csv"):
        abort(404)
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name=filename)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    app.run(debug=True, port=5000)
