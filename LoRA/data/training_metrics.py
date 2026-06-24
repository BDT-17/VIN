"""Training metrics: real-time monitoring, loss logging, and divergence detection.

Wraps the Diffusers training subprocess with live output parsing.
Hard-fails immediately on NaN/Inf loss or gradient.
Writes per-step metrics.jsonl, summary.json, and loss_curve.png.
"""

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


class TrainingMonitor:
    """Monitor a Diffusers training subprocess and collect per-step metrics.

    Usage:
        monitor = TrainingMonitor(run_id="run_20240101_120000", reports_base=Path("reports/training"))
        monitor.run(command)  # raises RuntimeError on NaN/Inf
    """

    def __init__(self, run_id: str, reports_base: Path):
        self.run_id = run_id
        self.reports_dir = Path(reports_base) / run_id
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.reports_dir / "metrics.jsonl"
        self.summary_path = self.reports_dir / "summary.json"
        self.steps: List[Dict] = []

    def run(self, command: List[str]) -> int:
        """Execute command with live metric monitoring.

        Echoes stdout to the caller. Writes metrics.jsonl in real time.
        Raises RuntimeError immediately on NaN/Inf loss or gradient.

        Returns:
            Process exit code (non-zero also raises CalledProcessError)
        """
        start_time = time.time()
        prev_step_time = start_time
        print(f"[TrainingMonitor] run_id={self.run_id}")
        print(f"[TrainingMonitor] reports → {self.reports_dir}")

        with open(self.metrics_path, "w", encoding="utf-8") as metrics_file:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for raw_line in proc.stdout:
                sys.stdout.write(raw_line)
                sys.stdout.flush()

                metrics = self._parse_line(raw_line)
                if metrics is None:
                    continue

                now = time.time()
                metrics["seconds_per_step"] = round(now - prev_step_time, 3)
                prev_step_time = now
                self._inject_gpu_memory(metrics)

                metrics_file.write(json.dumps(metrics) + "\n")
                metrics_file.flush()
                self.steps.append(metrics)

                self._check_health(metrics)

            proc.wait()

        elapsed = time.time() - start_time
        self._write_summary(elapsed)

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command)

        return proc.returncode

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_line(self, line: str) -> Optional[Dict]:
        """Extract step metrics from one line of Diffusers/Accelerate output.

        Handles two formats:
        1. Accelerate JSON logger: {"loss": 0.45, "lr": 1e-4, "step": 100}
        2. tqdm bar:  "Steps: 50%|█| 50/100 [..., loss=0.45, lr=0.0001]"
        """
        stripped = line.strip()

        # --- JSON format ---
        if stripped.startswith("{") and "loss" in stripped:
            try:
                data = json.loads(stripped)
                raw_loss = data.get("loss") or data.get("train_loss")
                if raw_loss is None:
                    return None
                loss = float(raw_loss)
                lr = float(data.get("lr") or data.get("learning_rate") or 0)
                step = int(data.get("step") or data.get("global_step") or 0)
                grad = data.get("grad_norm")
                return {
                    "step": step,
                    "train_loss": loss,
                    "learning_rate": lr,
                    "grad_norm": float(grad) if grad is not None else None,
                    "loss_is_finite": math.isfinite(loss),
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # --- tqdm bar ---
        step_m = re.search(r"\b(\d+)/(\d+)", stripped)
        loss_m = re.search(r"\bloss[=: ]+([0-9.e+\-]+|nan|inf)", stripped, re.IGNORECASE)
        lr_m = re.search(r"\blr[=: ]+([0-9.e+\-]+)", stripped, re.IGNORECASE)
        grad_m = re.search(r"\bgrad(?:_norm)?[=: ]+([0-9.e+\-]+)", stripped, re.IGNORECASE)

        if step_m and loss_m:
            try:
                loss_str = loss_m.group(1).lower()
                if loss_str == "nan":
                    loss = float("nan")
                elif "inf" in loss_str:
                    loss = float("inf")
                else:
                    loss = float(loss_str)
                return {
                    "step": int(step_m.group(1)),
                    "train_loss": loss,
                    "learning_rate": float(lr_m.group(1)) if lr_m else None,
                    "grad_norm": float(grad_m.group(1)) if grad_m else None,
                    "loss_is_finite": math.isfinite(loss),
                }
            except (ValueError, AttributeError):
                pass

        return None

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def _check_health(self, metrics: Dict) -> None:
        """Raise immediately on NaN/Inf loss or gradient."""
        step = metrics.get("step", "?")
        loss = metrics.get("train_loss")
        if loss is not None:
            if math.isnan(loss):
                raise RuntimeError(f"Training diverged: NaN loss at step {step}")
            if math.isinf(loss):
                raise RuntimeError(f"Training diverged: Inf loss at step {step}")

        grad = metrics.get("grad_norm")
        if grad is not None and not math.isfinite(grad):
            raise RuntimeError(f"Training diverged: non-finite grad_norm at step {step}")

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def _inject_gpu_memory(self, metrics: Dict) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu_memory_allocated_mb"] = round(
                    torch.cuda.memory_allocated() / (1024 * 1024), 1
                )
                metrics["gpu_memory_reserved_mb"] = round(
                    torch.cuda.memory_reserved() / (1024 * 1024), 1
                )
        except Exception:
            pass

    def _write_summary(self, total_seconds: float) -> None:
        if not self.steps:
            self.summary_path.write_text(
                json.dumps({"run_id": self.run_id, "total_steps": 0}, indent=2),
                encoding="utf-8",
            )
            return

        finite_losses = [
            s["train_loss"] for s in self.steps
            if s.get("train_loss") is not None and math.isfinite(s["train_loss"])
        ]
        lrs = [s["learning_rate"] for s in self.steps if s.get("learning_rate") is not None]
        sps = [s["seconds_per_step"] for s in self.steps if s.get("seconds_per_step") is not None]

        summary = {
            "run_id": self.run_id,
            "total_steps": len(self.steps),
            "total_seconds": round(total_seconds, 1),
            "loss": {
                "first": round(finite_losses[0], 6) if finite_losses else None,
                "last": round(finite_losses[-1], 6) if finite_losses else None,
                "min": round(min(finite_losses), 6) if finite_losses else None,
            },
            "learning_rate": {
                "last": round(lrs[-1], 8) if lrs else None,
            },
            "throughput": {
                "mean_seconds_per_step": round(sum(sps) / len(sps), 3) if sps else None,
                "steps_per_second": round(len(self.steps) / total_seconds, 3) if total_seconds > 0 else None,
            },
        }

        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[TrainingMonitor] Summary → {self.summary_path}")

        if finite_losses:
            self._plot_loss_curve(finite_losses)

    def _plot_loss_curve(self, losses: List[float]) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(losses) + 1), losses, linewidth=1.5, color="#2196F3")
            ax.set_xlabel("Step")
            ax.set_ylabel("Train Loss")
            ax.set_title(f"Training Loss — {self.run_id}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            curve_path = self.reports_dir / "loss_curve.png"
            fig.savefig(curve_path, dpi=100)
            plt.close(fig)
            print(f"[TrainingMonitor] Loss curve → {curve_path}")
        except Exception as exc:
            print(f"[TrainingMonitor] Could not plot loss curve: {exc}")
