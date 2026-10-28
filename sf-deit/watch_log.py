import argparse
import json
import time
from pathlib import Path

DEFAULT_LOG_DIR = Path("output/finetune-imagenet-1k")
DEFAULT_REFRESH_SECONDS = 10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="实时绘制一个或多个训练目录中的 log.txt 指标。"
    )
    parser.add_argument(
        "--log-dir",
        "-d",
        dest="log_dirs",
        action="append",
        nargs="+",
        type=Path,
        metavar="DIR",
        help=(
            "log.txt 所在目录；可一次指定多个目录，也可重复使用该参数。"
            f"默认：{DEFAULT_LOG_DIR}"
        ),
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=DEFAULT_REFRESH_SECONDS,
        metavar="SECONDS",
        help=f"刷新间隔（秒），默认：{DEFAULT_REFRESH_SECONDS:g}",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="仅在终端显示最新指标；不需要安装 matplotlib。",
    )
    args = parser.parse_args()
    args.log_dirs = (
        [directory for group in args.log_dirs for directory in group]
        if args.log_dirs
        else [DEFAULT_LOG_DIR]
    )
    if args.refresh <= 0:
        parser.error("--refresh 必须大于 0")
    return args


def read_metrics(log_path: Path):
    metrics = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc1": [],
        "test_acc5": [],
        "train_lr": [],
    }

    if not log_path.exists():
        return metrics

    records_by_epoch = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # 训练进程可能正在写最后一行，暂时跳过不完整内容。
                continue

            if "epoch" not in record:
                continue

            # A resumed run may append an epoch that already exists. Keep the
            # most recent complete record so curves do not double back.
            records_by_epoch[record["epoch"]] = record

    for epoch in sorted(records_by_epoch):
        record = records_by_epoch[epoch]
        for key in metrics:
            metrics[key].append(record.get(key))

    return metrics


def directory_labels(log_dirs):
    """生成可读且不重复的曲线标签。"""
    names = [directory.resolve().name or str(directory.resolve()) for directory in log_dirs]
    counts = {name: names.count(name) for name in names}
    return [
        str(directory.resolve()) if counts[name] > 1 else name
        for directory, name in zip(log_dirs, names)
    ]


def plot_metrics(axes, runs):
    ax_loss, ax_acc, ax_lr = axes
    for axis in axes:
        axis.clear()

    for label, metrics in runs:
        epochs = metrics["epoch"]
        if not epochs:
            continue
        ax_loss.plot(epochs, metrics["train_loss"], label=f"{label} train")
        ax_loss.plot(
            epochs, metrics["test_loss"], linestyle="--", label=f"{label} test"
        )
        ax_acc.plot(epochs, metrics["test_acc1"], label=f"{label} top-1")
        if any(value is not None for value in metrics["test_acc5"]):
            ax_acc.plot(
                epochs,
                metrics["test_acc5"],
                linestyle="--",
                label=f"{label} top-5",
            )
        ax_lr.plot(epochs, metrics["train_lr"], label=label)

    settings = (
        (ax_loss, "Training and Test Loss", "Loss"),
        (ax_acc, "Validation Accuracy", "Accuracy (%)"),
        (ax_lr, "Learning Rate", "Learning rate"),
    )
    for axis, title, ylabel in settings:
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True)
        if axis.lines:
            axis.legend()


def format_value(value, spec):
    return "N/A" if value is None else format(value, spec)


def print_latest(runs, log_paths):
    summaries = []
    for label, metrics in runs:
        if not metrics["epoch"]:
            continue
        latest = -1
        summaries.append(
            f"{label}: epoch {metrics['epoch'][latest]} | "
            f"train_loss {format_value(metrics['train_loss'][latest], '.4f')} | "
            f"test_loss {format_value(metrics['test_loss'][latest], '.4f')} | "
            f"acc1 {format_value(metrics['test_acc1'][latest], '.2f')}% | "
            f"acc5 {format_value(metrics['test_acc5'][latest], '.2f')}% | "
            f"lr {format_value(metrics['train_lr'][latest], '.6g')}"
        )
    if summaries:
        print("\r" + " || ".join(summaries), end="", flush=True)
    else:
        waiting = " || ".join(
            f"{label}: waiting for {log_path}"
            for (label, _), log_path in zip(runs, log_paths)
        )
        print("\r" + waiting, end="", flush=True)


def main():
    args = parse_args()
    labels = directory_labels(args.log_dirs)
    log_paths = [directory / "log.txt" for directory in args.log_dirs]

    if args.no_plot:
        try:
            while True:
                runs = [
                    (label, read_metrics(log_path))
                    for label, log_path in zip(labels, log_paths)
                ]
                print_latest(runs, log_paths)
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            print("\n停止监控，训练进程不会受到影响。")
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise SystemExit(
            "绘图模式需要 matplotlib。请运行 `conda install matplotlib`，"
            "或使用 `--no-plot` 进行终端监控。"
        ) from error

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.canvas.manager.set_window_title("Training Log Monitor")

    try:
        while plt.fignum_exists(fig.number):
            runs = [
                (label, read_metrics(log_path))
                for label, log_path in zip(labels, log_paths)
            ]
            plot_metrics(axes, runs)
            fig.tight_layout()
            fig.canvas.draw_idle()
            print_latest(runs, log_paths)
            plt.pause(args.refresh)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n停止监控，训练进程不会受到影响。")


if __name__ == "__main__":
    main()
