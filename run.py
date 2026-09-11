"""
Main entry point — runs the full pipeline end-to-end.

Usage:
    python run.py download     # Week 1: Download all datasets
    python run.py process      # Week 2: Feature engineering
    python run.py train        # Week 3: Model training
    python run.py dashboard    # Week 4: Launch Streamlit dashboard
    python run.py all          # Run full pipeline (download → process → train)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_download():
    from src.download_data import run_all
    run_all()


def cmd_process():
    from src.feature_engineering import run_feature_engineering
    run_feature_engineering()


def cmd_train():
    from src.model import run_training
    run_training()


def cmd_dashboard():
    import subprocess
    subprocess.run(["streamlit", "run", "app.py", "--server.headless", "true"])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    commands = {
        "download": cmd_download,
        "process": cmd_process,
        "train": cmd_train,
        "dashboard": cmd_dashboard,
    }

    if cmd == "all":
        cmd_download()
        cmd_process()
        cmd_train()
        print("\n✓ Full pipeline complete. Run 'python run.py dashboard' to view results.")
    elif cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
