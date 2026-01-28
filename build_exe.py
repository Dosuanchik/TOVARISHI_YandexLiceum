from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    entry = project_root / "battleship_arcade" / "main1.py"
    sounds_dir = project_root / "sounds"

    if not entry.exists():
        print(f"Entry file not found: {entry}")
        return 2
    add_data_args: list[str] = []
    if sounds_dir.exists():
        for mp3 in sounds_dir.glob("*.mp3"):
            add_data_args += ["--add-data", f"{mp3};sounds"]
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "battleship_arcade",
        *add_data_args,
        str(entry),
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(project_root), check=True)
    except FileNotFoundError:
        print("PyInstaller is not installed. Install it with: pip install pyinstaller")
        return 2
    except subprocess.CalledProcessError as e:
        return int(e.returncode)
    out_exe = project_root / "dist" / "battleship_arcade.exe"
    print("Build finished.")
    print("EXE:", out_exe)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
