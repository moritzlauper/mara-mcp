"""`mara-bridge setup`, `check` and `config`: the parts a person runs in a terminal."""
from __future__ import annotations

import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from . import MARA_REPO

UVX = "uvx --python 3.12 --from git+https://github.com/moritzlauper/mara-bridge mara-bridge"


def _mara_source() -> tuple[Path | None, str | None]:
    """The MARA checkout the installed `mixmatch` package comes from, or the git commit it was installed at."""
    from importlib import metadata

    try:
        direct = json.loads(metadata.distribution("mixmatch").read_text("direct_url.json") or "{}")
    except metadata.PackageNotFoundError:
        sys.exit("The MARA package (mixmatch) is not installed. Install mara-bridge with pip or uvx, see README.")
    url = direct.get("url", "")
    if url.startswith("file://"):
        root = Path(url[len("file://"):])
        if (root / "remote_script/MixMatch").is_dir():
            return root, None
    import mixmatch
    root = Path(mixmatch.__file__).resolve().parent.parent
    if (root / "remote_script/MixMatch").is_dir():
        return root, None
    return None, direct.get("vcs_info", {}).get("commit_id") or "main"


@contextmanager
def _live_files():
    """A folder with remote_script/MixMatch and device/ of the same MARA version as the Python code."""
    root, commit = _mara_source()
    if root is not None:
        yield root
        return
    url = f"https://codeload.github.com/{MARA_REPO}/tar.gz/{commit}"
    print(f"Downloading the Live connection of MARA {commit[:10]} …")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "mara.tar.gz"
        with urllib.request.urlopen(url, timeout=60) as response, open(archive, "wb") as f:
            shutil.copyfileobj(response, f)
        out = Path(tmp) / "mara"
        with tarfile.open(archive) as tar:
            wanted = [m for m in tar.getmembers() if m.isfile() and "/" in m.name
                      and m.name.split("/", 1)[1].startswith(("remote_script/MixMatch/", "device/"))
                      and ".." not in m.name]
            for m in wanted:
                target = out / m.name.split("/", 1)[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        yield out


def setup(library: str | None = None) -> None:
    from mixmatch import platforms

    library = library or str(platforms.library_default())
    with _live_files() as root:
        original = platforms.resources
        platforms.resources = lambda: root     # MARA's installer, pointed at these files
        try:
            result = platforms.install_bridge(library)
        except ValueError as e:
            sys.exit(f"{e} Not found: {library}. Pass it with --library.")
        finally:
            platforms.resources = original
    print(f"Installed into {result['library']}.\n")
    print("Next:")
    print("  1. Restart Live. Settings → Link/Tempo/MIDI → Control Surface: choose «MixMatch», Input and Output «None».")
    print("  2. Register the bridge in your agent:  mara-bridge config")
    print("  3. Open your set and ask the agent to add the measurement taps (or: mara-bridge check).")


def check() -> None:
    from mixmatch.live_client import LiveClient

    try:
        live = LiveClient().connect()
        live.request("ping")
    except OSError:
        sys.exit("✗ Live is not reachable on port 9877. Is Live open, and «MixMatch» selected as a control surface? "
                 "Run `mara-bridge setup` first if you have not.")
    state = live.get_set()
    tracks = state["tracks"]
    taps = [t for t in tracks if any(d["name"] == "MixMatch Tap" for d in t["devices"])]
    print(f"✓ Live: «{state.get('name') or 'unsaved set'}», {len(tracks)} tracks, {state.get('tempo')} BPM")
    print(f"{'✓' if len(taps) == len(tracks) else '•'} Taps on {len(taps)} of {len(tracks)} tracks"
          + ("" if taps else " (the agent adds them with add_taps)"))
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", 9878))
        print("✓ Port 9878 for the tap measurements is free")
    except OSError:
        print("• Port 9878 is in use: close the MARA app (or another bridge) while your agent mixes")
    finally:
        s.close()


def config() -> None:
    exe = shutil.which("mara-bridge") or sys.argv[0]
    print("Claude Code:\n")
    print(f"  claude mcp add mara-bridge -- {UVX}\n")
    print("Codex (~/.codex/config.toml):\n")
    command, *args = UVX.split()
    print(f'  [mcp_servers.mara-bridge]\n  command = "{command}"\n  args = {json.dumps(args)}\n')
    print("Claude Desktop, Cursor, Windsurf and others (mcpServers in their JSON settings):\n")
    print("  " + json.dumps({"mcpServers": {"mara-bridge": {"command": command, "args": args}}}, indent=2)
          .replace("\n", "\n  ") + "\n")
    if exe and Path(exe).exists() and ".cache/uv" not in exe and "uv/archive" not in exe:
        print(f"Using this installation instead of uvx: replace the command with {Path(exe).resolve()}")
