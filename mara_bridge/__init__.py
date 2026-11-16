"""MARA Bridge: an MCP server that lets Claude Code, Codex or any MCP client hear and change an Ableton Live set."""
__version__ = "0.1.0"

# The MARA version this bridge is built and tested against: its Python package (pyproject.toml) and the
# Live remote script and tap that `mara-bridge setup` installs come from the same commit.
MARA_REPO = "moritzlauper/mara"


def names() -> tuple[str, str]:
    """(tap device, control surface) as the installed MARA calls them: «MixMatch Tap»/«MixMatch» before the
    rename, «MARA Tap»/«MARA» after it."""
    from mixmatch.cli import TAP

    return TAP, ("MARA" if TAP.startswith("MARA") else "MixMatch")


TAP, SURFACE = names()
