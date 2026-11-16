# MARA Bridge

**Give Claude ears in Ableton Live.**

MARA Bridge is an MCP server that connects Claude (the desktop app or Claude Code), Codex, Cursor, or any other MCP client to your Ableton Live set. Your agent can then measure every channel, look at spectra and spectrograms, compare your mix with reference tracks, and set volume, pan, EQ and compression itself.

A model that only talks to Live's API sees track names and knob values. It changes a parameter and can't tell what happened. MARA Bridge adds measurement: a transparent tap at the end of every channel, the same analysis for reference files, and a recording of any passage when finer detail is needed. Every change is saved first and can be undone with one call.

> "Listen to the drop at bar 33. What's muddy?"
> "Here's my reference: ~/Music/refs/song.wav. Make it as loud as this without squashing the drums."
> "Make the kick like the one in the reference."
> "Show me the spectrum of the bass against the reference."
> "Mix it like this, but leave room for vocals."

## What it can do

**Hear**

- LUFS, peak, RMS, crest factor (transients), PLR, stereo width and mono low end for every channel, every group and Main, after faders. All channels are measured at once while Live plays.
- Masking between channels: which channel covers which, at which frequencies and how strongly.
- Spectrum, like SPAN: average, max hold and side, with a picture the agent can look at. Third-octave from the taps for any number of channels at once, or 1/6 to 1/24 octave from a recording of one channel or Main. Resonance peaks are listed.
- Spectrogram of any channel, Main or audio file, as a picture and a compact table.

**Compare with references**

- Loudness and density: integrated and loud-part LUFS, true peak, PLR, loudness range, crest.
- Tonal balance, stereo width, mono low end.
- Elements: kick, snare, hats, drums, bass, vocals and music. On your side every element comes from its own channels (grouped by role, which you can correct). From a reference master, kick, snare, hats and bass are read from their hits and timing: fundamental, decay, sub, knock, click, crack, snap, brightness, punch against the bass.
- Overlaid spectra with a level-matched difference curve.

**Change**

- Volume, pan and sends.
- EQ Eight by frequency, gain, Q and type (bell, shelves, cuts, notch). It reuses a matching band or picks a free one, and loads EQ Eight if the track has none.
- Live's Compressor: threshold, ratio, attack, release, makeup, knee, dry/wet, model.
- Any other device parameter, rack macro and mixer control. Load native devices or your installed VST3/AU plugins.
- Snapshots: every parameter is saved before the first change, and `restore` sets them all back.

## Requirements

- Ableton Live 11 or 12 with Max for Live.
- macOS or Windows.
- [uv](https://docs.astral.sh/uv/) (it fetches Python and everything else).
- An MCP client: the Claude desktop app, Claude Code, Codex, Cursor, …

## Setup

**1. Install the Live connection** (a control surface script and the measurement tap):

```bash
uvx --python 3.12 --from git+https://github.com/moritzlauper/mara-bridge mara-bridge setup
```

If your User Library is not in the default place, add `--library "/path/to/User Library"`.

**2. In Live:** restart it. In **Settings → Link/Tempo/MIDI → Control Surface**, choose **MixMatch** (**MARA** in newer versions) and set Input and Output to **None**.

**3. Register the bridge with your agent.** `mara-bridge config` prints these lines for you.

Claude desktop app (Settings → Developer → Edit Config, then restart Claude):

```json
{
  "mcpServers": {
    "mara-bridge": {
      "command": "uvx",
      "args": ["--python", "3.12", "--from", "git+https://github.com/moritzlauper/mara-bridge", "mara-bridge"]
    }
  }
}
```

If Claude can't find `uvx`, use its full path (`which uvx`, usually `~/.local/bin/uvx`).

Claude Code:

```bash
claude mcp add mara-bridge -- uvx --python 3.12 --from git+https://github.com/moritzlauper/mara-bridge mara-bridge
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.mara-bridge]
command = "uvx"
args = ["--python", "3.12", "--from", "git+https://github.com/moritzlauper/mara-bridge", "mara-bridge"]
tool_timeout_sec = 120
```

**4. Open your set and say hello.** Ask the agent to add the measurement taps; it calls `add_taps` once per set. `mara-bridge check` tests the connection from a terminal.

The first start downloads the dependencies and can take a minute. If the [MARA app](https://github.com/moritzlauper/mara) is open, close it: the taps report to one program at a time.

## References

Give the agent a path to a WAV, AIFF, FLAC or MP3 file. In Claude Code you can drag the file into the chat. In the Claude desktop app, paste the path; audio attachments don't reach a local MCP server. You can also put the song on an audio track named `REF …` in your set, and the bridge finds it there.

A reference is a finished stereo master, so not everything in it can be measured on its own. Loudness, tonal balance, stereo and spectrum are exact. Kick, snare, hats and bass come from their timing: a hit is a sharp rise in its band that falls again, and the bass is what stays in the low end between the kicks. That works for most kick-driven music. A bass that is louder than the kick hides hits, and vocals and the rest of the music can't be separated from a master. Your own set has no such limit, because every channel is measured solo.

## Tools

| Tool | What it does |
| --- | --- |
| `status` | Connection, tempo, transport, cue points, channels without a tap |
| `get_set` | Tracks, devices and parameters with display values and ranges |
| `listen` | Measure a passage: Main and every channel, masking, distance to references |
| `spectrum` | Frequency response (avg, max hold, side) as numbers and picture, optionally against a reference |
| `spectrogram` | Level over time and frequency as picture and table |
| `compare_elements` | Kick, snare, hats, drums, bass, vocals, music: yours against the reference |
| `record_and_compare` | Record Main or one channel and compare loudness, tone, hits and character with a reference |
| `analyze_audio` | Describe a reference file |
| `load_references` | Make reference files the target of every `listen` |
| `mixer` | Volume, pan, sends |
| `eq` | EQ Eight bands |
| `compressor` | Live's Compressor |
| `set_parameters` | Any parameter, in display units or raw |
| `add_device`, `remove_device` | Load or remove native devices and plugins |
| `transport` | Play, stop, jump, loop |
| `add_taps` | Put the measurement tap on every channel |
| `snapshot`, `restore`, `list_snapshots` | Save and recall every parameter |

## Safety

- Work on a copy of an important set.
- The bridge snapshots every parameter before its first change, and `restore` sets them back. A device the agent removed is not loaded back.
- Recordings (`source: recording`, `record_and_compare`) play the passage audibly. They record behind the Main fader, so Main can't be at −inf.
- Numbers are decision support. Listen before you keep anything.

## How it relates to MARA

[MARA](https://github.com/moritzlauper/mara) is a reference-guided mixing app that runs its own measured mix pass. MARA Bridge uses MARA's Live connection, taps and analysis, and leaves the decisions to your agent.

## Development

```bash
git clone https://github.com/moritzlauper/mara https://github.com/moritzlauper/mara-bridge
cd mara-bridge
uv venv --python 3.12 .venv
uv pip install --python .venv -e ../mara pytest
uv pip install --python .venv --no-deps -e .
.venv/bin/python -m pytest
```

The tests run against MARA's built-in Live simulator and synthetic audio, so they need neither Live nor a model account.

The protocol is MCP over stdio (JSON-RPC 2.0), implemented in [server.py](mara_bridge/server.py) without an MCP SDK. Stdout carries only the protocol, and everything else goes to stderr.
