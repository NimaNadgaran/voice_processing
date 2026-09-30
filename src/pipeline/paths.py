"""Pipeline *paths* -- named (denoiser + separator) combinations.

A "path" is one complete recipe: which denoiser cleans the audio, which
separator splits the speakers.  The frontend lets you tick several paths and run
them on the same upload so you can compare the results side by side; you can
also build a custom path from any pair of methods.

Presets are ordered from "always works on a bare install" to "needs the heavy
stack".  Every preset is validated at request time -- if a backend is missing the
UI greys the path out and tells you the pip command.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from ..core.registry import get_denoiser, get_separator
from ..core.utils import validate_component


@dataclass
class PipelinePath:
    id: str
    name: str
    denoiser: str
    separator: str
    tagline: str = ""
    description: str = ""  # one or two sentences, shown on the card
    badge: str = ""  # short UI chip: "fastest", "best quality", ...
    preset: bool = True

    # ---- graceful degradation ------------------------------------------------
    # Ordered stand-ins, tried when the preferred backend is not installable
    # here.  A preset that names a backend with no wheel for this interpreter
    # (DeepFilterNet on Python 3.12+) would otherwise be permanently dead; with
    # a stand-in the path still runs and the card says what it swapped and why.
    denoiser_alts: List[str] = field(default_factory=list)
    separator_alts: List[str] = field(default_factory=list)

    # ---- long-form, shown when the card's "What this path does" is expanded --
    best_for: str = ""      # the recording this path is designed for
    avoid_when: str = ""    # when to pick something else instead
    detail: str = ""        # a paragraph explaining the pairing
    output: str = ""        # what you get out of it

    def __post_init__(self) -> None:
        validate_component(self.id)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(availability(self))
        return data

    def resolved(self) -> "PipelinePath":
        """This path with any unavailable stage swapped for its stand-in.

        The runner executes the result, so filenames and the report record the
        backend that actually ran rather than the one the preset asked for.
        """
        info = availability(self)
        denoiser = info["denoiser_effective"]
        separator = info["separator_effective"]
        if denoiser == self.denoiser and separator == self.separator:
            return self
        return replace(self, denoiser=denoiser, separator=separator)


PRESET_PATHS: List[PipelinePath] = [
    PipelinePath(
        id="path1",
        name="Instant (no downloads)",
        denoiser="spectral_gate",
        separator="diarize_cluster",
        tagline="Spectral gating -> clustering diarization",
        badge="always works",
        description=(
            "Runs on a bare `pip install -r requirements.txt`. Removes steady noise "
            "and splits any number of turn-taking speakers in a couple of seconds."
        ),
        best_for=(
            "Any recording, as a first look. Meetings, interviews and voice notes with steady background noise."
        ),
        avoid_when=(
            "The background is music, or people talk over each other for most of the recording."
        ),
        detail=(
            "The safe starting point, and the only path guaranteed to run on a bare install. Spectral gating learns what the room's steady noise sounds like from the quiet moments and subtracts it; clustering diarization then fingerprints every 0.6 s of speech by pitch and timbre, groups those fingerprints into speakers, and cuts the timeline. Because the split is by turn rather than by acoustic unmixing, it handles any number of people -- four speakers really does give four files."
        ),
        output=(
            "One denoised file, plus one timeline-aligned file per speaker containing their turns."
        ),
    ),
    PipelinePath(
        id="path2",
        name="Balanced (recommended)",
        denoiser="deepfilternet",
        separator="pyannote",
        # DeepFilterNet has no wheel past CPython 3.11; pyannote needs a free
        # Hugging Face token the user has to create themselves. Rather than a
        # dead card, fall back to the best backend that is actually installed.
        denoiser_alts=["demucs_denoiser", "rnnoise", "spectral_gate"],
        separator_alts=["diarize_cluster"],
        tagline="DeepFilterNet 3 -> pyannote 3.1",
        badge="best for meetings",
        description=(
            "The pairing most real recordings want: SOTA neural denoising, then the "
            "reference diarization pipeline which finds any number of speakers and "
            "detects overlap."
        ),
        best_for=(
            "Real meetings, interviews and podcasts -- the pairing most recordings actually want."
        ),
        avoid_when=(
            "You need overlapped speech acoustically unmixed rather than attributed."
        ),
        detail=(
            "The best general-purpose combination available for free. DeepFilterNet 3 removes fans, keyboards, traffic and reverb tails at full 48 kHz while keeping the voice crisp, and then pyannote 3.1 -- the reference open-source diarization pipeline -- works out how many people there are on its own and marks every turn, including moments where two people are simultaneously active. Denoising first measurably helps the diarizer, because speaker embeddings are then computed on a cleaner signal."
        ),
        output=(
            "One denoised file, one file per speaker, and accurate turn boundaries for the timeline."
        ),
    ),
    PipelinePath(
        id="path3",
        name="Cocktail party",
        denoiser="deepfilternet",
        separator="sepformer",
        denoiser_alts=["demucs_denoiser", "rnnoise", "spectral_gate"],
        tagline="DeepFilterNet 3 -> SepFormer",
        badge="handles overlap",
        description=(
            "For people genuinely talking over each other. SepFormer acoustically "
            "unmixes simultaneous speech instead of just cutting the timeline."
        ),
        best_for=(
            "Two or three people genuinely talking at the same time, most of the time."
        ),
        avoid_when=(
            "You have four or more speakers -- no open separation model outputs four sources."
        ),
        detail=(
            "This is the only kind of path that truly unmixes overlapping voices. Diarization can only say 'both of you are talking now'; SepFormer reconstructs each voice as a complete, continuous signal, even underneath the other person. The cost is that the number of outputs is baked into the model weights (2 or 3), it is slow, and it was trained on clean studio mixtures -- which is exactly why DeepFilterNet runs first."
        ),
        output=(
            "One denoised file, plus 2-3 continuously separated voices (not turn-cut)."
        ),
    ),
    PipelinePath(
        id="path4",
        name="Classic DSP",
        denoiser="wiener_mmse",
        separator="diarize_cluster",
        tagline="MMSE-LSA -> clustering diarization",
        badge="most natural",
        description=(
            "Zero machine learning. The MMSE-LSA estimator never invents artefacts, "
            "so it is the safest choice when the audio will be used as evidence or "
            "fed to a transcriber."
        ),
        best_for=(
            "Audio used as evidence, transcribed, or analysed -- anywhere invented detail is unacceptable."
        ),
        avoid_when=(
            "You want maximum noise removal and do not mind a neural model's artefacts."
        ),
        detail=(
            "Zero machine learning end to end. The MMSE-LSA estimator is a statistical result from 1985 that computes, for each frequency at each instant, the most likely clean amplitude given the observation and a running noise model. It is gentle -- expect 6-12 dB rather than 20 -- but it cannot hallucinate, cannot smear, and produces no artefacts. Paired with numpy clustering the whole path is deterministic and auditable."
        ),
        output=(
            "One conservatively cleaned file, plus one file per speaker."
        ),
    ),
    PipelinePath(
        id="path5",
        name="Music / TV background",
        denoiser="demucs_vocals",
        separator="convtasnet_asteroid",
        tagline="Demucs vocal isolation -> Conv-TasNet",
        badge="beats music",
        description=(
            "When the interference is music or a TV rather than hiss: strip "
            "everything that is not a voice, then separate the voices."
        ),
        best_for=(
            "Speech recorded over music, a television, or a cafe with a sound system."
        ),
        avoid_when=(
            "The background is only hiss or a fan -- this is slow overkill."
        ),
        detail=(
            "Music defeats ordinary denoisers, because to a noise-print method music is not noise: it is structured, harmonic and constantly changing, so it reads as signal. htdemucs is trained to split a mixture into drums, bass, other and vocals; keeping just the vocals stem removes the band and leaves the voices. Conv-TasNet then separates the remaining speakers. The result also reports how much energy each musical stem held, so you can see how much of the file was actually music."
        ),
        output=(
            "One vocals-only file, plus 2-3 separated voices, and a stem-energy breakdown."
        ),
    ),
    PipelinePath(
        id="path6",
        name="Your own models",
        denoiser="local_unet",
        separator="local_convtasnet",
        tagline="Local SpectralUNet -> local Conv-TasNet",
        badge="fully yours",
        description=(
            "Both stages use checkpoints you trained yourself. Nothing downloads, "
            "nothing phones home, and both are tuned to your data."
        ),
        best_for=(
            "A recurring situation -- the same room, mic and people -- where a generic model is merely average."
        ),
        avoid_when=(
            "You have not trained the checkpoints yet, or you only have this one file."
        ),
        detail=(
            "Both stages use models you trained yourself with the scripts in src/*/training/. Nothing is downloaded and nothing leaves the machine. A generic model has to work for everybody, so it is average everywhere; a small model trained on twenty minutes of your actual conditions can beat models a hundred times its size on that material. The app never starts training -- you run the command, and these turn green once the checkpoints exist."
        ),
        output=(
            "One denoised file and one file per speaker, from models that are entirely yours."
        ),
    ),
    PipelinePath(
        id="path7",
        name="Cloud (no local compute)",
        denoiser="api_huggingface",
        separator="api_huggingface",
        tagline="Hugging Face API -> Hugging Face API",
        badge="uploads audio",
        description=(
            "Everything runs on Hugging Face's free tier. Useful on a weak machine "
            "-- but your audio leaves your computer, so it is off by default."
        ),
        best_for=(
            "A machine too weak to run models locally, or one where a 2.5 GB torch wheel is not an option."
        ),
        avoid_when=(
            "The audio is sensitive in any way, or you are offline."
        ),
        detail=(
            "Both stages run on Hugging Face's free Inference API, so your machine only uploads and downloads. The trade is explicit and unavoidable: the audio LEAVES YOUR COMPUTER and is processed by a third party. This path stays disabled until you set HF_TOKEN yourself, and every other method in this project is fully local. The free tier is also rate limited and cold-starts slowly."
        ),
        output=(
            "One denoised file and 2-3 separated speakers, produced remotely."
        ),
    ),
    PipelinePath(
        id="path8",
        name="Control (no denoising)",
        denoiser="none",
        separator="diarize_cluster",
        tagline="raw audio -> clustering diarization",
        badge="A/B baseline",
        description=(
            "Skips denoising entirely. Run it next to any other path to prove "
            "whether the denoiser helped or hurt the separation."
        ),
        best_for=(
            "Running alongside another path to prove whether the denoiser earned its place."
        ),
        avoid_when=(
            "Used on its own -- it removes no noise at all."
        ),
        detail=(
            "The control condition, and more useful than it sounds. Denoising is not automatically good for separation: aggressive noise removal strips exactly the low-level spectral detail that speaker embeddings rely on, and it can make a quiet speaker disappear entirely. Running the raw audio through the same separator gives you the counterfactual, and the comparison table shows immediately whether cleaning helped."
        ),
        output=(
            "The untouched audio, plus one file per speaker separated from it."
        ),
    ),
    PipelinePath(
        id="path9",
        name="Real-time stack",
        denoiser="rnnoise",
        separator="diarize_cluster",
        tagline="RNNoise -> clustering diarization",
        badge="lightest",
        description=(
            "The cheapest neural option: an 85 kB GRU denoiser plus numpy "
            "clustering. Practical for hour-long recordings on an old laptop."
        ),
        best_for=(
            "Hour-long recordings, old laptops, or batch work where wall-clock time dominates."
        ),
        avoid_when=(
            "Quality matters more than speed -- DeepFilterNet is far stronger."
        ),
        detail=(
            "The cheapest neural option that exists. RNNoise is an 85 kilobyte GRU that ships inside its wheel: no torch, no GPU, no download. It predicts gains for 22 perceptual bands every 10 ms, which is coarse but costs well under 1% of a CPU core in real time. Paired with numpy clustering, a one-hour recording finishes in a couple of minutes on hardware that would take hours with the neural paths."
        ),
        output=(
            "One lightly denoised file, one file per speaker, and a free voice-activity curve."
        ),
    ),
]

PRESETS_BY_ID: Dict[str, PipelinePath] = {p.id: p for p in PRESET_PATHS}


# --------------------------------------------------------------------------- #
def _stage(kind: str, preferred: str, alts: List[str]) -> Tuple[str, str, str, str, Optional[Dict[str, Any]]]:
    """Pick the backend this stage will really use.

    Returns ``(key, name, blocker, install_hint, substitution)``.  ``substitution``
    is None whenever the preferred backend is the one that runs -- it is only
    filled in when a stand-in took over, so the UI can say so out loud.
    """
    get = get_denoiser if kind == "denoise" else get_separator
    try:
        first = get(preferred).describe()
    except Exception as exc:
        return preferred, preferred, "%s '%s': %s" % (kind, preferred, exc), "", None

    if first.available:
        return preferred, first.name, "", "", None

    for alt in alts:
        try:
            stand_in = get(alt).describe()
        except Exception:
            continue
        if stand_in.available:
            return alt, stand_in.name, "", "", {
                "stage": kind,
                "from": preferred,
                "from_name": first.name,
                "to": alt,
                "to_name": stand_in.name,
                "reason": first.unavailable_reason,
                "hint": first.install_hint,
            }

    return preferred, first.name, "%s: %s" % (first.name, first.unavailable_reason), first.install_hint, None


def availability(path: PipelinePath) -> Dict[str, Any]:
    """Is every stage of this path runnable right now, and with what?"""
    d_key, d_name, d_problem, d_hint, d_sub = _stage("denoise", path.denoiser, path.denoiser_alts)
    s_key, s_name, s_problem, s_hint, s_sub = _stage("separate", path.separator, path.separator_alts)

    problems = [p for p in (d_problem, s_problem) if p]
    hints = [h for h in (d_hint, s_hint) if h]
    subs = [x for x in (d_sub, s_sub) if x]

    # A path can use the same backend for both stages (the Hugging Face API path
    # does), which would otherwise print the identical blocker and hint twice.
    return {
        "available": not problems,
        "blockers": _unique(problems),
        "install_hints": _unique(hints),
        "denoiser_name": d_name,
        "separator_name": s_name,
        "denoiser_effective": d_key,
        "separator_effective": s_key,
        "substitutions": subs,
    }


def _unique(items: List[str]) -> List[str]:
    """De-duplicate, keeping first-seen order."""
    seen: set = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def list_paths() -> List[Dict[str, Any]]:
    return [p.to_dict() for p in PRESET_PATHS]


def resolve_path(spec: Dict[str, Any]) -> PipelinePath:
    """Accept either ``{"id": "path1"}`` or a custom ``{"denoiser","separator"}``."""
    if not isinstance(spec, dict):
        raise ValueError("path spec must be an object")

    path_id = str(spec.get("id") or "").strip()
    if path_id and path_id in PRESETS_BY_ID and not (spec.get("denoiser") or spec.get("separator")):
        return PRESETS_BY_ID[path_id]

    denoiser = str(spec.get("denoiser") or "none")
    separator = str(spec.get("separator") or "diarize_cluster")
    get_denoiser(denoiser)
    get_separator(separator)
    return PipelinePath(
        id=path_id or ("custom_%s_%s" % (denoiser, separator)),
        name=str(spec.get("name") or ("%s + %s" % (denoiser, separator))),
        denoiser=denoiser,
        separator=separator,
        tagline="custom path",
        description="User-defined combination.",
        badge="custom",
        preset=False,
    )
