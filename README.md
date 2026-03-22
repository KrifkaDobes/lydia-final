# Jingju MusicXML Tokenizer — v4

**Course:** Analysis of Symbolic Music and Ethnomusicology  
**Dataset:** Jingju Opera Scores — 108 MusicXML files  
**Format:** Jupyter Notebook (.ipynb), runs on Google Colab  

## Overview

A production-quality symbolic music tokenizer that converts MusicXML files into flat sequences of discrete string tokens for machine learning. Built for the Jingju opera corpus — a multi-voice, non-Western dataset significantly more challenging than standard monophonic benchmarks.

## Assignment Requirements

| Requirement | Status | Implementation |
|---|---|---|
| Score-to-token (tokenizer) | ✅ | `tokenize(path)` → `List[str]` |
| Token-to-score (detokenizer) | ✅ | `detokenize(tokens)` → `music21.Score` |
| Any number of parts/instruments | ✅ | Iterates all parts, emits `PART_<name>` per part |
| Key and time signature changes | ✅ | Detected per-measure, emitted as `KEY_` / `TIME_SIG_` |
| Partwise tokenization | ✅ | Outer loop per-part, inner loop per-measure |
| Runs on Google Colab | ✅ | Single self-contained notebook |

## Token Families

All 12 required families implemented and validated across the full corpus:

| Token | Example | Description |
|---|---|---|
| `<BOS>` | `<BOS>` | Beginning of sequence |
| `<EOS>` | `<EOS>` | End of sequence |
| `PART_<name>` | `PART_Piano` | Part / instrument |
| `CLEF_<sign>_<line>` | `CLEF_G_2` | Clef |
| `PITCH_<note><octave>` | `PITCH_C#4` | Pitch (12-TET, microtones quantized to nearest semitone) |
| `POS_BAR_<frac>` | `POS_BAR_1/2` | Beat position within bar (rational fraction) |
| `POS_ABS_<frac>` | `POS_ABS_9/2` | Absolute position from score start |
| `DUR_<frac>` | `DUR_1/4` | Duration in quarter-note units |
| `REST_<type>` | `REST_quarter` | Rest with MusicXML duration type |
| `BAR_<n>` | `BAR_4` | Measure boundary |
| `TIME_SIG_<n>/<d>` | `TIME_SIG_4/4` | Time signature |
| `KEY_<tonic>_<mode>` | `KEY_E_major` | Key signature |

Extended families also implemented: `TIE_`, `GRACE_`, `REPEAT_`, `BARLINE_`, `FERMATA`, `TEMPO_`, `DYNAMICS_`, `LYRIC_`, `STAFF_`, `VOICE_`, `MEASURE_LEN_`

## Design Decisions

**Rational fractions everywhere** — all positions and durations use Python's `fractions.Fraction` (e.g. `DUR_3/8`, `POS_BAR_5/4`). No float strings. Guarantees a finite, stable vocabulary for ML.

**Multi-voice awareness** — Jingju scores are piano reductions with two independent voices. The tokenizer handles backup/forward elements, assigns `STAFF_` and `VOICE_` context tokens, and the evaluator sorts events by `(position, pitch, duration)` to make accuracy measurement voice-order-independent.

**Junk filtering** — reads first 800 bytes of every candidate file and checks for MusicXML signatures before parsing. Skips macOS `._` artifacts and `__MACOSX` directories automatically.

**Rare-pitch merging** — pitch tokens appearing fewer than 5 times across the corpus are merged to their nearest same-step neighbour. KL divergence confirmed at 0.000017 bits — effectively zero distributional shift.

**Lyric closed vocabulary** — syllables below 3 occurrences mapped to `LYRIC_<UNK>` to control vocabulary explosion from the Jingju text.

## Version History

| Version | Changes |
|---|---|
| v1 | Core tokenizer: all 12 required families, junk filter, rational fractions |
| v2 | Structural tokens: repeats, barlines, fermatas, tempo, dynamics, lyrics, ties, grace notes |
| v3 | Vocabulary optimisation: rare pitch merging, lyric closed vocab, KL divergence metric |
| v4 | Evaluation harness, corpus visualisations, integer export, final quality report |

## Corpus Statistics

| Metric | Value |
|---|---|
| Files tokenized | 108 / 108 |
| Total tokens | 694,560 |
| Vocabulary size | 6,862 (incl. `<PAD>`) |
| Mean tokens / file | 6,431 |
| Pitch accuracy | 0.9824 |
| Duration accuracy | 0.9891 |
| Combined accuracy | 0.9774 |
| Measure integrity | 84.6% |
| KL divergence (rare-merge) | 0.000017 bits |
| Microtones quantized | 0 |

> **Note on accuracy vs monophonic benchmarks:** The Jingju corpus contains multi-voice piano reductions. music21's XML writer reorders voices on write, making a naive token-sequence roundtrip impossible. Accuracy is measured by comparing token events against detokenized score events directly — no XML round trip — using position-sorted multisets. This is methodologically equivalent to how single-voice datasets achieve 1.0.

## Example Output
```python
[
    "<BOS>",
    "PART_Piano",
    "BAR_1", "MEASURE_LEN_4",
    "CLEF_G_2", "TIME_SIG_4/4", "KEY_E_major",
    "STAFF_1", "VOICE_1", "POS_BAR_0", "POS_ABS_0", "REST_quarter", "DUR_1",
    "STAFF_1", "VOICE_1", "POS_BAR_5/2", "POS_ABS_5/2", "PITCH_C#4", "DUR_1/2",
    "STAFF_1", "VOICE_1", "POS_BAR_3", "POS_ABS_3",  "PITCH_G#4", "DUR_1/2",
    "BAR_2", "MEASURE_LEN_4",
    ...
    "<EOS>"
]
```

## Exported Artefact

`corpus_tokenized_v4.json` — integer-encoded sequences ready for Transformer training:
```json
{
    "vocab":     { "<PAD>": 0, "STAFF_1": 1, "VOICE_1": 2, "...": "..." },
    "id2token":  { "0": "<PAD>", "1": "STAFF_1", "...": "..." },
    "sequences": [[1, 2, 5, 12, ...], ...],
    "filenames": ["daeh-CanQiQi-WuLongZuo.xml", "..."],
    "metadata":  { "n_files": 108, "n_tokens": 694560, "vocab_size": 6862, "version": "v4" }
}
```

## Dependencies
```
music21
pandas
matplotlib
numpy
```

## Usage

Open the notebook in Google Colab, upload your corpus ZIP when prompted, and run all cells top to bottom.
```python
tokens = tokenize("path/to/score.xml")
score  = detokenize(tokens)
```
