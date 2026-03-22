# Jingju Opera — Symbolic Music Tokenizer

**Course:** Analysis of Symbolic Music and Ethnomusicology  
**Dataset:** Jingju Opera Scores Corpus (108 MusicXML files)  
**Notebook:** runs top-to-bottom on Google Colab  

---

Converts MusicXML scores from the Jingju (Beijing opera) tradition into flat token sequences for use with machine learning models. The corpus is a collection of multi-voice piano reductions — a harder tokenization target than monophonic datasets, requiring explicit handling of backup/forward voice interleaving and staff-aware event ordering.

---

## Token Format

Positions and durations are encoded as **rational fractions** using Python's `fractions.Fraction` — never floats. This guarantees a finite, collision-free vocabulary regardless of tempo or time signature.
```
<BOS>
PART_Piano
BAR_1  MEASURE_LEN_4
CLEF_G_2  TIME_SIG_4/4  KEY_E_major
STAFF_1  VOICE_1  POS_BAR_0    POS_ABS_0    REST_quarter  DUR_1
STAFF_1  VOICE_1  POS_BAR_5/2  POS_ABS_5/2  PITCH_C#4     DUR_1/2
STAFF_1  VOICE_1  POS_BAR_3    POS_ABS_3    PITCH_G#4     DUR_1/2  TIE_START
BAR_2  MEASURE_LEN_4
STAFF_1  VOICE_1  POS_BAR_0    POS_ABS_4    PITCH_G#4     DUR_1/2  TIE_STOP
...
<EOS>
```

### Required families (all 12 present)

`<BOS>` `<EOS>` `PART_` `CLEF_` `TIME_SIG_` `KEY_` `PITCH_` `DUR_` `POS_BAR_` `POS_ABS_` `REST_` `BAR_`

### Extended families

`STAFF_` `VOICE_` `MEASURE_LEN_` `TIE_` `GRACE_` `REPEAT_` `BARLINE_` `FERMATA` `TEMPO_` `DYNAMICS_` `LYRIC_`

---

## How it works

**Junk filtering** reads the first 800 bytes of every candidate file and checks for MusicXML signatures before passing anything to the parser. macOS `._` artifacts and `__MACOSX` directories are silently skipped.

**Tokenization** walks each part measure-by-measure. Within a measure, a timeline of `(offset, priority, tokens)` tuples is built — structural tokens (clef, time, key) first, then notes sorted by onset. `flatten()` is called exactly once per measure to avoid the offset-reset bug.

**Detokenization** reconstructs a `music21.Score` using a state machine. Notes sharing the same `(staff, voice, pos, dur)` key are grouped into chords. The result is a playable score object.

**Evaluation** compares tokenized events against detokenized score events directly — no XML round trip. Events are sorted by `(position, pitch, duration)` making the comparison voice-order-independent, which is the correct approach for multi-voice scores where XML serialization may reorder voices.

**Vocabulary optimisation** merges rare pitch tokens (< 5 occurrences) to their nearest same-step neighbour. KL divergence between original and merged distributions: **0.000017 bits** — negligible. Rare lyric syllables (< 3 occurrences) collapse to `LYRIC_<UNK>`.

---

## Results

| Metric | Value |
|---|---|
| Files tokenized | 108 / 108 |
| Total tokens | 694,560 |
| Vocabulary | 6,862 tokens (incl. `<PAD>`) |
| Mean / min / max per file | 6,431 / 344 / 28,708 |
| Pitch accuracy | **0.9824** |
| Duration accuracy | **0.9891** |
| Combined accuracy | **0.9774** |
| Measure integrity | 84.6% |
| KL divergence | 0.000017 bits |

Accuracy is measured against the detokenized score object, not a re-tokenized XML file. The Jingju corpus is multi-voice piano — music21's XML writer reorders voices on serialization, making a naive byte-for-byte roundtrip impossible. Sorting events by `(position, pitch, duration)` before comparison removes this sensitivity while correctly measuring musical content preservation.

---

## Exported artefact

`corpus_tokenized_v4.json` — integer-encoded sequences ready for a Transformer:
```json
{
  "vocab":     { "<PAD>": 0, "STAFF_1": 1, "VOICE_1": 2 },
  "id2token":  { "0": "<PAD>", "1": "STAFF_1" },
  "sequences": [[1, 2, 5, 12], [1, 2, 7, 9]],
  "filenames": ["daeh-CanQiQi-WuLongZuo.xml"],
  "metadata":  {
    "n_files": 108, "n_tokens": 694560,
    "vocab_size": 6862, "version": "v4"
  }
}
```

---

## Dependencies

`music21` · `pandas` · `matplotlib` · `numpy`  
No installation needed on Google Colab — `!pip install music21` at the top of the notebook covers everything.

