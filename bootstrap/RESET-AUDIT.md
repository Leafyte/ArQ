# Reset and Repository Audit

Audit date: 2026-10-28 (local working date, Asia/Kolkata). The active project brief is preserved verbatim at `sources/user-brief.md`. This audit records the pre-reset repository state and the completed reversible reset.

## Before reset: verified inventory

- The workspace root had no `.git` directory and `git rev-parse --show-toplevel` failed there. The separate old `Q-Charge/` directory contained its own nested Git repository; it was clean on `main` at commit `a74278b`. That repository's `.git` data is preserved inside the archive.
- No root `AGENTS.md`, `README.md`, project environment manifest, source package, application, solver, QML model, optimizer, or frontend was found.
- 21 original top-level entries were inventoried: 4,866 files totaling 538,565,763 bytes. This includes the 23,985,737-byte M4A recording and `research-tools/` with 4,775 files (505,953,814 bytes), mostly a cached Python model/dependency environment.
- The active official QML-01 problem PDF and a historical team-submitted QML-02 PDF were identified before reset. Source copies were SHA-256 compared with the archived originals after the move. `bootstrap/archive-manifest.json` records every archived file's relative path, size, and SHA-256, along with counts and source-copy checks.

## Existing content and implementation audit

- `Q-Charge/README.md`, `main.tex`, `prompt.md`, root `Q-CHARGE-*` documents and the `PS1-*` strategy/review/product artifacts describe an EV charging/QML-02 direction. The README explicitly labels itself “planning / early development.” These materials contain formulations, proposals, reviews, and test plans; no runnable charging solver, QUBO/QAOA, trained predictor, benchmark outputs, or application implementation was found.
- `preparation/` is a prior event-preparation pack. It contains prose, briefing HTML/build helpers, verification scripts, and two copied scientific-reading skills. The scripts verify/build that pack; they are not fleet-dispatch components. The old skill copies remain under the archive as historical context.
- `research-tools/transcribe.py` is an audio-to-text helper using a bundled `faster-whisper` package/model cache and writes a transcript at the workspace root. It is unrelated to the new algorithm. No transcript was present in the inventory.
- The root research documents include QML-02-specific planning and adversarial reviews. They are archived historical evidence, not active QML-01 instructions.
- No root `AGENTS.md` existed before reset. Existing files do not override the active user brief. No credential values were intentionally read or copied.

## Official problem source and extraction

- `sources/QpiAI __ Q-BITS _ Industry Problem Statements & Judgement Rubrics.pdf` is the active official problem/rubric PDF. Existing `pdftotext.exe` extracted page 2 (QML-01 and QpiAI requirements) and page 8 (Section 8 QpiAI technology rubric) into `sources/problem-extract.txt`. The complete text extraction is at `sources/qpi-pdf-fulltext.txt`.
- Page 2 requires delay-risk-aware contract-logistics fleet dispatch with vehicle capacities, delivery windows, commitments, uncertain traffic/service times; deliver exact vehicle-to-route assignments, ordered stops, ETA/risk per commitment, and total financial exposure; evaluate SLA violations, coverage, cost, calibration, solver quality, and runtime. QML requires meaningful QpiAI QCloud Quantum and QpiAI Opt QUBO/Ising/HUBO.
- Page 8 assigns 20 QpiAI points: 5 meaningful product/domain integration, 5 QCloud technical correctness, 5 Opt technical correctness, and 5 working execution evidence/reproducibility. It says actual invocation, relevance, correctness, and measured integration matter.
- `sources/historical/QUANT-A-MAZE 3.O (DebugLeaf).pdf` is a prior team-submitted EV charging/QML-02 brief, not the current official QML-01 statement. It is retained only as historical context.

## Completed reversible archive

All 21 pre-existing top-level entries were moved (not deleted) into:

`_archive/pre-debugleaf-20261028-220317/`

The archive contains 4,866 files / 538,565,763 bytes. The counts and total bytes match the pre-move inventory. SHA-256 was calculated for every original file before moving and recalculated for every archived file; all 4,866 pre/post hashes matched. The matching per-file hashes are recorded in `bootstrap/archive-manifest.json`. Both retained PDF copies also match their archived originals byte-for-byte. New `sources/` and `bootstrap/` entries were excluded; `.git` would have been retained if it had existed at workspace root. The old nested `Q-Charge/.git` directory moved with its repository. No original entry was permanently deleted.

### Restore procedure

The archive is the rollback source; the workspace root has no Git history for rollback. The nested `Q-Charge` Git repository is separately preserved at clean commit `a74278b`.

To restore a top-level entry, use the exact archive directory above. First confirm the destination under the workspace root does not exist. If it exists, do not overwrite, merge, or remove it; leave the archived copy intact and resolve the conflict manually. For example:

```powershell
$root = 'C:\ommd3076\Proejcts\nmit-hackathon'
$archive = Join-Path $root '_archive\pre-debugleaf-20261028-220317'
$name = 'preparation' # repeat for one exact top-level entry at a time
$source = Join-Path $archive $name
$destination = Join-Path $root $name
if (Test-Path -LiteralPath $destination) { throw "Restore conflict: $destination already exists" }
$resolvedSource = (Resolve-Path -LiteralPath $source).Path
if (-not $resolvedSource.StartsWith($archive + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Archive source escaped archive root' }
$rootFull = [System.IO.Path]::GetFullPath($root).TrimEnd('\')
$destinationFull = [System.IO.Path]::GetFullPath($destination)
if (-not $destinationFull.StartsWith($rootFull + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Restore destination escaped workspace root' }
if (($destinationFull.Substring($rootFull.Length + 1) -split '[\\/]').Count -ne 1) { throw 'Restore must target one original top-level entry' }
Move-Item -LiteralPath $resolvedSource -Destination $destination
```

Restore only the entries required. The source PDFs in `sources/` are preserved copies and may remain in place; the archived originals are not changed by this procedure.

## After reset

The original project and research entries are now inactive in the archive. Current source references are under `sources/`; reset evidence and the file-level archive manifest are under `bootstrap/`. No algorithm, runtime, or frontend implementation was created as part of this reset.

