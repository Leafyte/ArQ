# Freeze-date correction

On 2026-10-28 UTC, the frozen lock was written with `freeze_date_utc` set to `2026-10-29`. The environment's local date was 2026-10-29 in Asia/Calcutta, and that local date was mistakenly used for the UTC field. The host UTC clock confirms the freeze/evaluation activity occurred on 2026-10-28 UTC. This note corrects the interpretation only; no locked protocol, lock, manifest, runner, or result file was edited. The original field remains preserved as written.
