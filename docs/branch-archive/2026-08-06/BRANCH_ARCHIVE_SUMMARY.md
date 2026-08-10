# GeoReliab v4 Local Branch Archive

Archive date: 2026-08-06  
Archive phase: cleanup completed with one fail-closed retained detached Worktree

## Canonical entrypoint

`codex/v4-canonical` points exactly to commit `a9c687c73eca6bc3a26659b3e0247c227e7c5238` and tree `c8c47baff7deb082e502db83f8fc8c942f5d3e3d`.

Authoritative Gate 1 evidence is Linux/A100 (`gate1-cpu-a7875354-13`): 147/147 cases passed, `V4_RECOVERY_CPU_FAULT_MATRIX_PASS`, `V4_RECOVERY_RUNTIME_READY`, and `NO_SCIENTIFIC_RESULT`. Windows is recorded only as `WINDOWS_PORTABILITY_BLOCKED_UNSUPPORTED_SIGNAL` because `signal.SIGHUP` is unavailable there.

## Final local branches

Only `main` and `codex/v4-canonical` remain as local branches. The complete manifest contains 13 original local-branch records, including these two retained refs and the 11 deleted historical refs.

| Local branch | Commit | Tree | Zero-drift | Archive tag | Status |
| --- | --- | --- | --- | --- | --- |
| `main` | `a1b324f` | `fc6530c` | no | — | retained |
| `codex/v4-canonical` | `a9c687c` | `c8c47ba` | yes | — | retained |
| `codex/p2-paired-governance-repair` | `cf6f1f2` | `df9cde1` | no | `cf6f1f2` | deleted |
| `codex/storage-architecture-refactor` | `e6c7c8e` | `391fa7a` | no | `e6c7c8e` | deleted |
| `codex/v4-gpu-selection-attempt-02` | `8b80f40` | `3529766` | yes | `8b80f40` | deleted |
| `codex/v4-gpu-selection-attempt-03` | `e6bc12f` | `f133fde` | yes | `e6bc12f` | deleted |
| `codex/v4-gpu-selection-attempt-04` | `b7faf49` | `0a934f7` | yes | `b7faf49` | deleted |
| `codex/v4-gpu-selection-authorization` | `3785a38` | `e83b2d9` | yes | `3785a38` | deleted |
| `codex/v4-mve-execution-attempt-05` | `0f4fd14` | `79918d4` | yes | `0f4fd14` | deleted |
| `codex/v4-overlay-resource-resolution-fix` | `6e91a41` | `2f9b264` | yes | `6e91a41` | deleted |
| `codex/v4-ranking-warning` | `7381e60` | `f4e2b11` | yes | `7381e60` | deleted |
| `codex/v4-rectified-member-closure` | `580aeb9` | `883c3d7` | yes | `580aeb9` | deleted |
| `real-georeliab-mve` | `f5397b2` | `6444936` | no | `f5397b2` | deleted |

## Remote refs

The direct read-only `git ls-remote --heads a100` snapshot contains 30 remote heads. The local remote-tracking namespace had only 14 cached refs, so the complete 30-ref snapshot is recorded in `BRANCH_ARCHIVE_MANIFEST.json`; no remote ref was written, updated, pushed, or deleted.

## Worktree result

- 12 clean historical Worktree records were removed, followed by `git worktree prune`.
- The detached Worktree at unique commit `b2f6831` remains fail-closed because status enumeration encountered permission-denied directories; its archive tag is verified and no files were deleted.
- The retained main Worktree has 87 pre-existing untracked entries and is preserved as user state.
- The empty temporary directory `C:/tmp/georeliab-v4-canonical-verify` has no Git Worktree record and is marked `FILESYSTEM_TOMBSTONE_PENDING_HANDLE_RELEASE`.

## Archive artifacts

- `BRANCH_ARCHIVE_MANIFEST.json`: complete local branch, direct remote-head, detached Worktree, tag, and cleanup inventory.
- `BRANCH_ARCHIVE_MANIFEST_LOCAL_TRACKING.json`: preserved original local-tracking inventory.
- `BRANCH_ARCHIVE_CLEANUP_RESULT.json`: final cleanup result sidecar.

## Verification markers

`CANONICAL_V4_BRANCH_READY`  
`LOCAL_BRANCH_ARCHIVE_PASS`  
`LOCAL_WORKTREE_CLEANUP_PARTIAL_FAIL_CLOSED`  
`REMOTE_BRANCHES_UNCHANGED`  
`NO_SCIENTIFIC_RESULT`
