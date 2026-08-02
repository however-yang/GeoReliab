from __future__ import annotations

import json

import georeliab_mve.cli as cli
from georeliab_mve.cli import main


def test_attempt03_cli_argument_errors_are_structured(
    capsys,
) -> None:
    code = main(['v4-attempt03-gpu-preflight'])
    captured = capsys.readouterr()

    assert code == 2
    assert captured.err == ''
    payload = json.loads(captured.out)
    assert payload['status'] == 'FAIL'
    assert payload['reason_code'] == 'V4_ATTEMPT03_CLI_ARGUMENT_ERROR'
    assert payload['error_type'] == 'ArgumentParserError'


def test_attempt03_resource_blocker_returns_nonzero(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(
        cli,
        'revalidate_attempt03_resources',
        lambda **_kwargs: {
            'status': 'FAIL',
            'reason_code': 'V4_RESOURCE_CLOSURE_REVALIDATION_FAILED',
        },
    )

    code = main(
        [
            'v4-attempt03-revalidate-resources',
            '--worktree',
            str(tmp_path),
            '--runtime-root',
            str(tmp_path),
            '--rectified-root',
            str(tmp_path),
            '--closure-root',
            str(tmp_path),
            '--overlay',
            str(tmp_path / 'overlay.toml'),
            '--output',
            str(
                tmp_path
                / 'authorization-attempts'
                / 'attempt-03'
                / 'resource.json'
            ),
        ]
    )

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload['reason_code'] == (
        'V4_RESOURCE_CLOSURE_REVALIDATION_FAILED'
    )
