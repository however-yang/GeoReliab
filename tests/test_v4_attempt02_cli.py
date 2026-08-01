from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from georeliab_mve import cli
from georeliab_mve.v4_execution import V4ExecutionError


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _history_paths() -> list[Path]:
    return [
        (SOURCE_ROOT / relative).resolve()
        for relative in cli.ATTEMPT02_REQUIRED_HISTORY_PATHS
    ]


def _history_args(paths: list[Path] | None = None) -> list[str]:
    result: list[str] = []
    for path in _history_paths() if paths is None else paths:
        result.extend(['--historical-evidence', str(path)])
    return result


@pytest.mark.parametrize(
    ('command', 'arguments'),
    [
        (
            'v4-attempt02-prepare-resources',
            ['--root', 'r', '--source-manifest', 's', '--output', 'o'],
        ),
        (
            'v4-attempt02-gpu-preflight',
            [
                '--output',
                'o',
                '--resource-snapshot',
                's',
                '--historical-evidence',
                'h',
            ],
        ),
        (
            'v4-attempt02-create-execution-authorization',
            [
                '--root',
                'r',
                '--receipt',
                'p',
                '--resource-snapshot',
                's',
                '--run-root',
                'run',
                '--artifact-root',
                'artifacts',
                '--final-evidence-path',
                'final',
                '--output',
                'o',
            ],
        ),
        (
            'v4-attempt02-validate-execution-authorization',
            ['authorization.json'],
        ),
    ],
)
def test_attempt02_parser_exposes_exact_public_commands(
    command: str, arguments: list[str]
) -> None:
    parsed = cli.build_parser().parse_args([command, *arguments])
    assert parsed.command == command


def test_attempt02_prepare_resources_dispatches_exact_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def materialize(**kwargs):
        captured.update(kwargs)
        return {'status': 'PASS', 'resource_snapshot_path': 'snapshot.json'}

    monkeypatch.setattr(cli, 'materialize_attempt_resources', materialize)
    root = tmp_path / 'attempt-02'
    source = root / 'source.json'
    output = root / 'artifacts' / 'snapshot.json'
    code = cli.main(
        [
            'v4-attempt02-prepare-resources',
            '--root',
            str(root),
            '--source-manifest',
            str(source),
            '--output',
            str(output),
        ]
    )

    assert code == 0
    assert captured == {
        'root': root,
        'source_manifest_path': source,
        'output_path': output,
    }
    assert json.loads(capsys.readouterr().out)['status'] == 'PASS'


def test_attempt02_preflight_validates_resources_and_complete_history_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}
    resource = tmp_path / 'attempt-02' / 'resource.json'
    output = tmp_path / 'attempt-02' / 'preflight.json'
    monkeypatch.setattr(
        cli,
        'validate_attempt_resources',
        lambda path: {'schedule_sha256': 'a' * 64, 'validated_path': str(path)},
    )

    def preflight(**kwargs):
        captured.update(kwargs)
        return {'status': 'PASS', 'reason_code': 'PASS'}

    monkeypatch.setattr(cli, 'create_attempt_hardware_preflight', preflight)
    code = cli.main(
        [
            'v4-attempt02-gpu-preflight',
            '--output',
            str(output),
            '--resource-snapshot',
            str(resource),
            *_history_args(),
        ]
    )

    assert code == 0
    assert captured == {
        'output_path': output,
        'schedule_sha256': 'a' * 64,
        'historical_evidence_paths': tuple(_history_paths()),
    }
    assert json.loads(capsys.readouterr().out)['status'] == 'PASS'


def test_attempt02_preflight_requires_all_canonical_and_original_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        'create_attempt_hardware_preflight',
        lambda **kwargs: pytest.fail('preflight must not run'),
    )
    code = cli.main(
        [
            'v4-attempt02-gpu-preflight',
            '--output',
            str(tmp_path / 'attempt-02' / 'preflight.json'),
            '--resource-snapshot',
            str(tmp_path / 'attempt-02' / 'resource.json'),
            *_history_args(_history_paths()[:-1]),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload['status'] == 'FAIL'
    assert payload['reason_code'].startswith(
        'V4_ATTEMPT01_CANONICAL_HISTORY_REQUIRED'
    )


def test_attempt02_preflight_missing_history_argument_is_json_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        [
            'v4-attempt02-gpu-preflight',
            '--output',
            str(tmp_path / 'attempt-02' / 'preflight.json'),
            '--resource-snapshot',
            str(tmp_path / 'attempt-02' / 'resource.json'),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload == {
        'status': 'FAIL',
        'reason_code': 'V4_ATTEMPT02_CLI_ARGUMENT_ERROR',
        'error_type': 'ArgumentParserError',
    }


def test_attempt02_public_module_cli_argument_fail_is_json_exit_2(
    tmp_path: Path,
) -> None:
    output = tmp_path / 'attempt-02' / 'preflight.json'
    completed = subprocess.run(
        [
            sys.executable,
            '-m',
            'georeliab_mve',
            'v4-attempt02-gpu-preflight',
            '--output',
            str(output),
            '--resource-snapshot',
            str(tmp_path / 'attempt-02' / 'resource.json'),
        ],
        cwd=SOURCE_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)['status'] == 'FAIL'
    assert completed.stderr == ''
    assert not output.exists()


def test_attempt02_preflight_rejects_non_attempt02_output_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        'validate_attempt_resources',
        lambda path: {'schedule_sha256': 'a' * 64},
    )
    code = cli.main(
        [
            'v4-attempt02-gpu-preflight',
            '--output',
            str(tmp_path / 'preflight.json'),
            '--resource-snapshot',
            str(tmp_path / 'attempt-02' / 'resource.json'),
            *_history_args(),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload['reason_code'] == 'V4_ATTEMPT_PATH_MISMATCH'
    assert not (tmp_path / 'preflight.json').exists()


def test_attempt02_preflight_fail_is_json_exit_2_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        'validate_attempt_resources',
        lambda path: {'schedule_sha256': 'a' * 64},
    )
    calls: list[dict[str, object]] = []

    def fail(**kwargs):
        calls.append(kwargs)
        return {'status': 'FAIL', 'reason_code': 'V4_NO_ELIGIBLE_IDLE_GPU'}

    monkeypatch.setattr(cli, 'create_attempt_hardware_preflight', fail)
    code = cli.main(
        [
            'v4-attempt02-gpu-preflight',
            '--output',
            str(tmp_path / 'attempt-02' / 'preflight.json'),
            '--resource-snapshot',
            str(tmp_path / 'attempt-02' / 'resource.json'),
            *_history_args(),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload['reason_code'] == 'V4_NO_ELIGIBLE_IDLE_GPU'
    assert len(calls) == 1


def test_attempt02_create_authorization_dispatches_exact_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def authorize(**kwargs):
        captured.update(kwargs)
        return {'status': 'PASS', 'authorization_path': 'authorization.json'}

    monkeypatch.setattr(cli, 'create_attempt_execution_authorization', authorize)
    root = tmp_path / 'attempt-02'
    values = {
        'receipt': root / 'receipt.json',
        'resource_snapshot': root / 'resource.json',
        'run_root': root / 'runs',
        'artifact_root': root / 'artifacts',
        'final_evidence_path': root / 'final.json',
        'output': root / 'authorization.json',
    }
    code = cli.main(
        [
            'v4-attempt02-create-execution-authorization',
            '--root',
            str(root),
            '--receipt',
            str(values['receipt']),
            '--resource-snapshot',
            str(values['resource_snapshot']),
            '--run-root',
            str(values['run_root']),
            '--artifact-root',
            str(values['artifact_root']),
            '--final-evidence-path',
            str(values['final_evidence_path']),
            '--output',
            str(values['output']),
        ]
    )

    assert code == 0
    assert captured == {
        'root': root,
        'receipt_path': values['receipt'],
        'resource_snapshot_path': values['resource_snapshot'],
        'run_root': values['run_root'],
        'artifact_root': values['artifact_root'],
        'final_evidence_path': values['final_evidence_path'],
        'output_path': values['output'],
    }
    assert json.loads(capsys.readouterr().out)['status'] == 'PASS'


def test_attempt02_validate_authorization_is_json_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    authorization = tmp_path / 'attempt-02' / 'authorization.json'
    monkeypatch.setattr(
        cli,
        'validate_attempt_execution_authorization',
        lambda path: {'attempt_id': 'attempt-02', 'path': str(path)},
    )
    assert (
        cli.main(
            [
                'v4-attempt02-validate-execution-authorization',
                str(authorization),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)['attempt_id'] == 'attempt-02'

    monkeypatch.setattr(
        cli,
        'validate_attempt_execution_authorization',
        lambda path: (_ for _ in ()).throw(
            V4ExecutionError('V4_AUTHORIZATION_TAMPER')
        ),
    )
    assert (
        cli.main(
            [
                'v4-attempt02-validate-execution-authorization',
                str(authorization),
            ]
        )
        == 2
    )
    failure = json.loads(capsys.readouterr().out)
    assert failure['status'] == 'FAIL'
    assert failure['reason_code'] == 'V4_AUTHORIZATION_TAMPER'


def test_legacy_v4_preflight_dispatch_remains_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def legacy(**kwargs):
        captured.update(kwargs)
        return {'status': 'FAIL', 'reason_code': 'legacy'}

    monkeypatch.setattr(cli, 'create_hardware_preflight', legacy)
    monkeypatch.setattr(
        cli,
        'create_attempt_hardware_preflight',
        lambda **kwargs: pytest.fail('attempt-02 preflight must not run'),
    )
    output = tmp_path / 'legacy.json'
    code = cli.main(
        [
            'v4-gpu-preflight',
            '--output',
            str(output),
            '--requested-index',
            '1',
        ]
    )

    assert code == 2
    assert captured['output_path'] == output
    assert captured['requested_physical_index'] == 1
    assert json.loads(capsys.readouterr().out)['reason_code'] == 'legacy'


def test_legacy_v4_create_authorization_dispatch_remains_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def legacy(**kwargs):
        captured.update(kwargs)
        return {'legacy': True}

    monkeypatch.setattr(cli, 'create_execution_authorization', legacy)
    monkeypatch.setattr(
        cli,
        'create_attempt_execution_authorization',
        lambda **kwargs: pytest.fail('attempt-02 authorization must not run'),
    )
    code = cli.main(
        [
            'v4-create-execution-authorization',
            '--root',
            str(tmp_path),
            '--receipt',
            'receipt.json',
            '--resource-inventory',
            'inventory.json',
            '--run-root',
            'runs',
            '--artifact-root',
            'artifacts',
            '--final-evidence-path',
            'final.json',
            '--output',
            'authorization.json',
        ]
    )

    assert code == 0
    assert captured['resource_inventory_path'] == Path('inventory.json')
    assert json.loads(capsys.readouterr().out) == {'legacy': True}


def test_legacy_v4_validate_authorization_dispatch_remains_unchanged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        'validate_execution_authorization',
        lambda path: {'legacy_path': str(path)},
    )
    monkeypatch.setattr(
        cli,
        'validate_attempt_execution_authorization',
        lambda path: pytest.fail('attempt-02 validation must not run'),
    )
    code = cli.main(
        ['v4-validate-execution-authorization', 'authorization.json']
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        'legacy_path': 'authorization.json'
    }
