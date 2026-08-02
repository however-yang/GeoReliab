from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path

import pytest

import georeliab_mve.v4_overlay_resource_resolution as resolution


SAMPLE_SHA = hashlib.sha256(b'samples').hexdigest()
POINTS_SHA = hashlib.sha256(b'points').hexdigest()


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    worktree = tmp_path / 'worktree'
    runtime = tmp_path / 'runtime'
    data = runtime / 'data'
    cache = runtime / 'cache'
    manifests = runtime / 'manifests'
    for directory in (worktree / 'configs', data, cache, manifests):
        directory.mkdir(parents=True, exist_ok=True)
    overlay = worktree / 'configs' / 'a100_real_mve_overlay.toml'
    overlay.write_text(
        "[runtime]\n"
        "root = '/runtime'\n"
        "data = '/runtime/data'\n"
        "cache = '/runtime/cache'\n"
        "[resources]\n"
        "dtu_sampleset_url = 'https://example.test/SampleSet.zip'\n"
        f"dtu_sampleset_bytes = {len(b'samples')}\n"
        f"dtu_sampleset_sha256 = '{SAMPLE_SHA}'\n"
        "dtu_points_url = 'https://example.test/Points.zip'\n"
        f"dtu_points_bytes = {len(b'points')}\n"
        f"dtu_points_sha256 = '{POINTS_SHA}'\n",
        encoding='utf-8',
        newline='\n',
    )
    (data / 'SampleSet.zip').write_bytes(b'samples')
    (data / 'Points.zip').write_bytes(b'points')
    frozen = manifests / 'frozen_materialization.json'
    frozen.write_text(
        json.dumps(
            {
                'archives': {
                    'SampleSet.zip': {
                        'path': '/obsolete/cache/SampleSet.zip',
                        'bytes': len(b'samples'),
                        'sha256': SAMPLE_SHA,
                    },
                    'Points.zip': {
                        'path': '/obsolete/cache/Points.zip',
                        'bytes': len(b'points'),
                        'sha256': POINTS_SHA,
                    },
                }
            }
        ),
        encoding='utf-8',
    )
    return {
        'worktree': worktree,
        'runtime': runtime,
        'data': data,
        'cache': cache,
        'overlay': overlay,
        'frozen': frozen,
    }


def _resolve(paths: dict[str, Path], **kwargs: object) -> dict[str, object]:
    anchor = kwargs.pop('anchor_overlay_bytes', paths['overlay'].read_bytes())
    return resolution.resolve_overlay_resource_identities(
        worktree=paths['worktree'],
        runtime_root=paths['runtime'],
        overlay_path=paths['overlay'],
        frozen_materialization_path=paths['frozen'],
        host_system='Linux',
        anchor_overlay_bytes=anchor,
        _posix_path_mapper=lambda value: paths['runtime']
        / Path(value).relative_to('/runtime'),
        **kwargs,
    )


@pytest.mark.parametrize('shadow', [None, b'samples', b'stale'])
def test_overlay_data_is_authoritative_over_cache(
    tmp_path: Path, shadow: bytes | None
) -> None:
    paths = _write_fixture(tmp_path)
    if shadow is not None:
        (paths['cache'] / 'SampleSet.zip').write_bytes(shadow)

    payload = _resolve(paths)

    sample = payload['resources']['dtu_sampleset']
    assert sample['realpath'] == str(
        (paths['data'] / 'SampleSet.zip').resolve()
    )
    assert sample['observed_sha256'] == SAMPLE_SHA
    assert sample['shadow_candidate']['exists'] is (shadow is not None)
    assert sample['shadow_candidate']['status'] == 'IGNORED_SHADOW'


@pytest.mark.parametrize('mode', ['missing', 'wrong-hash'])
def test_cache_never_falls_back_for_bad_data(
    tmp_path: Path, mode: str
) -> None:
    paths = _write_fixture(tmp_path)
    target = paths['data'] / 'SampleSet.zip'
    if mode == 'missing':
        target.unlink()
    else:
        target.write_bytes(b'wrong')
    (paths['cache'] / 'SampleSet.zip').write_bytes(b'samples')

    with pytest.raises(resolution.OverlayResolutionError):
        _resolve(paths)


def test_override_is_only_an_equality_assertion(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    payload = _resolve(
        paths,
        resource_overrides={
            'dtu_sampleset': paths['data'] / 'SampleSet.zip'
        },
    )
    assert payload['resources']['dtu_sampleset']['override_asserted'] is True
    with pytest.raises(
        resolution.OverlayResolutionError,
        match='V4_OVERLAY_RESOURCE_OVERRIDE_MISMATCH',
    ):
        _resolve(
            paths,
            resource_overrides={
                'dtu_sampleset': paths['cache'] / 'SampleSet.zip'
            },
        )


def test_environment_override_must_match_overlay(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    with pytest.raises(
        resolution.OverlayResolutionError,
        match='V4_OVERLAY_RESOURCE_OVERRIDE_MISMATCH',
    ):
        _resolve(
            paths,
            environment={
                'GEORELIAB_V4_DTU_SAMPLESET_PATH': str(
                    paths['cache'] / 'SampleSet.zip'
                )
            },
        )


def test_overlay_anchor_and_materialization_are_frozen(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    with pytest.raises(
        resolution.OverlayResolutionError,
        match='V4_OVERLAY_DESCRIPTOR_SHA_MISMATCH',
    ):
        _resolve(paths, anchor_overlay_bytes=b'different')
    frozen = json.loads(paths['frozen'].read_text(encoding='utf-8'))
    frozen['archives']['Points.zip']['sha256'] = '0' * 64
    paths['frozen'].write_text(json.dumps(frozen), encoding='utf-8')
    with pytest.raises(
        resolution.OverlayResolutionError,
        match='V4_OVERLAY_MATERIALIZATION_IDENTITY_MISMATCH',
    ):
        _resolve(paths)


@pytest.mark.parametrize(
    'mutation',
    ['missing', 'case-drift', 'duplicate-case', 'wrong-basename', 'traversal'],
)
def test_invalid_logical_resource_mapping_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    paths = _write_fixture(tmp_path)
    text = paths['overlay'].read_text(encoding='utf-8')
    if mutation == 'missing':
        text = text.replace(
            "dtu_points_url = 'https://example.test/Points.zip'\n", ''
        )
    elif mutation == 'case-drift':
        text = text.replace('dtu_points_url', 'DTU_POINTS_URL')
    elif mutation == 'duplicate-case':
        text += "DTU_SAMPLESET_URL = 'https://example.test/SampleSet.zip'\n"
    elif mutation == 'wrong-basename':
        text = text.replace('Points.zip', 'points.zip')
    else:
        text = text.replace(
            'https://example.test/Points.zip',
            'https://example.test/../Points.zip',
        )
    paths['overlay'].write_text(text, encoding='utf-8')

    with pytest.raises(resolution.OverlayResolutionError):
        _resolve(paths)


def test_posix_overlay_is_rejected_on_windows(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    with pytest.raises(
        resolution.OverlayResolutionError,
        match='V4_OVERLAY_HOST_PATH_FLAVOR_MISMATCH',
    ):
        resolution.resolve_overlay_resource_identities(
            worktree=paths['worktree'],
            runtime_root=paths['runtime'],
            overlay_path=paths['overlay'],
            frozen_materialization_path=paths['frozen'],
            host_system='Windows',
            anchor_overlay_bytes=paths['overlay'].read_bytes(),
        )


@pytest.mark.skipif(os.name == 'nt', reason='symlink creation needs POSIX')
def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    outside = tmp_path / 'outside.zip'
    outside.write_bytes(b'samples')
    target = paths['data'] / 'SampleSet.zip'
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(
        resolution.OverlayResolutionError,
        match='V4_OVERLAY_RESOURCE_ROOT_ESCAPE',
    ):
        _resolve(paths)


def test_stat_hash_change_is_detected(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)

    def mutate(path: Path) -> None:
        if path.name == 'SampleSet.zip':
            path.write_bytes(b'changed')

    with pytest.raises(
        resolution.OverlayResolutionError,
        match='V4_OVERLAY_RESOURCE_CHANGED_DURING_HASH',
    ):
        _resolve(paths, before_hash=mutate)


def test_signed_artifacts_round_trip_and_tamper_fails(
    tmp_path: Path
) -> None:
    paths = _write_fixture(tmp_path)
    output = paths['runtime'] / 'artifacts' / (
        'v4-overlay-resource-resolution/' + 'a' * 40
    )
    receipt = resolution.resolve_overlay_resources(
        worktree=paths['worktree'],
        runtime_root=paths['runtime'],
        overlay_path=paths['overlay'],
        frozen_materialization_path=paths['frozen'],
        output_dir=output,
        host_system='Linux',
        anchor_overlay_bytes=paths['overlay'].read_bytes(),
        tooling_commit='a' * 40,
        tooling_tree='b' * 40,
        _posix_path_mapper=lambda value: paths['runtime']
        / Path(value).relative_to('/runtime'),
    )
    assert receipt['status'] == 'PASS'
    assert resolution.validate_overlay_resource_resolution(
        output / 'resource-resolution-receipt.json',
        host_system='Linux',
    )['status'] == 'PASS'
    identities = output / 'resolved-resource-identities.json'
    payload = json.loads(identities.read_text(encoding='utf-8'))
    payload['resources']['dtu_points']['observed_sha256'] = '0' * 64
    identities.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(resolution.OverlayResolutionError):
        resolution.validate_overlay_resource_resolution(
            output / 'resource-resolution-receipt.json',
            host_system='Linux',
        )


def test_failed_resolution_publishes_no_pass_artifacts(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    (paths['data'] / 'Points.zip').unlink()
    output = paths['runtime'] / 'artifacts' / (
        'v4-overlay-resource-resolution/' + 'a' * 40
    )
    with pytest.raises(resolution.OverlayResolutionError):
        resolution.resolve_overlay_resources(
            worktree=paths['worktree'],
            runtime_root=paths['runtime'],
            overlay_path=paths['overlay'],
            frozen_materialization_path=paths['frozen'],
            output_dir=output,
            host_system='Linux',
            anchor_overlay_bytes=paths['overlay'].read_bytes(),
            tooling_commit='a' * 40,
            tooling_tree='b' * 40,
            _posix_path_mapper=lambda value: paths['runtime']
            / Path(value).relative_to('/runtime'),
        )
    assert not output.exists()


def test_signed_semantically_invalid_staging_is_never_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_fixture(tmp_path)
    output = paths['runtime'] / 'artifacts' / (
        'v4-overlay-resource-resolution/' + 'a' * 40
    )
    original_signed = resolution._signed

    def sign_with_invalid_receipt(
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        candidate = dict(payload)
        if candidate.get('schema_version') == resolution.RECEIPT_SCHEMA:
            candidate['resolved_resource_count'] = 3
        return original_signed(candidate)

    monkeypatch.setattr(resolution, '_signed', sign_with_invalid_receipt)
    with pytest.raises(
        resolution.OverlayResolutionError,
        match='V4_OVERLAY_RECEIPT_INVALID',
    ):
        resolution.resolve_overlay_resources(
            worktree=paths['worktree'],
            runtime_root=paths['runtime'],
            overlay_path=paths['overlay'],
            frozen_materialization_path=paths['frozen'],
            output_dir=output,
            host_system='Linux',
            anchor_overlay_bytes=paths['overlay'].read_bytes(),
            tooling_commit='a' * 40,
            tooling_tree='b' * 40,
            _posix_path_mapper=lambda value: paths['runtime']
            / Path(value).relative_to('/runtime'),
        )
    assert not output.exists()
    assert not output.with_name(output.name + '.partial').exists()
