from __future__ import annotations

import sys
import types

import pytest

from georeliab_mve.preparation import PreparationError
from georeliab_mve.tartanair import (
    P000_FRAME_COUNT,
    download_tartanair_whole_archive,
    select_uniform_aligned_p000_members,
)


def test_official_api_adapter_never_sends_imaginary_trajectory_kwargs(monkeypatch, tmp_path):
    calls: dict[str, object] = {}
    api = types.SimpleNamespace(
        init=lambda root: calls.setdefault('root', root),
        download=lambda **kwargs: calls.setdefault('download', kwargs),
    )
    monkeypatch.setitem(sys.modules, 'tartanair', api)
    assert download_tartanair_whole_archive(tmp_path) == tmp_path
    kwargs = calls['download']
    assert 'trajectory_id' not in kwargs
    assert 'target_path' not in kwargs
    assert kwargs['env'] == 'GreatMarsh'


def test_p000_selection_requires_complete_aligned_central_directories():
    image = [f'GreatMarsh/Data_easy/P000/image_lcam_front/{index:06d}_lcam_front.png' for index in range(P000_FRAME_COUNT)]
    depth = [f'GreatMarsh/Data_easy/P000/depth_lcam_front/{index:06d}_lcam_front_depth.png' for index in range(P000_FRAME_COUNT)]
    selected = select_uniform_aligned_p000_members(image, depth)
    assert len(selected.image_members) == len(selected.depth_members) == 100
    with pytest.raises(PreparationError, match='misaligned'):
        select_uniform_aligned_p000_members(image, depth[:-1] + [depth[-1].replace('003536', '999999')])
