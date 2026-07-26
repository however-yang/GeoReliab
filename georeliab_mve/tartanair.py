'''Official-API and bounded-range acquisition helpers for TartanAir V2.

The public API's download operation is whole-archive granular.  P000-only
preparation therefore uses the official archive URL with ZIP central-directory
and HTTP Range validation; it never accepts fictional trajectory kwargs.
'''

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
from pathlib import Path
import re
from typing import Any

from .preparation import PreparationError


P000_FRAME_COUNT = 3537
_IMAGE_PREFIX = 'GreatMarsh/Data_easy/P000/image_lcam_front/'
_DEPTH_PREFIX = 'GreatMarsh/Data_easy/P000/depth_lcam_front/'
_FRAME_RE = re.compile(r'^(?P<frame>.+)_lcam_front(?:_depth)?\.(?:png|npy)$')


@dataclass(frozen=True)
class TartanAirSelection:
    image_members: tuple[str, ...]
    depth_members: tuple[str, ...]


def download_tartanair_whole_archive(destination: Path, *, dry_run: bool = False) -> Path:
    '''Use the documented TartanAir API and make its whole-archive scope explicit.'''

    if dry_run:
        return destination
    try:
        api = importlib.import_module('tartanair')
    except ImportError as exc:
        raise PreparationError('official tartanair Python API is required only for full acquisition') from exc
    initialize = getattr(api, 'init', None)
    download = getattr(api, 'download', None)
    if not callable(initialize) or not callable(download):
        raise PreparationError('official tartanair API requires init and download')
    destination.mkdir(parents=True, exist_ok=True)
    initialize(str(destination))
    # The official API has no trajectory selector: do not add trajectory_id or
    # target_path kwargs. P000-bounded preparation must use range extraction.
    download(
        env='GreatMarsh',
        difficulty='easy',
        modality=['image', 'depth'],
        camera_name='lcam_front',
        unzip=True,
        delete_zip=False,
    )
    return destination


def _frames(members: Sequence[str], prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for member in members:
        if not member.startswith(prefix):
            continue
        match = _FRAME_RE.match(member.removeprefix(prefix))
        if match is None:
            continue
        frame = match.group('frame')
        if frame in result:
            raise PreparationError(f'duplicate TartanAir P000 frame: {frame}')
        result[frame] = member
    return result


def select_uniform_aligned_p000_members(
    image_members: Sequence[str], depth_members: Sequence[str], *, count: int = 100
) -> TartanAirSelection:
    '''Fail closed on a non-identical 3,537-frame official P000 intersection.'''

    if count != 100:
        raise PreparationError('frozen TartanAir sanity requires exactly 100 frames')
    images = _frames(image_members, _IMAGE_PREFIX)
    depths = _frames(depth_members, _DEPTH_PREFIX)
    if len(images) != P000_FRAME_COUNT or len(depths) != P000_FRAME_COUNT:
        raise PreparationError('TartanAir P000 archives must contain exactly 3,537 aligned image/depth frames each')
    if set(images) != set(depths):
        raise PreparationError('TartanAir image/depth P000 central-directory members are misaligned')
    ordered = sorted(images, key=lambda frame: (int(frame) if frame.isdigit() else frame))
    indices = tuple(round(index * (len(ordered) - 1) / (count - 1)) for index in range(count))
    selected = [ordered[index] for index in indices]
    return TartanAirSelection(
        image_members=tuple(images[frame] for frame in selected),
        depth_members=tuple(depths[frame] for frame in selected),
    )
