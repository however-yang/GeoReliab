'''Strict parser for the verified official DTU extracted layout.'''
from __future__ import annotations

from pathlib import Path
import re
import numpy as np

from . import preparation as _base


def parse_dtu_inventory(root: Path) -> tuple[_base.DtuScene, ...]:
    rectified, points, masks = root / 'Rectified', root / 'Points' / 'stl', root / 'ObsMask'
    camera_dirs = [path for path in root.rglob('cal18') if path.is_dir()]
    if not rectified.is_dir() or not points.is_dir() or not masks.is_dir() or len(camera_dirs) != 1:
        raise _base.PreparationError('verified DTU inventory requires Rectified, Points/stl, ObsMask, and one cal18 camera directory')
    cameras = camera_dirs[0]
    centers: dict[int, np.ndarray] = {}
    for view in range(1, 50):
        path = cameras / f'pos_{view:03d}.txt'
        if not path.is_file():
            raise _base.PreparationError(f'missing official camera pos_{view:03d}.txt')
        matrix = np.loadtxt(path, dtype=np.float64)
        if matrix.size != 12:
            raise _base.PreparationError(f'official pos_{view:03d}.txt is not 3x4')
        matrix = matrix.reshape(3, 4)
        try:
            centers[view] = -np.linalg.solve(matrix[:, :3], matrix[:, 3])
        except np.linalg.LinAlgError as exc:
            raise _base.PreparationError(f'official pos_{view:03d}.txt has singular M') from exc
    result = []
    for directory in sorted(rectified.glob('scan*')):
        match = re.fullmatch(r'scan(\d+)', directory.name)
        if not match:
            continue
        scene = int(match.group(1))
        names = tuple(f'rect_{view:03d}_3_r5000.png' for view in range(1, 50))
        if not all((directory / name).is_file() for name in names):
            continue
        cloud, mask = points / f'stl{scene:03d}_total.ply', masks / f'ObsMask{scene}_10.mat'
        if cloud.is_file() and mask.is_file():
            result.append(_base.DtuScene(scene, names, dict(centers), str(cloud), str(mask)))
    if not result:
        raise _base.PreparationError('verified DTU inventory contains no complete official scenes')
    return tuple(result)
