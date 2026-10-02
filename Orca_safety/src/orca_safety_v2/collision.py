"""Geometry-only interface; physical joint radians in caller-provided order."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class CollisionDistance:
    pair_name: str
    distance: float
    collision: bool
    is_lower_bound: bool = False


class CollisionModel(Protocol):
    def distances(self, q, pair_names=None) -> list[CollisionDistance]: ...


class FCLDistanceModel:
    """URDF mesh backend. Each configured link pair aggregates all its meshes.

    Offset mapping is supplied by the embodiment adapter, never inferred from
    array position. Mesh overlap reports collision; penetration depths are not
    assumed to provide a smooth signed-distance field.
    """
    def __init__(self, urdf_path, package_dirs, joint_names, reference_offsets, pairs):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.geometry = pin.buildGeomFromUrdf(
            self.model, str(urdf_path), pin.GeometryType.COLLISION,
            package_dirs=[str(p) for p in package_dirs],
        )
        self.joint_names = tuple(joint_names)
        self.offsets = np.asarray(reference_offsets, dtype=float).copy()
        if len(set(joint_names)) != len(joint_names) or len(joint_names) != self.model.nq:
            raise ValueError('Joint names must cover every URDF DoF exactly once')
        indices = []
        for name in joint_names:
            if not self.model.existJointName(name):
                raise ValueError(f'Missing URDF joint: {name}')
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1:
                raise ValueError('Only scalar hand joints are supported')
            indices.append(joint.idx_q)
        self.indices = np.array(indices, dtype=int)
        if self.offsets.shape != (self.model.nq,) or not np.all(np.isfinite(self.offsets)):
            raise ValueError('Invalid reference offsets')
        self.velocity_limits = self.model.velocityLimit[self.indices].copy()
        by_link = {}
        for i, obj in enumerate(self.geometry.geometryObjects):
            name = self.model.frames[obj.parentFrame].name
            by_link.setdefault(name, []).append(i)
        self.pair_indices = {}
        self.gradient_columns = {}
        seen = set()
        for a, b in pairs:
            key = f'{a}__{b}'
            if a == b or frozenset((a,b)) in seen or a not in by_link or b not in by_link:
                raise ValueError(f'Invalid/duplicate collision pair: {a}, {b}')
            seen.add(frozenset((a,b)))
            ids = []
            for i in by_link[a]:
                for j in by_link[b]:
                    self.geometry.addCollisionPair(pin.CollisionPair(i,j))
                    ids.append(len(self.geometry.collisionPairs)-1)
            self.pair_indices[key] = ids
            def ancestors(link):
                joint = self.model.frames[self.model.getFrameId(link)].parentJoint
                result = set()
                while joint:
                    result.add(joint)
                    joint = int(self.model.parents[joint])
                return result
            # A common ancestor moves both objects rigidly and cannot change
            # their distance. Other joints have exactly zero derivatives.
            relevant = ancestors(a) ^ ancestors(b)
            self.gradient_columns[key] = {
                i for i,n in enumerate(self.joint_names)
                if self.model.getJointId(n) in relevant
            }
        if not self.pair_indices:
            raise ValueError('At least one explicit collision pair is required')
        self.geometry_data = self.geometry.createData()
        self.broadphase_distance = None
        centers, halves = [], []
        for obj in self.geometry.geometryObjects:
            obj.geometry.computeLocalAABB()
            aabb = obj.geometry.aabb_local
            centers.append((aabb.min_ + aabb.max_) / 2)
            halves.append((aabb.max_ - aabb.min_) / 2)
        self._local_centers = np.asarray(centers)
        self._local_halves = np.asarray(halves)
        for req in self.geometry_data.distanceRequests:
            req.enable_signed_distance = True

    def _update(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != self.offsets.shape or not np.all(np.isfinite(q)):
            raise ValueError('Expected finite joint radians in configured order')
        urdf_q = np.empty(self.model.nq)
        urdf_q[self.indices] = q - self.offsets
        self.pin.updateGeometryPlacements(self.model, self.data, self.geometry,
                                         self.geometry_data, urdf_q)
        return urdf_q

    def distances(self, q, pair_names=None):
        self._update(q)
        centers = np.array([t.rotation @ c + t.translation for t,c in
                            zip(self.geometry_data.oMg, self._local_centers)])
        halves = np.array([np.abs(t.rotation) @ h for t,h in
                           zip(self.geometry_data.oMg, self._local_halves)])
        names = list(self.pair_indices) if pair_names is None else list(pair_names)

        def measure(name, cutoff):
            bounds = []
            for i in self.pair_indices[name]:
                pair = self.geometry.collisionPairs[i]
                gap = np.maximum(np.abs(centers[pair.first]-centers[pair.second])
                                 - halves[pair.first]-halves[pair.second], 0.)
                # Conservative world AABB separation; account for roundoff.
                bounds.append((max(0., float(np.linalg.norm(gap))-1e-10), i))
            bounds.sort()
            if cutoff is not None and bounds[0][0] > cutoff:
                return CollisionDistance(name, bounds[0][0], False, True)
            best = float('inf'); hit = False
            for bound,i in bounds:
                if bound > max(best, 0.):
                    continue
                distance = float(self.pin.computeDistance(self.geometry, self.geometry_data, i).min_distance)
                if not np.isfinite(distance):
                    raise ValueError(f'Non-finite distance: {name}')
                collision = distance <= 0 and bool(self.pin.computeCollision(self.geometry, self.geometry_data, i))
                hit |= collision
                best = min(best, min(distance, 0.) if collision else distance)
            return CollisionDistance(name, best, hit)

        # Selected-pair queries (gradients) are always exact. Far-pair bounds
        # only skip work outside activation, never inside a CBF constraint.
        cutoff = self.broadphase_distance if pair_names is None else None
        result = [measure(name, cutoff) for name in names]
        # Refine until the global minimum is exact, even if every pair is far.
        while result:
            i = min(range(len(result)), key=lambda i: result[i].distance)
            if not result[i].is_lower_bound:
                break
            result[i] = measure(result[i].pair_name, None)
        return result


def finite_difference_gradients(model: CollisionModel, q, pair_names, epsilon=1e-4):
    """Central differences in caller order; only requested active pairs queried."""
    q = np.asarray(q, dtype=float)
    names = list(pair_names)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive')
    gradients = np.zeros((len(names), q.size))
    if not names:
        return gradients
    for i in range(q.size):
        perturb = np.zeros_like(q)
        perturb[i] = epsilon
        columns = getattr(model, 'gradient_columns', None)
        selected = [r for r,name in enumerate(names) if columns is None or i in columns[name]]
        if not selected:
            continue
        subset = [names[r] for r in selected]
        plus = model.distances(q + perturb, subset)
        minus = model.distances(q - perturb, subset)
        gradients[selected, i] = [(p.distance-m.distance)/(2*epsilon) for p,m in zip(plus,minus,strict=True)]
    if not np.all(np.isfinite(gradients)):
        raise ValueError('Non-finite gradient')
    return gradients
