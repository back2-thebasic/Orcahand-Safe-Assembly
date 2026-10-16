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
        velocity_indices = []
        for name in joint_names:
            if not self.model.existJointName(name):
                raise ValueError(f'Missing URDF joint: {name}')
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError('Only scalar hand joints are supported')
            indices.append(joint.idx_q)
            velocity_indices.append(joint.idx_v)
        self.indices = np.array(indices, dtype=int)
        self.velocity_indices = np.array(velocity_indices, dtype=int)
        if self.offsets.shape != (self.model.nq,) or not np.all(np.isfinite(self.offsets)):
            raise ValueError('Invalid reference offsets')
        self.velocity_limits = self.model.velocityLimit[self.velocity_indices].copy()
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
            # Coal computes nearest points along with distance (including meshes).
        self._witness_q = None
        self._witnesses = {}
        self.last_gradient_info = {}

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
        # Witnesses belong only to this query, including its selected mesh pair.
        self._witness_q = None
        self._witnesses = {}
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
            witness = None
            tied = False
            for bound,i in bounds:
                if bound > max(best, 0.):
                    continue
                result = self.pin.computeDistance(self.geometry, self.geometry_data, i)
                distance = float(result.min_distance)
                if not np.isfinite(distance):
                    raise ValueError(f'Non-finite distance: {name}')
                collision = distance <= 0 and bool(self.pin.computeCollision(self.geometry, self.geometry_data, i))
                hit |= collision
                value = min(distance, 0.) if collision else distance
                if abs(value - best) <= 1e-10:
                    tied = True
                if value < best:
                    tied = tied if best - value <= 1e-10 else False
                    best = value
                    witness = (i, distance, np.array(result.getNearestPoint1(), copy=True),
                               np.array(result.getNearestPoint2(), copy=True))
            self._witnesses[name] = None if hit or tied else witness
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
        self._witness_q = np.asarray(q, dtype=float).copy()
        return result

    def distance_gradients(self, q, pair_names, epsilon=1e-4):
        """World-space witness Jacobians; finite differences for degenerate rows.

        Reuses exact witnesses from distances(q). Lower bounds are never
        differentiated. Coincident/penetrating or inconsistent witnesses and
        tied mesh minima use the original numerical path, not an invented normal.
        Like distances(), this mutable geometry model is not thread-safe.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != self.offsets.shape or not np.all(np.isfinite(q)):
            raise ValueError('Expected finite joint radians in configured order')
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('epsilon must be positive')
        names = list(pair_names)
        gradients = np.zeros((len(names), q.size))
        self.last_gradient_info = {'analytic_rows': 0, 'finite_difference_rows': 0}
        if not names:
            return gradients
        if (self._witness_q is None or not np.array_equal(q, self._witness_q)
                or any(name not in self._witnesses for name in names)):
            self.distances(q, names)
        urdf_q = np.empty(self.model.nq)
        urdf_q[self.indices] = q - self.offsets
        self.pin.computeJointJacobians(self.model, self.data, urdf_q)
        joint_jacobians = {}

        def point_jacobian(geometry_id, point):
            joint_id = self.geometry.geometryObjects[geometry_id].parentJoint
            if joint_id == 0:
                return np.zeros((3, self.model.nv))
            if joint_id not in joint_jacobians:
                joint_jacobians[joint_id] = self.pin.getJointJacobian(
                    self.model, self.data, joint_id, self.pin.LOCAL_WORLD_ALIGNED).copy()
            jac = joint_jacobians[joint_id]
            offset = point - self.data.oMi[joint_id].translation
            # v_point = v_joint + omega x (point - joint_origin).
            return jac[:3] + np.cross(jac[3:].T, offset).T

        numerical = []
        for row, name in enumerate(names):
            witness = self._witnesses[name]
            if witness is None:
                numerical.append(row)
                continue
            index, distance, a, b = witness
            separation = np.linalg.norm(a - b)
            if (not np.all(np.isfinite(np.r_[a, b, distance, separation]))
                    or distance <= 1e-9 or separation <= 1e-9
                    or not np.isclose(separation, distance, rtol=1e-5, atol=1e-8)):
                numerical.append(row)
                continue
            pair = self.geometry.collisionPairs[index]
            jac = point_jacobian(pair.first, a) - point_jacobian(pair.second, b)
            gradients[row] = ((a - b) / separation @ jac)[self.velocity_indices]
            irrelevant = [i for i in range(q.size) if i not in self.gradient_columns[name]]
            gradients[row, irrelevant] = 0.
        if numerical:
            # Calculate these last: numerical queries overwrite the witness cache.
            gradients[numerical] = finite_difference_gradients(
                self, q, [names[i] for i in numerical], epsilon)
        self.last_gradient_info = {'analytic_rows': len(names) - len(numerical),
                                   'finite_difference_rows': len(numerical)}
        if not np.all(np.isfinite(gradients)):
            raise ValueError('Non-finite gradient')
        return gradients


def distance_gradients(model: CollisionModel, q, pair_names, epsilon=1e-4):
    """Use the analytic backend when available; support distance-only models."""
    method = getattr(model, 'distance_gradients', None)
    if method is not None:
        return method(q, pair_names, epsilon)
    return finite_difference_gradients(model, q, pair_names, epsilon)


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
