//! Geometry-derived face constraints retain the principal physical directions.
//! The original samples and constructive geometry acceptance remain unchanged.
use super::{
    cross, curvature_target_size, geometry_error, norm, normalize, relations::TopologyRelations,
};
use super::{SharedCurveError, SharedCurveErrorKind};
use runmat_geometry_core::{
    surface_principal_curvature, ExactEdge, ExactFace, ExactPcurveEvaluator, ExactSurfaceEvaluator,
    GeometryAllocationReservation, GeometryEvaluationControl,
};
use runmat_meshing_core::{
    MetricContribution, MetricContributionScope, MetricSourceKind, MetricTensor3,
    SurfaceQualityTargets,
};

use runmat_meshing_size::metric::DirectionalMetricEnvelope;

/// One regular boundary sample of a face: where an edge's pcurve lands on it, with the unit
/// surface normal there (orientation as parameterized; consumers use it up to sign).
#[derive(Clone, Debug)]
pub(super) struct FaceNormalSample {
    pub face_id: runmat_geometry_core::PersistentEntityId,
    pub point_m: [f64; 3],
    pub normal: [f64; 3],
}

pub(super) struct FaceMetricSamples<'a> {
    base: DirectionalMetricEnvelope,
    quality: SurfaceQualityTargets,
    faces: &'a [ExactFace],
    metrics: Vec<Option<DirectionalMetricEnvelope>>,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> FaceMetricSamples<'a> {
    pub fn new(
        faces: &'a [ExactFace],
        control: &'a dyn GeometryEvaluationControl,
        base: MetricTensor3,
        quality: SurfaceQualityTargets,
    ) -> Result<Self, SharedCurveError> {
        let bytes = faces
            .len()
            .checked_mul(std::mem::size_of::<Option<DirectionalMetricEnvelope>>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                SharedCurveError::invalid_request("face metric inventory", "storage overflow")
            })?;
        let storage = GeometryAllocationReservation::new(control, bytes).map_err(|error| {
            SharedCurveError::new(
                SharedCurveErrorKind::GeometryEvaluation(error.kind),
                "face metric inventory",
                error.reason,
            )
        })?;
        Ok(Self {
            base: DirectionalMetricEnvelope::new(base).map_err(|error| {
                SharedCurveError::invalid_request("face metric baseline", error.to_string())
            })?,
            quality,
            faces,
            metrics: vec![None; faces.len()],
            _storage: storage,
        })
    }
    pub fn sample(
        &mut self,
        relations: &TopologyRelations<'_>,
        edge: &ExactEdge,
        evaluator: &(impl ExactPcurveEvaluator + ExactSurfaceEvaluator + ?Sized),
        control: &dyn GeometryEvaluationControl,
        parameter: f64,
    ) -> Result<Vec<FaceNormalSample>, SharedCurveError> {
        let mut normals = Vec::new();
        for coedge in relations.edge_coedges(&edge.id) {
            let face = relations.face(&coedge.face_id);
            let uv = ExactPcurveEvaluator::point(
                evaluator,
                &coedge.pcurve_evaluator_id,
                parameter,
                control,
            )
            .map_err(|error| geometry_error(edge, error))?;
            let raw = ExactSurfaceEvaluator::derivatives(
                evaluator,
                &face.surface_evaluator_id,
                uv.map(|coordinate| coordinate.with_native_coordinate(|value, _axis| value)),
                control,
            )
            .map_err(|error| geometry_error(edge, error))?;
            let transform = relations.world_transform_for(&face.id).map_err(|error| {
                SharedCurveError::invalid_request(
                    "face metric occurrence transform",
                    error.to_string(),
                )
                .for_edge(&edge.id)
            })?;
            let derivatives = raw
                .transformed(&transform)
                .map_err(|error| geometry_error(edge, error))?;
            let failure = || {
                SharedCurveError::new(
                    SharedCurveErrorKind::GeometryEvaluation(
                        runmat_geometry_core::GeometryEvaluationErrorKind::InvalidResult,
                    ),
                    "directional face curvature metric",
                    "surface does not define finite regular physical directions",
                )
                .for_edge(&edge.id)
            };
            // At a parameterization singularity (a sphere pole, a cone apex) the surface
            // derivatives span no tangent plane, so the sample has no normal or principal frame.
            // The edge's other samples carry the face's curvature and its normal for feature
            // detection.
            let du = derivatives
                .du_m
                .with_canonical_storage(|values, _axes| *values);
            let dv = derivatives
                .dv_m
                .with_canonical_storage(|values, _axes| *values);
            let area = cross(du, dv);
            let scale = norm(du).powi(2) + norm(dv).powi(2);
            if norm(area).partial_cmp(&(SINGULAR_PARAMETERIZATION_RATIO * scale))
                != Some(std::cmp::Ordering::Greater)
            {
                continue;
            }
            let normal = normalize(area).ok_or_else(failure)?;
            let curvature = surface_principal_curvature(&derivatives)
                .map_err(|error| geometry_error(edge, error))?;
            normals.push(FaceNormalSample {
                face_id: face.id.clone(),
                point_m: derivatives.point_m.canonical_value(),
                normal,
            });
            if curvature.minimum_1_per_m.canonical_value() == 0.0
                && curvature.maximum_1_per_m.canonical_value() == 0.0
            {
                continue;
            }
            let physical =
                |uv: [f64; 2]| std::array::from_fn(|axis| du[axis] * uv[0] + dv[axis] * uv[1]);
            let (first, second) = principal_frame(
                physical(curvature.minimum_direction_uv),
                physical(curvature.maximum_direction_uv),
                du,
                normal,
            )
            .ok_or_else(failure)?;
            let index = self
                .faces
                .binary_search_by(|known| known.id.cmp(&face.id))
                .expect("validated exact face inventory");
            let mut metric = self.metrics[index].unwrap_or(self.base);
            for (curvature, direction) in [
                (curvature.minimum_1_per_m.canonical_value(), first),
                (curvature.maximum_1_per_m.canonical_value(), second),
            ] {
                if curvature == 0.0 {
                    continue;
                }
                let length = curvature_target_size(
                    curvature.abs(),
                    self.quality.maximum_chordal_deviation_m.canonical_value(),
                    self.quality.maximum_normal_deviation_degrees,
                )
                .ok_or_else(failure)?;
                metric.include(direction, length).map_err(|error| {
                    SharedCurveError::invalid_request("directional face metric", error.to_string())
                        .for_edge(&edge.id)
                })?;
            }
            self.metrics[index] = Some(metric);
        }
        Ok(normals)
    }
    pub fn into_contributions(self) -> Result<Vec<MetricContribution>, SharedCurveError> {
        self.metrics
            .into_iter()
            .zip(self.faces)
            .filter_map(|(metric, face)| {
                metric.map(|metric| {
                    Ok(MetricContribution {
                        source: MetricSourceKind::Face,
                        scope: MetricContributionScope::Entity {
                            entity_id: face.id.clone(),
                        },
                        metric: metric.metric().map_err(|error| {
                            SharedCurveError::invalid_request(
                                "face metric envelope",
                                error.to_string(),
                            )
                        })?,
                    })
                })
            })
            .collect()
    }
}

/// Area element below this fraction of the squared derivative scale marks a parameterization
/// singularity: at a sphere pole one derivative is roundoff (~1e-16 of the other).
const SINGULAR_PARAMETERIZATION_RATIO: f64 = 1.0e-10;

/// Orthonormal physical principal directions from the curvature solver's UV directions mapped to
/// physical space. At an umbilic the solver may return a nonorthogonal basis, so the second
/// direction is orthogonalized against the first. Near an umbilic (a spherical blend corner)
/// both directions come back nearly parallel and their difference is roundoff; every tangent
/// direction is then principal and the two curvatures agree, so the tangent direction orthogonal
/// to the first is exact.
pub(super) fn principal_frame(
    minimum: [f64; 3],
    maximum: [f64; 3],
    du: [f64; 3],
    normal: [f64; 3],
) -> Option<([f64; 3], [f64; 3])> {
    let first = normalize(minimum).or_else(|| normalize(du))?;
    let projection: f64 = first.iter().zip(maximum).map(|(a, b)| a * b).sum();
    let remainder: [f64; 3] = std::array::from_fn(|axis| maximum[axis] - projection * first[axis]);
    let scale = norm(maximum);
    let second = if norm(remainder) > 1.0e-8 * scale {
        normalize(remainder)?
    } else {
        normalize(cross(normal, first))?
    };
    Some((first, second))
}

#[cfg(test)]
mod tests {
    use super::principal_frame;

    fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn umbilic_points_with_parallel_principal_directions_get_an_orthonormal_tangent_frame() {
        // A spherical fillet corner: the solver returned the same direction twice.
        let normal = [0.0, 0.0, 1.0];
        let direction = [0.6, 0.8, 0.0];
        for maximum in [direction, [0.6 + 1.0e-14, 0.8, 0.0], [0.0; 3]] {
            let (first, second) = principal_frame(direction, maximum, [1.0, 0.0, 0.0], normal)
                .expect("an umbilic still has a tangent frame");
            assert!((dot(first, first) - 1.0).abs() < 1e-12);
            assert!((dot(second, second) - 1.0).abs() < 1e-12);
            assert!(dot(first, second).abs() < 1e-12);
            assert!(dot(second, normal).abs() < 1e-12);
        }
        // Distinct directions are only orthogonalized.
        let (first, second) =
            principal_frame([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], normal).unwrap();
        assert_eq!(first, [1.0, 0.0, 0.0]);
        assert!((second[1] - 1.0).abs() < 1e-12);
    }
}
