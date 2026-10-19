//! Geometry-derived face constraints retain the principal physical directions.
//! The original samples and constructive geometry acceptance remain unchanged.
use super::{
    cross, curvature_target_size, geometry_error, normalize, relations::TopologyRelations,
};
use super::{SharedCurveError, SharedCurveErrorKind};
use runmat_geometry_core::{
    surface_principal_curvature, ExactEdge, ExactFace, ExactPcurveEvaluator, ExactSurfaceEvaluator,
    GeometryAllocationReservation, GeometryEvaluationControl, SurfaceDerivatives,
};
use runmat_meshing_core::{
    MetricContribution, MetricContributionScope, MetricSourceKind, MetricTensor3,
    SurfaceQualityTargets,
};

use runmat_meshing_size::metric::DirectionalMetricEnvelope;

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
    ) -> Result<Vec<[f64; 3]>, SharedCurveError> {
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
                uv,
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
            let derivatives = SurfaceDerivatives {
                point_m: transform.transform_point(raw.point_m),
                du_m: transform.transform_vector(raw.du_m),
                dv_m: transform.transform_vector(raw.dv_m),
                duu_m: transform.transform_vector(raw.duu_m),
                duv_m: transform.transform_vector(raw.duv_m),
                dvv_m: transform.transform_vector(raw.dvv_m),
            };
            let curvature = surface_principal_curvature(&derivatives)
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
            normals.push(normalize(cross(derivatives.du_m, derivatives.dv_m)).ok_or_else(failure)?);
            if curvature.minimum_1_per_m == 0.0 && curvature.maximum_1_per_m == 0.0 {
                continue;
            }
            let physical = |uv: [f64; 2]| {
                std::array::from_fn(|axis| {
                    derivatives.du_m[axis] * uv[0] + derivatives.dv_m[axis] * uv[1]
                })
            };
            let first = normalize(physical(curvature.minimum_direction_uv)).ok_or_else(failure)?;
            let second = physical(curvature.maximum_direction_uv);
            let projection: f64 = first.iter().zip(second).map(|(a, b)| a * b).sum();
            // At an umbilic the geometry API may return a nonorthogonal UV
            // basis. Orthonormalize in physical space, where curvature acts.
            let second = normalize(std::array::from_fn(|axis| {
                second[axis] - projection * first[axis]
            }))
            .ok_or_else(failure)?;
            let index = self
                .faces
                .binary_search_by(|known| known.id.cmp(&face.id))
                .expect("validated exact face inventory");
            let mut metric = self.metrics[index].unwrap_or(self.base);
            for (curvature, direction) in [
                (curvature.minimum_1_per_m, first),
                (curvature.maximum_1_per_m, second),
            ] {
                if curvature == 0.0 {
                    continue;
                }
                let length = curvature_target_size(
                    curvature.abs(),
                    self.quality.maximum_chordal_deviation_m,
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
