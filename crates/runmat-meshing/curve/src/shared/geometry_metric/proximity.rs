use std::collections::BTreeSet;

use runmat_geometry_core::{
    GeometryAllocationReservation, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::{
    MetricContribution, MetricContributionScope, MetricSourceKind, MetricTensor3,
};

use super::{SharedCurveError, SharedCurveErrorKind};
use runmat_meshing_size::metric::DirectionalMetricEnvelope;

#[derive(Clone, Debug)]
pub(super) struct EdgeWitness {
    pub edge_id: PersistentEntityId,
    pub endpoint_ids: BTreeSet<PersistentEntityId>,
    pub face_ids: BTreeSet<PersistentEntityId>,
    /// Solids bounded by the edge's faces; empty for sheet geometry.
    pub solid_ids: BTreeSet<PersistentEntityId>,
    pub points_m: Vec<[f64; 3]>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct ContactProximityPairs {
    pub faces: BTreeSet<(PersistentEntityId, PersistentEntityId)>,
    pub vertices: BTreeSet<(PersistentEntityId, PersistentEntityId)>,
}

pub(super) fn derive_proximity_contributions(
    mut witnesses: Vec<EdgeWitness>,
    contact_pairs: &ContactProximityPairs,
    base_metric: MetricTensor3,
    search_radius_m: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<Vec<MetricContribution>, SharedCurveError> {
    if !search_radius_m.is_finite() || search_radius_m <= 0.0 {
        return Err(SharedCurveError::invalid_request(
            "curve proximity radius",
            "must be finite and greater than zero",
        ));
    }
    for witness in &witnesses {
        if witness.points_m.is_empty()
            || witness
                .points_m
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
        {
            return Err(SharedCurveError::new(
                SharedCurveErrorKind::GeometryEvaluation(
                    runmat_geometry_core::GeometryEvaluationErrorKind::InvalidResult,
                ),
                "curve proximity witnesses",
                "each exact edge must provide finite witness points",
            )
            .for_edge(&witness.edge_id));
        }
    }
    witnesses.sort_by(|left, right| {
        bounds(left).0[0]
            .total_cmp(&bounds(right).0[0])
            .then_with(|| left.edge_id.cmp(&right.edge_id))
    });
    let bytes = witnesses
        .len()
        .checked_mul(std::mem::size_of::<[Option<DirectionalMetricEnvelope>; 2]>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            SharedCurveError::invalid_request("proximity metric inventory", "storage overflow")
        })?;
    let _storage = GeometryAllocationReservation::new(control, bytes).map_err(|error| {
        SharedCurveError::new(
            SharedCurveErrorKind::GeometryEvaluation(error.kind),
            "proximity metric inventory",
            error.reason,
        )
    })?;
    let base = DirectionalMetricEnvelope::new(base_metric).map_err(|error| {
        SharedCurveError::invalid_request("proximity metric baseline", error.to_string())
    })?;
    let mut metrics = vec![[None; 2]; witnesses.len()];
    for left_index in 0..witnesses.len() {
        let left = &witnesses[left_index];
        let left_bounds = bounds(left);
        for (right_index, right) in witnesses.iter().enumerate().skip(left_index + 1) {
            let right_bounds = bounds(right);
            if right_bounds.0[0] - left_bounds.1[0] > search_radius_m {
                break;
            }
            if !left.endpoint_ids.is_disjoint(&right.endpoint_ids)
                || bounds_distance(left_bounds, right_bounds) > search_radius_m
            {
                continue;
            }
            let source = if witnesses_are_contact(left, right, contact_pairs) {
                MetricSourceKind::Contact
            } else {
                MetricSourceKind::Proximity
            };
            let mut minimum = f64::INFINITY;
            let mut separation_direction = [0.0; 3];
            for left_point in &left.points_m {
                for right_point in &right.points_m {
                    control
                        .consume_search_work(1)
                        .map_err(|error| geometry_error(&left.edge_id, error))?;
                    let separation = distance(*left_point, *right_point);
                    if separation < minimum {
                        minimum = separation;
                        separation_direction =
                            std::array::from_fn(|axis| right_point[axis] - left_point[axis]);
                    }
                }
            }
            if minimum == 0.0 {
                // Edges of separate bodies that touch (a part seated in a recess, stacked
                // blocks) coincide; that is contact, with no gap to resolve. Within one body,
                // coincident nonincident edges are a topology defect.
                let separate_bodies = !left.solid_ids.is_empty()
                    && !right.solid_ids.is_empty()
                    && left.solid_ids.is_disjoint(&right.solid_ids);
                if source == MetricSourceKind::Contact || separate_bodies {
                    continue;
                }
                return Err(SharedCurveError::new(
                    SharedCurveErrorKind::GeometricMismatch,
                    "curve proximity",
                    "nonincident exact edge witnesses coincide",
                )
                .for_edge(&left.edge_id));
            }
            if minimum < search_radius_m {
                let source_index = usize::from(source == MetricSourceKind::Contact);
                for index in [left_index, right_index] {
                    let current = metrics[index][source_index].get_or_insert(base);
                    current
                        .include(separation_direction, minimum * 0.5)
                        .map_err(|error| {
                            SharedCurveError::invalid_request(
                                "directional proximity metric",
                                error.to_string(),
                            )
                            .for_edge(&witnesses[index].edge_id)
                        })?;
                }
            }
        }
    }
    let mut contributions = Vec::new();
    for (witness, metrics) in witnesses.into_iter().zip(metrics) {
        for (metric, source) in metrics
            .into_iter()
            .zip([MetricSourceKind::Proximity, MetricSourceKind::Contact])
        {
            if let Some(metric) = metric {
                contributions.push(MetricContribution {
                    source,
                    scope: MetricContributionScope::Entity {
                        entity_id: witness.edge_id.clone(),
                    },
                    metric: metric.metric().map_err(|error| {
                        SharedCurveError::invalid_request(
                            "proximity metric envelope",
                            error.to_string(),
                        )
                        .for_edge(&witness.edge_id)
                    })?,
                });
            }
        }
    }
    // The parent request performs the one authoritative canonical ordering
    // together with curvature and feature contributions.
    Ok(contributions)
}

fn bounds(witness: &EdgeWitness) -> ([f64; 3], [f64; 3]) {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for point in &witness.points_m {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    (minimum, maximum)
}

fn bounds_distance(left: ([f64; 3], [f64; 3]), right: ([f64; 3], [f64; 3])) -> f64 {
    (0..3)
        .map(|axis| {
            if left.1[axis] < right.0[axis] {
                right.0[axis] - left.1[axis]
            } else if right.1[axis] < left.0[axis] {
                left.0[axis] - right.1[axis]
            } else {
                0.0
            }
        })
        .map(|distance| distance * distance)
        .sum::<f64>()
        .sqrt()
}

fn witnesses_are_contact(
    left: &EdgeWitness,
    right: &EdgeWitness,
    contact_pairs: &ContactProximityPairs,
) -> bool {
    sets_have_pair(&left.face_ids, &right.face_ids, &contact_pairs.faces)
        || sets_have_pair(
            &left.endpoint_ids,
            &right.endpoint_ids,
            &contact_pairs.vertices,
        )
}

fn sets_have_pair(
    left: &BTreeSet<PersistentEntityId>,
    right: &BTreeSet<PersistentEntityId>,
    pairs: &BTreeSet<(PersistentEntityId, PersistentEntityId)>,
) -> bool {
    left.iter().any(|left| {
        right
            .iter()
            .any(|right| pairs.contains(&ordered_pair(left.clone(), right.clone())))
    })
}

pub(super) fn ordered_pair(
    left: PersistentEntityId,
    right: PersistentEntityId,
) -> (PersistentEntityId, PersistentEntityId) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn geometry_error(
    edge_id: &PersistentEntityId,
    error: runmat_geometry_core::GeometryEvaluationError,
) -> SharedCurveError {
    SharedCurveError::new(
        SharedCurveErrorKind::GeometryEvaluation(error.kind),
        "curve proximity search",
        error.reason,
    )
    .for_edge(edge_id)
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{GeometryEvaluationError, PersistentEntityKind};

    #[test]
    fn sweep_emits_canonical_nearby_constraints_and_skips_far_edges() {
        let mut witnesses = vec![witness(3, 5.0), witness(1, 0.0), witness(2, 0.2)];
        witnesses[0].points_m.push([6.0, 0.0, 0.0]);
        let contributions = derive_proximity_contributions(
            witnesses,
            &ContactProximityPairs::default(),
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
            1.0,
            &Control,
        )
        .unwrap();
        assert_eq!(contributions.len(), 2);
        assert!(contributions
            .iter()
            .all(|contribution| contribution.source == MetricSourceKind::Proximity));
        assert!(contributions[0].scope != contributions[1].scope);
    }

    #[test]
    fn sweep_preserves_typed_search_budget_failure() {
        let error = derive_proximity_contributions(
            vec![witness(1, 0.0), witness(2, 0.2)],
            &ContactProximityPairs::default(),
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
            1.0,
            &ExhaustedControl,
        )
        .unwrap_err();
        assert_eq!(
            error.kind,
            SharedCurveErrorKind::GeometryEvaluation(
                runmat_geometry_core::GeometryEvaluationErrorKind::SearchWorkBudgetExceeded
            )
        );
    }

    #[test]
    fn coincident_edges_of_separate_bodies_are_contact_and_of_one_body_a_defect() {
        let mut left = witness(1, 0.0);
        let mut right = witness(2, 0.0);
        left.solid_ids.insert(id(PersistentEntityKind::Solid, 21));
        right.solid_ids.insert(id(PersistentEntityKind::Solid, 22));
        let derive = |left: EdgeWitness, right: EdgeWitness| {
            derive_proximity_contributions(
                vec![left, right],
                &ContactProximityPairs::default(),
                MetricTensor3::isotropic_length_m(1.0).unwrap(),
                1.0,
                &Control,
            )
        };
        // A part seated against another (the speaker cone in the air's recess): no gap.
        assert!(derive(left.clone(), right.clone()).unwrap().is_empty());
        right.solid_ids = left.solid_ids.clone();
        assert!(derive(left, right).is_err());
    }

    #[test]
    fn authored_face_pair_emits_contact_constraints() {
        let mut left = witness(1, 0.0);
        let mut right = witness(2, 0.2);
        let left_face = id(PersistentEntityKind::Face, 11);
        let right_face = id(PersistentEntityKind::Face, 12);
        left.face_ids.insert(left_face.clone());
        right.face_ids.insert(right_face.clone());

        let contributions = derive_proximity_contributions(
            vec![left, right],
            &ContactProximityPairs {
                faces: BTreeSet::from([ordered_pair(left_face, right_face)]),
                vertices: BTreeSet::new(),
            },
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
            1.0,
            &Control,
        )
        .unwrap();

        assert_eq!(contributions.len(), 2);
        assert!(contributions
            .iter()
            .all(|contribution| contribution.source == MetricSourceKind::Contact));
    }

    #[test]
    fn authored_contact_boundary_vertices_allow_edges_that_only_meet_at_a_corner() {
        let mut left = witness(1, 0.0);
        let mut right = witness(2, 0.0);
        let left_vertex = left.endpoint_ids.iter().next().unwrap().clone();
        let right_vertex = right.endpoint_ids.iter().next().unwrap().clone();
        left.face_ids.insert(id(PersistentEntityKind::Face, 21));
        right.face_ids.insert(id(PersistentEntityKind::Face, 22));

        let contributions = derive_proximity_contributions(
            vec![left, right],
            &ContactProximityPairs {
                faces: BTreeSet::new(),
                vertices: BTreeSet::from([ordered_pair(left_vertex, right_vertex)]),
            },
            MetricTensor3::isotropic_length_m(1.0).unwrap(),
            1.0,
            &Control,
        )
        .unwrap();

        assert!(contributions.is_empty());
    }

    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Three translated/rotated long-edge pairs retain half-separation resolution across a
    //   thin gap while preserving the long direction at the global baseline. Original contact and
    //   search-control cases remain separate. Sampled witnesses are not a closest-distance certificate
    //   or general local spatial-support qualification.
    #[test]
    fn thin_extrusion_gap_resolution_preserves_long_direction_under_rotation_and_translation() {
        for (long, gap) in [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.6, 0.8, 0.0], [-0.8, 0.6, 0.0]),
            ([0.0, 0.0, 1.0], [0.6, 0.8, 0.0]),
        ] {
            let separation = 0.0003535533905932738;
            let base = MetricTensor3::isotropic_length_m(0.01).unwrap();
            let mut left = witness(1, 0.0);
            let mut right = witness(2, 0.0);
            left.points_m = [0.0, 0.0175, 0.035]
                .map(|distance| {
                    std::array::from_fn(|axis| [2.0, -3.0, 4.0][axis] + distance * long[axis])
                })
                .to_vec();
            right.points_m = left
                .points_m
                .iter()
                .map(|point| std::array::from_fn(|axis| point[axis] + separation * gap[axis]))
                .collect();
            let contributions = derive_proximity_contributions(
                vec![left, right],
                &ContactProximityPairs::default(),
                base,
                0.02,
                &Control,
            )
            .unwrap();
            assert_eq!(contributions.len(), 2);
            let density = |metric: MetricTensor3, v: [f64; 3]| {
                let [x, y, z] = v;
                metric.xx * x * x
                    + metric.yy * y * y
                    + metric.zz * z * z
                    + 2.0 * (metric.xy * x * y + metric.xz * x * z + metric.yz * y * z)
            };
            for contribution in contributions {
                assert!((density(contribution.metric, long) / base.xx - 1.0).abs() < 1e-10);
                assert!(
                    (density(contribution.metric, gap) * (separation * 0.5).powi(2) - 1.0).abs()
                        < 1e-8
                );
            }
        }
    }

    fn witness(seed: u8, x: f64) -> EdgeWitness {
        EdgeWitness {
            edge_id: id(PersistentEntityKind::Edge, seed),
            endpoint_ids: BTreeSet::from([id(PersistentEntityKind::Vertex, seed)]),
            face_ids: BTreeSet::new(),
            solid_ids: BTreeSet::new(),
            points_m: vec![[x, 0.0, 0.0]],
        }
    }

    fn id(kind: PersistentEntityKind, seed: u8) -> PersistentEntityId {
        PersistentEntityId::new(kind, format!("entity-{seed}"), Vec::new())
    }

    struct Control;

    impl GeometryEvaluationControl for Control {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }

    struct ExhaustedControl;

    impl GeometryEvaluationControl for ExhaustedControl {
        fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
        fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Err(GeometryEvaluationError::new(
                runmat_geometry_core::GeometryEvaluationErrorKind::SearchWorkBudgetExceeded,
                "test search budget exhausted",
            ))
        }
        fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
            Ok(())
        }
    }
}
