use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{GeometryEvaluationControl, PersistentEntityId};
use runmat_meshing_core::{
    MetricContribution, MetricContributionScope, MetricSourceKind, MetricTensor3,
};

use super::{SharedCurveError, SharedCurveErrorKind};

#[derive(Clone, Debug)]
pub(super) struct EdgeWitness {
    pub edge_id: PersistentEntityId,
    pub endpoint_ids: BTreeSet<PersistentEntityId>,
    pub face_ids: BTreeSet<PersistentEntityId>,
    pub points_m: Vec<[f64; 3]>,
}

pub(super) fn derive_proximity_contributions(
    mut witnesses: Vec<EdgeWitness>,
    contact_face_pairs: &BTreeSet<(PersistentEntityId, PersistentEntityId)>,
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
    let mut minimum_by_source = BTreeMap::<(PersistentEntityId, MetricSourceKind), f64>::new();
    for left_index in 0..witnesses.len() {
        let left = &witnesses[left_index];
        let left_bounds = bounds(left);
        for right in &witnesses[left_index + 1..] {
            let right_bounds = bounds(right);
            if right_bounds.0[0] - left_bounds.1[0] > search_radius_m {
                break;
            }
            if !left.endpoint_ids.is_disjoint(&right.endpoint_ids)
                || bounds_distance(left_bounds, right_bounds) > search_radius_m
            {
                continue;
            }
            let source =
                if face_sets_are_contact(&left.face_ids, &right.face_ids, contact_face_pairs) {
                    MetricSourceKind::Contact
                } else {
                    MetricSourceKind::Proximity
                };
            let mut minimum = f64::INFINITY;
            for left_point in &left.points_m {
                for right_point in &right.points_m {
                    control
                        .consume_search_work(1)
                        .map_err(|error| geometry_error(&left.edge_id, error))?;
                    minimum = minimum.min(distance(*left_point, *right_point));
                }
            }
            if minimum == 0.0 {
                if source == MetricSourceKind::Contact {
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
                for edge_id in [&left.edge_id, &right.edge_id] {
                    minimum_by_source
                        .entry((edge_id.clone(), source))
                        .and_modify(|current| *current = current.min(minimum))
                        .or_insert(minimum);
                }
            }
        }
    }
    minimum_by_source
        .into_iter()
        .map(|((edge_id, source), separation)| {
            Ok(MetricContribution {
                source,
                scope: MetricContributionScope::Entity {
                    entity_id: edge_id.clone(),
                },
                metric: MetricTensor3::isotropic_length_m(separation * 0.5).map_err(|error| {
                    SharedCurveError::invalid_request("curve proximity metric", error.to_string())
                        .for_edge(&edge_id)
                })?,
            })
        })
        .collect()
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

fn face_sets_are_contact(
    left: &BTreeSet<PersistentEntityId>,
    right: &BTreeSet<PersistentEntityId>,
    contact_pairs: &BTreeSet<(PersistentEntityId, PersistentEntityId)>,
) -> bool {
    left.iter().any(|left| {
        right
            .iter()
            .any(|right| contact_pairs.contains(&ordered_pair(left.clone(), right.clone())))
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
        let contributions =
            derive_proximity_contributions(witnesses, &BTreeSet::new(), 1.0, &Control).unwrap();
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
            &BTreeSet::new(),
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
    fn authored_face_pair_emits_contact_constraints() {
        let mut left = witness(1, 0.0);
        let mut right = witness(2, 0.2);
        let left_face = id(PersistentEntityKind::Face, 11);
        let right_face = id(PersistentEntityKind::Face, 12);
        left.face_ids.insert(left_face.clone());
        right.face_ids.insert(right_face.clone());

        let contributions = derive_proximity_contributions(
            vec![left, right],
            &BTreeSet::from([ordered_pair(left_face, right_face)]),
            1.0,
            &Control,
        )
        .unwrap();

        assert_eq!(contributions.len(), 2);
        assert!(contributions
            .iter()
            .all(|contribution| contribution.source == MetricSourceKind::Contact));
    }

    fn witness(seed: u8, x: f64) -> EdgeWitness {
        EdgeWitness {
            edge_id: id(PersistentEntityKind::Edge, seed),
            endpoint_ids: BTreeSet::from([id(PersistentEntityKind::Vertex, seed)]),
            face_ids: BTreeSet::new(),
            points_m: vec![[x, 0.0, 0.0]],
        }
    }

    fn id(kind: PersistentEntityKind, seed: u8) -> PersistentEntityId {
        PersistentEntityId {
            kind,
            source_topology_id: format!("entity-{seed}"),
            assembly_path: Vec::new(),
        }
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
