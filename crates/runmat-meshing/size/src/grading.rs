use std::collections::{BTreeMap, BTreeSet};

mod incremental;
mod indexed;
pub use incremental::{
    plan_metric_cap_update, MetricCapGraph, MetricCapUpdate, MetricCapUpdateError,
};
/// Realize a validated cap against a raw evaluation, exactly once.
pub fn apply_metric_grading_cap(
    evaluation: &mut ResolvedMetricEvaluation,
    cap: f64,
) -> Result<(), MetricContractError> {
    indexed::apply_cap(evaluation, cap)
}

pub use indexed::grade_indexed_metric_evaluations;

use crate::metric::{MetricContractError, ResolvedMetricEvaluation};

/// Applies a conservative Lipschitz-like size bound over a canonical entity-adjacency graph.
///
/// A fine metric may constrain a graph neighbor to at most `maximum_grading_ratio` times its
/// characteristic size. Tightening scales the original tensor, preserving its anisotropy
/// without weakening any direction. The result is independent of map construction and traversal order.
pub fn grade_metric_evaluations<Identity: Clone + Ord>(
    maximum_grading_ratio: f64,
    adjacency: &BTreeMap<Identity, BTreeSet<Identity>>,
    evaluations: &mut BTreeMap<Identity, ResolvedMetricEvaluation>,
) -> Result<(), MetricContractError> {
    if !maximum_grading_ratio.is_finite() || maximum_grading_ratio < 1.0 {
        return Err(invalid(
            "maximum metric grading ratio",
            "must be finite and at least one",
        ));
    }
    validate_graph(adjacency, evaluations)?;
    let identities = adjacency.keys().collect::<Vec<_>>();
    let indices = identities
        .iter()
        .enumerate()
        .map(|(index, identity)| (*identity, index))
        .collect::<BTreeMap<_, _>>();
    let mut offsets = vec![0];
    let mut neighbors = Vec::new();
    for identity in &identities {
        neighbors.extend(
            adjacency[*identity]
                .iter()
                .map(|neighbor| indices[neighbor]),
        );
        offsets.push(neighbors.len());
    }
    let updates = indexed::plan(
        maximum_grading_ratio,
        &identities,
        &offsets,
        &neighbors,
        |index| {
            evaluations[identities[index]]
                .metric
                .conservative_minimum_length_m()
        },
    )?;
    for (index, cap) in updates {
        let evaluation = evaluations
            .get_mut(identities[index])
            .ok_or_else(|| invalid("metric grading graph", "lost a validated entity"))?;
        indexed::apply_cap(evaluation, cap)?;
    }
    Ok(())
}

fn validate_graph<Identity: Ord>(
    adjacency: &BTreeMap<Identity, BTreeSet<Identity>>,
    evaluations: &BTreeMap<Identity, ResolvedMetricEvaluation>,
) -> Result<(), MetricContractError> {
    if adjacency.len() != evaluations.len()
        || adjacency.keys().ne(evaluations.keys())
        || adjacency.iter().any(|(entity, neighbors)| {
            neighbors.contains(entity)
                || neighbors.iter().any(|neighbor| {
                    !evaluations.contains_key(neighbor)
                        || !adjacency
                            .get(neighbor)
                            .is_some_and(|reverse| reverse.contains(entity))
                })
        })
    {
        return Err(invalid(
            "metric grading graph",
            "must exactly cover the evaluations with symmetric, known, non-self adjacency",
        ));
    }
    Ok(())
}

fn invalid(field: &str, reason: &str) -> MetricContractError {
    MetricContractError {
        field: field.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{MetricSourceKind, MetricTensor3};
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

    #[test]
    fn grading_propagates_canonical_size_caps_and_records_clipping() {
        let entities = [entity(1), entity(2), entity(3)];
        let adjacency = BTreeMap::from([
            (entities[0].clone(), BTreeSet::from([entities[1].clone()])),
            (
                entities[1].clone(),
                BTreeSet::from([entities[0].clone(), entities[2].clone()]),
            ),
            (entities[2].clone(), BTreeSet::from([entities[1].clone()])),
        ]);
        let mut evaluations = BTreeMap::from([
            (entities[0].clone(), evaluation(0.1)),
            (entities[1].clone(), evaluation(1.0)),
            (entities[2].clone(), evaluation(1.0)),
        ]);

        grade_metric_evaluations(2.0, &adjacency, &mut evaluations).unwrap();

        assert_eq!(evaluations[&entities[0]].clipped_contribution_count, 0);
        assert_eq!(evaluations[&entities[1]].clipped_contribution_count, 1);
        assert_eq!(evaluations[&entities[2]].clipped_contribution_count, 1);
        assert!(
            evaluations[&entities[1]]
                .metric
                .conservative_minimum_length_m()
                .unwrap()
                <= 0.2
        );
        assert!(
            evaluations[&entities[2]]
                .metric
                .conservative_minimum_length_m()
                .unwrap()
                <= 0.4
        );
        assert_eq!(
            evaluations[&entities[2]].active_sources,
            vec![MetricSourceKind::Global]
        );
    }

    #[test]
    fn grading_rejects_asymmetric_or_incomplete_graphs() {
        let left = entity(1);
        let right = entity(2);
        let adjacency = BTreeMap::from([
            (left.clone(), BTreeSet::from([right.clone()])),
            (right.clone(), BTreeSet::new()),
        ]);
        let mut evaluations = BTreeMap::from([(left, evaluation(0.1)), (right, evaluation(1.0))]);
        assert!(grade_metric_evaluations(1.2, &adjacency, &mut evaluations).is_err());
    }

    #[test]
    fn grading_propagates_over_long_paths_from_every_source() {
        let entities = (0..8).map(entity).collect::<Vec<_>>();
        let adjacency = entities
            .iter()
            .enumerate()
            .map(|(index, current)| {
                let mut neighbors = BTreeSet::new();
                if index > 0 {
                    neighbors.insert(entities[index - 1].clone());
                }
                if index + 1 < entities.len() {
                    neighbors.insert(entities[index + 1].clone());
                }
                (current.clone(), neighbors)
            })
            .collect::<BTreeMap<_, _>>();
        let mut evaluations = entities
            .iter()
            .enumerate()
            .map(|(index, current)| {
                let length = match index {
                    0 => 0.01,
                    7 => 0.02,
                    _ => 10.0,
                };
                (current.clone(), evaluation(length))
            })
            .collect::<BTreeMap<_, _>>();

        grade_metric_evaluations(2.0, &adjacency, &mut evaluations).unwrap();

        let expected = [0.01, 0.02, 0.04, 0.08, 0.16, 0.08, 0.04, 0.02];
        for (current, expected) in entities.iter().zip(expected) {
            let actual = evaluations[current]
                .metric
                .conservative_minimum_length_m()
                .unwrap();
            assert!(actual <= expected);
            assert!(actual > expected * 0.99);
        }
    }

    #[test]
    fn unit_grading_ratio_propagates_the_global_minimum() {
        let entities = [entity(1), entity(2), entity(3)];
        let adjacency = BTreeMap::from([
            (entities[0].clone(), BTreeSet::from([entities[1].clone()])),
            (
                entities[1].clone(),
                BTreeSet::from([entities[0].clone(), entities[2].clone()]),
            ),
            (entities[2].clone(), BTreeSet::from([entities[1].clone()])),
        ]);
        let mut evaluations = BTreeMap::from([
            (entities[0].clone(), evaluation(0.25)),
            (entities[1].clone(), evaluation(1.0)),
            (entities[2].clone(), evaluation(2.0)),
        ]);

        grade_metric_evaluations(1.0, &adjacency, &mut evaluations).unwrap();

        let propagated = entities[1..]
            .iter()
            .map(|current| {
                evaluations[current]
                    .metric
                    .conservative_minimum_length_m()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(propagated, vec![0.25, 0.25]);
        let once = evaluations.clone();
        grade_metric_evaluations(1.0, &adjacency, &mut evaluations).unwrap();
        assert_eq!(
            evaluations, once,
            "regrading must not invent additional density"
        );
        assert_eq!(evaluations[&entities[1]].clipped_contribution_count, 1);
        assert_eq!(evaluations[&entities[2]].clipped_contribution_count, 1);
    }

    fn evaluation(length_m: f64) -> ResolvedMetricEvaluation {
        ResolvedMetricEvaluation {
            metric: MetricTensor3::isotropic_length_m(length_m).unwrap(),
            active_sources: vec![MetricSourceKind::Global],
            applied_contribution_count: 0,
            clipped_contribution_count: 0,
            rejected_contribution_count: 0,
        }
    }

    fn entity(seed: u8) -> PersistentEntityId {
        PersistentEntityId::new(
            PersistentEntityKind::Edge,
            format!("edge-{seed}"),
            Vec::new(),
        )
    }
}
