use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;

use crate::metric::{MetricContractError, MetricTensor3, ResolvedMetricEvaluation};

/// Applies a conservative Lipschitz-like size bound over a canonical entity-adjacency graph.
///
/// A fine metric may constrain a graph neighbor to at most `maximum_grading_ratio` times its
/// characteristic size. Tightening is isotropic, so it cannot weaken an existing anisotropic
/// tensor in any direction. The result is independent of map construction and traversal order.
pub fn grade_metric_evaluations(
    maximum_grading_ratio: f64,
    adjacency: &BTreeMap<PersistentEntityId, BTreeSet<PersistentEntityId>>,
    evaluations: &mut BTreeMap<PersistentEntityId, ResolvedMetricEvaluation>,
) -> Result<(), MetricContractError> {
    if !maximum_grading_ratio.is_finite() || maximum_grading_ratio < 1.0 {
        return Err(invalid(
            "maximum metric grading ratio",
            "must be finite and at least one",
        ));
    }
    validate_graph(adjacency, evaluations)?;
    let mut size_caps = evaluations
        .iter()
        .map(|(entity, evaluation)| {
            Ok((
                entity.clone(),
                evaluation.metric.conservative_minimum_length_m()?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>, MetricContractError>>()?;
    let original_caps = size_caps.clone();

    // Bellman-Ford relaxation computes min(source_size * ratio^graph_distance). A simple
    // canonical pass is preferable here to queue ordering that could expose floating ties.
    for _ in 1..size_caps.len() {
        let previous = size_caps.clone();
        let mut changed = false;
        for (entity, neighbors) in adjacency {
            let source = previous[entity];
            for neighbor in neighbors {
                let candidate = source * maximum_grading_ratio;
                if candidate.is_finite() && candidate < size_caps[neighbor] {
                    size_caps.insert(neighbor.clone(), candidate);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    for (entity, size_cap) in size_caps {
        if size_cap < original_caps[&entity] {
            let grading_metric = MetricTensor3::isotropic_length_m(size_cap)?;
            let evaluation = evaluations
                .get_mut(&entity)
                .expect("validated graph entity");
            evaluation.metric = add(evaluation.metric, grading_metric)?;
            evaluation.clipped_contribution_count = evaluation
                .clipped_contribution_count
                .checked_add(1)
                .ok_or_else(|| invalid("metric grading evidence", "count overflowed"))?;
        }
    }
    Ok(())
}

fn validate_graph(
    adjacency: &BTreeMap<PersistentEntityId, BTreeSet<PersistentEntityId>>,
    evaluations: &BTreeMap<PersistentEntityId, ResolvedMetricEvaluation>,
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

fn add(left: MetricTensor3, right: MetricTensor3) -> Result<MetricTensor3, MetricContractError> {
    let metric = MetricTensor3 {
        xx: left.xx + right.xx,
        yy: left.yy + right.yy,
        zz: left.zz + right.zz,
        xy: left.xy + right.xy,
        xz: left.xz + right.xz,
        yz: left.yz + right.yz,
    };
    metric.validate()?;
    Ok(metric)
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
    use crate::metric::MetricSourceKind;
    use runmat_geometry_core::PersistentEntityKind;

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
        PersistentEntityId {
            kind: PersistentEntityKind::Edge,
            source_topology_id: format!("edge-{seed}"),
            assembly_path: Vec::new(),
        }
    }
}
