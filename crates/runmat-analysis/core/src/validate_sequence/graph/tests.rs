use super::*;
use crate::{ProcedureStepId, SequenceDependency, SequenceDependencyReason};

fn sequence(count: usize, edges: &[(usize, usize)]) -> AnalysisSequence {
    let mut sequence = AnalysisSequence::from_steps(
        "graph-reference",
        (0..count)
            .map(|i| ProcedureStep::structural_static(format!("step-{i}")))
            .collect(),
    );
    sequence.dependencies = edges
        .iter()
        .map(|&(a, b)| SequenceDependency {
            predecessor: ProcedureStepId(format!("step-{a}")),
            successor: ProcedureStepId(format!("step-{b}")),
            reason: SequenceDependencyReason::AuthoredOrdering,
        })
        .collect();
    sequence
}

#[test]
fn original_ancestry_matches_independent_boolean_matrix_closure_under_storage_permutation() {
    const N: usize = 24;
    for seed in 0..16 {
        let edges: Vec<_> = (0..N)
            .flat_map(|a| {
                (a + 1..N)
                    .filter(move |&b| (a * 17 + b * 31 + seed * 13) % 11 < 3)
                    .map(move |b| (a, b))
            })
            .collect();
        // Independent Warshall reference; production uses reverse adjacency search.
        let mut reach = [[false; N]; N];
        for &(a, b) in &edges {
            reach[a][b] = true;
        }
        for k in 0..N {
            for a in 0..N {
                for b in 0..N {
                    reach[a][b] |= reach[a][k] && reach[k][b];
                }
            }
        }
        let mut original = sequence(N, &edges);
        for permutation in 0..2 {
            if permutation == 1 {
                original.steps.reverse();
                original.dependencies.reverse();
            }
            let graph = validate(&original).unwrap();
            for (a, row) in reach.iter().enumerate() {
                for (b, &expected) in row.iter().enumerate() {
                    assert_eq!(
                        graph.is_ancestor(&format!("step-{a}"), &format!("step-{b}")),
                        expected,
                        "seed={seed}/{a}->{b}"
                    );
                }
            }
            assert!(!graph.is_ancestor("absent", "step-0"));
            assert!(!graph.is_ancestor("step-0", "absent"));
        }
    }
}

#[test]
fn deep_chain_validation_and_late_cycle_rejection_use_no_recursive_stack() {
    std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(|| {
            let count = 8192;
            let edges: Vec<_> = (0..count - 1).map(|i| (i, i + 1)).collect();
            let mut original = sequence(count, &edges);
            {
                let graph = validate(&original).unwrap();
                assert_eq!(graph.predecessors.neighbors.len(), count - 1);
                assert!(graph.is_ancestor("step-0", "step-8191"));
                assert!(!graph.is_ancestor("step-8191", "step-0"));
            }
            original.dependencies.push(SequenceDependency {
                predecessor: ProcedureStepId("step-8191".into()),
                successor: ProcedureStepId("step-0".into()),
                reason: SequenceDependencyReason::AuthoredOrdering,
            });
            assert!(matches!(
                validate(&original),
                Err(AnalysisValidationError::InvalidSequence { .. })
            ));
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn declaration_error_priority_preserves_original_order() {
    let mut original = sequence(4, &[(0, 1), (0, 1)]);
    original.steps.push(ProcedureStep::structural_static(" "));
    original.steps.insert(2, original.steps[0].clone());
    let error = validate(&original).err().unwrap();
    assert!(
        matches!(error, AnalysisValidationError::InvalidProcedureStep { detail, .. } if detail == "duplicate step id `step-0`")
    );
    original.steps.remove(2);
    original.steps.pop();
    original.dependencies.push(SequenceDependency {
        predecessor: ProcedureStepId("absent".into()),
        successor: ProcedureStepId("step-3".into()),
        reason: SequenceDependencyReason::AuthoredOrdering,
    });
    let error = validate(&original).err().unwrap();
    assert!(
        matches!(error, AnalysisValidationError::InvalidSequence { detail, .. } if detail == "duplicate dependency `step-0` -> `step-1`")
    );
}
