//! Node relations for scalar diffusion (electric potential, temperature): periodic pairs,
//! ties across nonmatching faces and known jumps across them.
//!
//! Each relation holds a secondary node at a weighted combination of primary nodes plus a
//! known offset, u_s = sum_k w_k u_pk + c (the weights come from the shared pairing
//! projection, so nonmatching meshes are tied through the primary face's interpolation). The
//! solve eliminates secondary nodes by Galerkin reduction, K_r = T^T K T, which keeps a
//! symmetric operator symmetric, and expands the reduced solution back to every node.
use runmat_analysis_core::{PairedConstraint, PeriodicTransform, RegionTarget};
use runmat_meshing_core::SolverMeshArtifact;
use std::collections::BTreeMap;

use crate::algebra::RealSparseMatrix;
use crate::topology::AnalysisProjection;

/// u_secondary = sum(weight * u_primary) + offset.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScalarNodeRelation {
    pub secondary: usize,
    pub primaries: Vec<(usize, f64)>,
    pub offset: f64,
}

/// One authored relation between two face sets.
pub(crate) struct ScalarPairing<'a> {
    pub constraint_id: &'a str,
    pub pair: &'a PairedConstraint,
    pub transform: &'a PeriodicTransform,
    /// u_secondary - u_primary (zero for periodic and tie relations).
    pub offset: f64,
}

pub(crate) fn relations(
    mesh: &SolverMeshArtifact,
    projection: &AnalysisProjection,
    pairings: &[ScalarPairing<'_>],
) -> Result<Vec<ScalarNodeRelation>, String> {
    let index: BTreeMap<_, _> = projection
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.stable_identity, i))
        .collect();
    let mut relations = BTreeMap::<usize, ScalarNodeRelation>::new();
    for pairing in pairings {
        let resolve = |target: &RegionTarget| {
            crate::constraints::resolve_nodal_target(mesh, target, pairing.constraint_id)
                .map_err(|error| error.to_string())
        };
        let primary = resolve(&pairing.pair.primary)?;
        let secondary = resolve(&pairing.pair.secondary)?;
        let evidence = crate::constraints::pair_targets(
            mesh,
            &primary,
            &secondary,
            &pairing.pair.pairing,
            pairing.transform,
            pairing.constraint_id,
        )
        .map_err(|error| error.to_string())?;
        for pair in evidence {
            let secondary = *index
                .get(&pair.secondary_node_identity)
                .ok_or("paired node is absent from the projection")?;
            let primaries = pair
                .primary_weights
                .iter()
                .map(|weight| {
                    index
                        .get(&weight.primary_node_identity)
                        .map(|&node| (node, weight.weight))
                        .ok_or("paired node is absent from the projection")
                })
                .collect::<Result<Vec<_>, _>>()?;
            // A node on both faces of a pair relates to itself; it needs no relation.
            if primaries.len() == 1 && primaries[0].0 == secondary {
                continue;
            }
            let relation = ScalarNodeRelation {
                secondary,
                primaries,
                offset: pairing.offset,
            };
            if let Some(existing) = relations.insert(secondary, relation.clone()) {
                if existing != relation {
                    return Err(format!(
                        "node {} is the secondary of two different relations",
                        projection.nodes[secondary].node_id
                    ));
                }
            }
        }
    }
    for relation in relations.values() {
        if relation
            .primaries
            .iter()
            .any(|(node, _)| relations.contains_key(node))
        {
            return Err(
                "a relation's primary node is itself a secondary node; chained relations are not resolved".into(),
            );
        }
    }
    Ok(relations.into_values().collect())
}

/// Solve K u = b on the unknown nodes with `known` values fixed and the relations eliminated.
/// Returns the full nodal vector and the relative residual of the reduced system.
pub(crate) fn solve(
    stiffness: &RealSparseMatrix,
    rhs: &[f64],
    known: &[Option<f64>],
    relations: &[ScalarNodeRelation],
) -> Result<(Vec<f64>, f64), String> {
    let count = known.len();
    let by_secondary: BTreeMap<_, _> = relations.iter().map(|r| (r.secondary, r)).collect();
    let mut column = vec![usize::MAX; count];
    let mut unknowns = 0;
    for node in 0..count {
        if known[node].is_none() && !by_secondary.contains_key(&node) {
            column[node] = unknowns;
            unknowns += 1;
        }
    }
    // Each node as (reduced columns with weights, constant part).
    let expand = |node: usize| -> (Vec<(usize, f64)>, f64) {
        if let Some(value) = known[node] {
            return (Vec::new(), value);
        }
        if let Some(relation) = by_secondary.get(&node) {
            let mut terms = Vec::new();
            let mut constant = relation.offset;
            for &(primary, weight) in &relation.primaries {
                match known[primary] {
                    Some(value) => constant += weight * value,
                    None => terms.push((column[primary], weight)),
                }
            }
            return (terms, constant);
        }
        (vec![(column[node], 1.0)], 0.0)
    };
    let expansions: Vec<_> = (0..count).map(expand).collect();
    let pattern = stiffness.pattern();
    let offsets = pattern.row_offsets();
    let columns = pattern.column_indices();
    let values = stiffness.values();
    let mut entries = Vec::new();
    let mut reduced_rhs = vec![0.0; unknowns];
    for row in 0..count {
        if known[row].is_some() {
            continue;
        }
        let (row_terms, _) = &expansions[row];
        for &(r, wr) in row_terms {
            reduced_rhs[r] += wr * rhs[row];
        }
        for k in offsets[row] as usize..offsets[row + 1] as usize {
            let col = columns[k] as usize;
            let value = values[k];
            let (col_terms, constant) = &expansions[col];
            for &(r, wr) in row_terms {
                for &(c, wc) in col_terms {
                    entries.push((r, c, wr * value * wc));
                }
                reduced_rhs[r] -= wr * value * constant;
            }
        }
    }
    let solved = crate::solve::sparse_direct::solve_real(unknowns, &entries, &reduced_rhs)
        .map_err(|error| error.to_string())?;
    let solution = (0..count)
        .map(|node| {
            let (terms, constant) = &expansions[node];
            constant + terms.iter().map(|&(c, w)| w * solved.solution[c]).sum::<f64>()
        })
        .collect();
    Ok((solution, solved.relative_residual))
}
