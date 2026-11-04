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
use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use std::collections::BTreeMap;
use thiserror::Error;

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

/// Sets every secondary node from its primaries (primaries are never secondaries).
pub(crate) fn enforce(values: &mut [f64], relations: &[ScalarNodeRelation]) {
    for relation in relations {
        values[relation.secondary] = relation.offset
            + relation
                .primaries
                .iter()
                .map(|&(node, weight)| weight * values[node])
                .sum::<f64>();
    }
}

/// The residual of the reduced equations, T^T r, in full node order: each secondary node's
/// residual moves to its primaries with the relation weights, and the secondary entry is zero.
pub(crate) fn reduce_residual(residual: &[f64], relations: &[ScalarNodeRelation]) -> Vec<f64> {
    let mut reduced = residual.to_vec();
    for relation in relations {
        let moved = residual[relation.secondary];
        for &(node, weight) in &relation.primaries {
            reduced[node] += weight * moved;
        }
        reduced[relation.secondary] = 0.0;
    }
    reduced
}

/// Relations of a correction that keeps already satisfied relations satisfied.
pub(crate) fn homogeneous(relations: &[ScalarNodeRelation]) -> Vec<ScalarNodeRelation> {
    relations
        .iter()
        .map(|relation| ScalarNodeRelation {
            offset: 0.0,
            ..relation.clone()
        })
        .collect()
}

/// The same relations for an increment u = x - x_ref: each offset absorbs the reference
/// difference between the secondary node and its primaries.
pub(crate) fn about_reference(
    relations: &[ScalarNodeRelation],
    reference: &[f64],
) -> Vec<ScalarNodeRelation> {
    relations
        .iter()
        .map(|relation| ScalarNodeRelation {
            offset: relation.offset
                + relation
                    .primaries
                    .iter()
                    .map(|&(node, weight)| weight * reference[node])
                    .sum::<f64>()
                - reference[relation.secondary],
            ..relation.clone()
        })
        .collect()
}

/// A reduced solve: the full nodal vector and the shared solver's account of it.
pub(crate) struct ScalarSolve {
    pub solution: Vec<f64>,
    /// Relative residual of the reduced system.
    pub relative_residual: f64,
    pub iterations: u32,
    pub method: crate::contracts::SolverMethod,
    pub preconditioner: crate::contracts::SolverPreconditioner,
}

#[derive(Debug, Error)]
pub(crate) enum ScalarSolveError {
    #[error("scalar solve numerical accounting failed: {0}")]
    Usage(#[from] runmat_simulation_usage::LedgerError),
    #[error("{0}")]
    Detail(String),
}

impl From<String> for ScalarSolveError {
    fn from(detail: String) -> Self {
        Self::Detail(detail)
    }
}

fn scalar_count(n: usize) -> Result<OperationCount, ScalarSolveError> {
    Ok(OperationCount::from_operations(
        u64::try_from(n).map_err(|_| LedgerError::CountOverflow)?,
    ))
}

fn scalar_add(
    left: OperationCount,
    right: OperationCount,
) -> Result<OperationCount, ScalarSolveError> {
    Ok(left.checked_add(right).map_err(LedgerError::from)?)
}

fn scalar_mul(
    work: OperationCount,
    repetitions: usize,
) -> Result<OperationCount, ScalarSolveError> {
    Ok(work
        .checked_mul(u64::try_from(repetitions).map_err(|_| LedgerError::CountOverflow)?)
        .map_err(LedgerError::from)?)
}

fn measured_scalar<R>(
    work: OperationCount,
    run: impl FnOnce() -> R,
) -> Result<R, ScalarSolveError> {
    let Some(context) = UsageContext::capture() else {
        return Ok(run());
    };
    let mut reservation = context.ledger().reserve(UsageCategory::Solve, work)?;
    let result = run();
    reservation.complete(work)?;
    Ok(result)
}

/// Solve K u = b on the unknown nodes with `known` values fixed and the relations eliminated.
/// K is symmetric positive (semi)definite and the reduction K_r = T^T K T keeps it so; the
/// shared SPD solver chooses sparse Cholesky or preconditioned conjugate gradients by size.
pub(crate) fn solve(
    stiffness: &RealSparseMatrix,
    rhs: &[f64],
    known: &[Option<f64>],
    relations: &[ScalarNodeRelation],
    relative_tolerance: f64,
    strategy: crate::solve::preconditioner::SpdPreconditionerKind,
) -> Result<ScalarSolve, ScalarSolveError> {
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
    let expansion_work = relations
        .iter()
        .filter(|relation| known[relation.secondary].is_none())
        .try_fold(OperationCount::ZERO, |total, relation| {
            let known_primaries = relation
                .primaries
                .iter()
                .filter(|(node, _)| known[*node].is_some())
                .count();
            scalar_add(total, scalar_mul(scalar_count(known_primaries)?, 2)?)
        })?;
    let expansions: Vec<_> = measured_scalar(expansion_work, || (0..count).map(expand).collect())?;
    let pattern = stiffness.pattern();
    let offsets = pattern.row_offsets();
    let columns = pattern.column_indices();
    let values = stiffness.values();
    let mut rows = vec![BTreeMap::<usize, f64>::new(); unknowns];
    let mut reduced_rhs = vec![0.0; unknowns];
    let mut reduction_work = OperationCount::ZERO;
    for row in 0..count {
        if known[row].is_some() {
            continue;
        }
        let row_terms = expansions[row].0.len();
        reduction_work = scalar_add(reduction_work, scalar_mul(scalar_count(row_terms)?, 2)?)?;
        for k in offsets[row] as usize..offsets[row + 1] as usize {
            let col_terms = expansions[columns[k] as usize].0.len();
            let pair_work = scalar_mul(scalar_count(col_terms)?, 3)?;
            let per_row_term = scalar_add(pair_work, scalar_count(3)?)?;
            reduction_work = scalar_add(reduction_work, scalar_mul(per_row_term, row_terms)?)?;
        }
    }
    measured_scalar(reduction_work, || {
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
                        *rows[r].entry(c).or_insert(0.0) += wr * value * wc;
                    }
                    reduced_rhs[r] -= wr * value * constant;
                }
            }
        }
    })?;
    let mut row_offsets = Vec::with_capacity(unknowns + 1);
    let mut column_indices = Vec::new();
    let mut reduced_values = Vec::new();
    row_offsets.push(0u64);
    for row in rows {
        for (c, v) in row {
            column_indices.push(c as u64);
            reduced_values.push(v);
        }
        row_offsets.push(column_indices.len() as u64);
    }
    let nnz = reduced_values.len();
    let reduced = RealSparseMatrix::try_new(
        crate::algebra::SparsePattern::try_new(
            unknowns as u64,
            unknowns as u64,
            row_offsets,
            column_indices,
        )
        .map_err(|error| ScalarSolveError::Detail(format!("reduced pattern: {error:?}")))?,
        reduced_values,
        crate::algebra::OperatorRole::Stiffness,
        crate::algebra::MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        Vec::new(),
    )
    .map_err(|error| ScalarSolveError::Detail(format!("reduced operator: {error:?}")))?;
    use crate::solve::shared_sparse;
    let strategy = shared_sparse::resolve_spd_strategy(strategy, unknowns, nnz, 1, false);
    let outcome = shared_sparse::solve_spd(shared_sparse::SpdSolveRequest {
        matrix: &reduced,
        rhs: &reduced_rhs,
        initial_guess: None,
        relative_tolerance,
        plan: shared_sparse::ambient_plan(),
        strategy,
        block_size: 1,
        factor: None,
        device: None,
    })?;
    if let Some(failure) = outcome.failure {
        return Err(ScalarSolveError::Detail(failure));
    }
    if outcome.cancelled {
        return Err(ScalarSolveError::Detail(
            "reduced scalar solve cancelled".into(),
        ));
    }
    let relative_residual = if outcome.reference_norm > 0.0 {
        measured_scalar(scalar_count(1)?, || {
            outcome.residual_norm / outcome.reference_norm
        })?
    } else {
        outcome.residual_norm
    };
    if !outcome.converged {
        return Err(ScalarSolveError::Detail(format!(
            "reduced scalar solve did not reach tolerance {relative_tolerance}: relative residual {relative_residual}"
        )));
    }
    let recovery_work = expansions
        .iter()
        .try_fold(OperationCount::ZERO, |total, (terms, _)| {
            let row_work = if terms.is_empty() {
                scalar_count(1)?
            } else {
                scalar_mul(scalar_count(terms.len())?, 2)?
            };
            scalar_add(total, row_work)
        })?;
    let solution = measured_scalar(recovery_work, || {
        (0..count)
            .map(|node| {
                let (terms, constant) = &expansions[node];
                constant
                    + terms
                        .iter()
                        .map(|&(c, w)| w * outcome.solution[c])
                        .sum::<f64>()
            })
            .collect()
    })?;
    let (method, preconditioner) = strategy.provenance();
    Ok(ScalarSolve {
        solution,
        relative_residual,
        iterations: outcome.iterations,
        method,
        preconditioner,
    })
}
