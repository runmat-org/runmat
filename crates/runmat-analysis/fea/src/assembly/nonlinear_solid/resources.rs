//! Nonlinear-only requested storage, not whole-task admission or measured RSS.
//! The caller separately reserves topology, properties, constraints, initial operators,
//! reduction, Krylov workspace, source preparation, observations and transport.

use std::mem::size_of;

use super::*;
mod authority;
use crate::solve::nonlinear::solid::AcceptedSolidState;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NonlinearSolidStorageDemand {
    pub prepared_basis_bytes: u64,
    pub point_history_bytes: u64,
    /// One complete evaluation, excluding its sparse tangent (charged separately).
    pub evaluation_bytes: u64,
    pub additional_tangent_bytes: u64,
    /// Planned assembly: copied support and its one f64 contribution buffer.
    pub tangent_contribution_bytes: u64,
    pub authority_construction_bytes: u64,
    /// Driver snapshots plus concurrently constructed pipeline field value buffers.
    pub retained_output_bytes: u64,
    pub nonlinear_peak_bytes: u64,
    /// Read-only accepted-state validation and original field projection; no Newton solve.
    pub recovery_peak_bytes: u64,
}

impl NonlinearSolidStorageDemand {
    /// Assumes the driver consumes one accepted increment at a time. A retained Vec
    /// of full increments requires a different envelope and is not covered here.
    pub(crate) fn derive(
        plan: &CanonicalSolidIntegrationPlan,
        properties: &CompiledPropertyMapIdentity,
        dof_count: usize,
        maximum_tangent_nonzeros: usize,
        maximum_accepted_increments: usize,
        line_search: bool,
    ) -> Result<Self, NonlinearSolidError> {
        if dof_count == 0 || maximum_accepted_increments == 0 {
            return Err(invalid("nonlinear storage inventories must be positive"));
        }
        let elements = plan.elements().len();
        let points = plan.total_point_count();
        let point_nodes = plan.total_point_node_count();
        let nodes = plan.elements().iter().try_fold(0usize, |n, e| {
            n.checked_add(e.node_count).ok_or_else(overflow)
        })?;
        let prepared_basis_bytes = sum(&[
            size_of::<CanonicalNonlinearSolidBasis>() as u64,
            slots::<PreparedElement>(elements)?,
            slots::<[usize; 3]>(nodes)?,
            slots::<PreparedBodyNode>(nodes)?,
            slots::<PreparedPoint>(points)?,
            slots::<[f64; 3]>(point_nodes)?,
        ])?;
        let point_history_bytes = sum(&[
            size_of::<NonlinearSolidState>() as u64,
            slots::<NonlinearSolidPointIdentity>(points)?,
            slots::<PlasticityPointState>(points)?,
        ])?;
        let evaluation_bytes = sum(&[
            size_of::<NonlinearSolidEvaluation>() as u64,
            slots::<f64>(dof_count)?,
            point_history_bytes,
            slots::<NonlinearSolidPointEvidence>(points)?,
            slots::<NonlinearSolidElementEvidence>(elements)?,
        ])?;
        let csr = sum(&[
            size_of::<RealSparseMatrix>() as u64,
            slots::<u64>(dof_count.checked_add(1).ok_or_else(overflow)?)?,
            slots::<u64>(maximum_tangent_nonzeros)?,
            slots::<f64>(maximum_tangent_nonzeros)?,
            1024, // contribution vector/string and fixed sparse authority metadata
        ])?;
        // Baseline operator planning charges one current tangent; candidate and
        // accepted validation can coexist with it. No Krylov storage is included.
        let additional_tangent_bytes = mul(csr, 2)?;
        // One retained complete support, plus planned assembly's support/value copy.
        // No allocator-private tree-node size is inferred from CSR nonzeros.
        let tangent_contribution_bytes = sum(&[
            slots::<u64>(dof_count.checked_add(1).ok_or_else(overflow)?)?,
            slots::<u64>(maximum_tangent_nonzeros)?,
            csr,
        ])?;
        let authority_construction_bytes =
            authority::construction_bytes(plan, properties, dof_count)?;
        // Driver: u/internal force, 6 stress/6 total strain/6 plastic/alpha/mises per element,
        // load/residual/correction/iteration scalar inventories. Pipeline adds
        // projected translations/rotations, field clones and history-source
        // clones, with element-field clones and scalar field construction.
        let values_per_increment = sum(&[mul(dof_count as u64, 8)?, mul(elements as u64, 28)?, 7])?;
        let capacity = maximum_accepted_increments
            .checked_mul(2)
            .and_then(|n| n.checked_add(4))
            .ok_or_else(overflow)?;
        let retained_output_bytes = sum(&[
            mul(mul(values_per_increment, capacity as u64)?, 8)?,
            // 8 constructed fields and 10 raw inner-vector inventories, including
            // explicitly empty contact vectors, plus vector/scalar outer slots.
            mul(capacity as u64, 18 * 256)?,
            slots::<f64>(dof_count)?,
            slots::<f64>(elements)?,
        ])?;
        let state = sum(&[
            size_of::<AcceptedSolidState>() as u64,
            slots::<f64>(dof_count)?,
            point_history_bytes,
        ])?;
        let nonlinear_peak_bytes = sum(&[
            prepared_basis_bytes,
            mul(state, 4)?, // driver current, retained carrier, trial and validation
            mul(evaluation_bytes, if line_search { 3 } else { 2 })?,
            additional_tangent_bytes,
            tangent_contribution_bytes,
            authority_construction_bytes,
            retained_output_bytes,
            slots::<f64>(dof_count.checked_mul(8).ok_or_else(overflow)?)?,
        ])?;
        // Original carrier plus its recovery clone; physical validation and terminal
        // projection can each construct an evaluation. Sparse support/authority and
        // all output capacities stay covered, including eight scratch vectors.
        let recovery_peak_bytes = sum(&[
            prepared_basis_bytes,
            mul(state, 2)?,
            mul(evaluation_bytes, 2)?,
            additional_tangent_bytes,
            tangent_contribution_bytes,
            authority_construction_bytes,
            retained_output_bytes,
            slots::<f64>(dof_count.checked_mul(8).ok_or_else(overflow)?)?,
        ])?;
        Ok(Self {
            prepared_basis_bytes,
            point_history_bytes,
            evaluation_bytes,
            additional_tangent_bytes,
            tangent_contribution_bytes,
            authority_construction_bytes,
            retained_output_bytes,
            nonlinear_peak_bytes,
            recovery_peak_bytes,
        })
    }
}

fn slots<T>(count: usize) -> Result<u64, NonlinearSolidError> {
    mul(count as u64, size_of::<T>() as u64)
}
fn mul(a: u64, b: u64) -> Result<u64, NonlinearSolidError> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn sum(parts: &[u64]) -> Result<u64, NonlinearSolidError> {
    parts
        .iter()
        .try_fold(0u64, |a, b| a.checked_add(*b).ok_or_else(overflow))
}
fn invalid(reason: &str) -> NonlinearSolidError {
    NonlinearSolidError::Authority(reason.into())
}
fn overflow() -> NonlinearSolidError {
    invalid("nonlinear requested storage overflow")
}

#[cfg(test)]
mod tests;
