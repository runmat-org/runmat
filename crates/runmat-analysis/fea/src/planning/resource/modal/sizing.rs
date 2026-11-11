use runmat_meshing_core::SolverMeshArtifact;

use crate::{constraints::CanonicalStructuralConstraintPlan, execution::StructuralModalWorkload};

use super::{checked_add, checked_bytes, checked_mul, count};
use crate::planning::resource::{
    structural_facts::derive_structural_resource_facts, PreparedOutputResourceFacts,
    ResourcePlanningError,
};

const ARTIFACT_OVERHEAD_BYTES: u64 = 64 * 1024;
const MODAL_RECEIPT_PAYLOAD_BYTES_UPPER_BOUND: u64 = 16 * 1024;

/// `f64` values the shift-invert eigensolver holds for one search of `subspace_dimension`
/// vectors, from the options the solver itself uses.
pub(super) fn modal_eigensolver_working_values(
    reduced_dimension: u64,
    subspace_dimension: u64,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
) -> Result<u64, ResourcePlanningError> {
    let reduced = usize::try_from(reduced_dimension)
        .map_err(|_| ResourcePlanningError::Overflow("modal reduced dimension"))?;
    let subspace = usize::try_from(subspace_dimension)
        .map_err(|_| ResourcePlanningError::Overflow("modal subspace dimension"))?;
    let options = crate::solve::modal::modal_eigen_options(subspace, convergence);
    runmat_sparse_solvers::eigen::modal_working_value_count(&options, reduced)
        .and_then(|values| u64::try_from(values).ok())
        .ok_or(ResourcePlanningError::Overflow(
            "modal eigensolver working set",
        ))
}

/// `f64` values the LOBPCG iteration holds for one search of `subspace_dimension` vectors,
/// from the options the solver itself uses.
pub(super) fn modal_lobpcg_working_values(
    reduced_dimension: u64,
    subspace_dimension: u64,
    convergence: crate::pipeline::modal_governance::ModalConvergencePolicy,
) -> Result<u64, ResourcePlanningError> {
    let reduced = usize::try_from(reduced_dimension)
        .map_err(|_| ResourcePlanningError::Overflow("modal reduced dimension"))?;
    let subspace = usize::try_from(subspace_dimension)
        .map_err(|_| ResourcePlanningError::Overflow("modal subspace dimension"))?;
    let options = crate::solve::modal::modal_lobpcg_options(subspace, convergence);
    runmat_sparse_solvers::eigen::modal_lobpcg_working_value_count(&options, reduced)
        .and_then(|values| u64::try_from(values).ok())
        .ok_or(ResourcePlanningError::Overflow("modal LOBPCG working set"))
}

/// What the eigen-search holds besides its vectors while it iterates, by path.
#[derive(Clone, Copy, Debug)]
pub(super) enum ModalSearchResidency {
    /// The coercivity-proving Cholesky factor (the shift-invert operator), or the local-energy
    /// proof that runs before it and releases its storage when it returns.
    ShiftInvertFactor {
        factor_bytes: u64,
        energy_proof_bytes: u64,
    },
    /// The AMG hierarchy of K that preconditions LOBPCG.
    Multigrid { preconditioner_bytes: u64 },
}

/// Bytes of one modal search, with each store held only for its lifetime:
///
/// 1. the reduced operators (K, M and C) throughout;
/// 2. the eigen-search: the path's factor or preconditioner together with the search vectors
///    (`eigensolver_values`, plus the expanded full-length candidates);
/// 3. the completeness proof after the factor or preconditioner is released: the factor-free
///    inertia count (`inertia_bytes`) beside the retained candidates.
///
/// The peak is the reduced operators plus the larger of phases 2 and 3.
pub(super) fn modal_workspace_bytes(
    full_dimension: u64,
    reduced_dimension: u64,
    subspace_dimension: u64,
    eigensolver_values: u64,
    residual_vector_count: u64,
    reduced_stiffness_nonzeros: u64,
    reduced_mass_nonzeros: u64,
    inertia_bytes: u64,
    residency: ModalSearchResidency,
) -> Result<u64, ResourcePlanningError> {
    let reduced_row_offsets = checked_bytes(
        checked_mul(
            checked_add(reduced_dimension, 1, "reduced row offsets")?,
            3,
            "reduced row offsets",
        )?,
        8,
        "reduced row offsets",
    )?;
    let reduced_entries = checked_add(
        checked_add(
            reduced_stiffness_nonzeros,
            reduced_mass_nonzeros,
            "reduced K+M nonzeros",
        )?,
        reduced_stiffness_nonzeros,
        "reduced K+M+C nonzeros",
    )?;
    let reduced_operator_bytes = checked_add(
        reduced_row_offsets,
        checked_bytes(reduced_entries, 16, "reduced operator entries")?,
        "reduced operators",
    )?;
    let candidates = checked_bytes(
        checked_mul(full_dimension, subspace_dimension, "expanded modal vectors")?,
        8,
        "expanded modal vectors",
    )?;
    let residual_workspace = if residual_vector_count == 0 {
        0
    } else {
        checked_bytes(
            checked_add(
                checked_mul(full_dimension, 6, "residual-vector full workspace")?,
                checked_mul(reduced_dimension, 8, "residual-vector reduced workspace")?,
                "residual-vector workspace",
            )?,
            8,
            "residual-vector workspace",
        )?
    };
    let search_store = match residency {
        ModalSearchResidency::ShiftInvertFactor {
            factor_bytes,
            energy_proof_bytes,
        } => factor_bytes.max(energy_proof_bytes),
        ModalSearchResidency::Multigrid {
            preconditioner_bytes,
        } => preconditioner_bytes,
    };
    let search = checked_add(
        search_store,
        checked_bytes(eigensolver_values, 8, "modal eigensolver working set")?,
        "modal eigen-search",
    )?;
    let phases = search.max(inertia_bytes);
    [
        reduced_operator_bytes,
        phases,
        candidates,
        residual_workspace,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        checked_add(total, value, "modal workspace")
    })
}

pub(super) fn modal_output_value_count(
    full_dimension: u64,
    retained_modes: u64,
    residual_vector_count: u64,
) -> Result<u64, ResourcePlanningError> {
    let modal_vectors = checked_mul(full_dimension, retained_modes, "modal basis vectors")?;
    let square_metrics = checked_mul(retained_modes, retained_modes, "modal square metrics")?;
    [
        modal_vectors,
        checked_mul(square_metrics, 2, "modal orthogonality metrics")?,
        // Basis/directional/eigenspace metrics, result-facing scalar fields and
        // numeric container nodes; map keys remain in the metadata inventory.
        checked_mul(retained_modes, 64, "modal scalar metrics and containers")?,
        checked_mul(
            checked_add(full_dimension, 16, "residual-vector values")?,
            residual_vector_count,
            "modal residual-vector output values",
        )?,
        32,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        checked_add(total, value, "modal output values")
    })
}

pub(super) fn modal_phase_value_count(
    output_values: u64,
    full_dimension: u64,
    retained_modes: u64,
    nodes: u64,
    elements: u64,
) -> Result<u64, ResourcePlanningError> {
    let recovery_per_mode = checked_add(
        checked_mul(nodes, 6, "modal phase nodal recovery")?,
        checked_mul(elements, 13, "modal phase element recovery")?,
        "modal phase recovery",
    )?;
    [
        output_values,
        checked_mul(
            recovery_per_mode,
            retained_modes,
            "all recovered modal fields",
        )?,
        checked_mul(
            checked_mul(nodes, 6, "projected translation and rotation values")?,
            retained_modes,
            "projected modal fields",
        )?,
        checked_mul(full_dimension, retained_modes, "solver modal vectors")?,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        checked_add(total, value, "modal phase values")
    })
}

/// Deterministic canonical payload ceiling for the modal solver-result object.
pub fn structural_modal_result_payload_bytes_upper_bound(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralModalWorkload,
) -> Result<u64, ResourcePlanningError> {
    Ok(
        structural_modal_result_output_resource_facts(mesh, constraints, workload)?
            .encoded_payload_bytes_upper_bound
            .0,
    )
}

/// Shared before execution and independently rederived by the native worker.
pub fn structural_modal_result_output_resource_facts(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralModalWorkload,
) -> Result<PreparedOutputResourceFacts, ResourcePlanningError> {
    workload
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let facts = derive_structural_resource_facts(workload.model(), mesh, constraints)?;
    let reduction = facts
        .constraint_matrix
        .affine_reduction()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    if reduction.offset.iter().any(|value| *value != 0.0) || reduction.reduced_dof_count == 0 {
        return Err(ResourcePlanningError::Invalid(
            "modal payload planning requires a nonempty homogeneous reduction".into(),
        ));
    }
    let retained_modes = count(
        workload
            .convergence()
            .admitted_subspace_dimension(
                reduction.reduced_dof_count,
                workload.truncation().requested_mode_count(),
            )
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        "modal retained modes",
    )?;
    let residual_vectors = count(
        workload.residual_vector_directions().len(),
        "modal residual-vector count",
    )?;
    let values = modal_output_value_count(
        count(reduction.full_dof_count, "modal full dimension")?,
        retained_modes,
        residual_vectors,
    )?;
    super::output::result_output(
        values,
        retained_modes,
        residual_vectors,
        count(reduction.full_dof_count, "modal physical DOF rows")?,
        facts.topology.nodes.0,
    )
}

/// Canonical payload ceiling for the compact fixed-shape numerical receipt.
pub const fn structural_modal_receipt_payload_bytes_upper_bound() -> u64 {
    MODAL_RECEIPT_PAYLOAD_BYTES_UPPER_BOUND
}

pub(super) fn modal_result_payload_bytes(
    values: u64,
    embedded_observation_bytes: u64,
) -> Result<u64, ResourcePlanningError> {
    checked_add(
        checked_add(
            checked_bytes(values, 8, "modal result values")?,
            embedded_observation_bytes,
            "modal result observations",
        )?,
        ARTIFACT_OVERHEAD_BYTES,
        "modal result payload",
    )
}

pub(super) fn prepared_output_covers_transport(output: &PreparedOutputResourceFacts) -> bool {
    output.transport_closure_bytes_upper_bound >= output.encoded_payload_bytes_upper_bound
        && output.construction_resident_bytes_admission_budget
            >= output.encoded_payload_bytes_upper_bound
}

/// Known device-resident lower envelope for the native sparse CG adapter used by modal solves.
pub(in crate::planning::resource) fn structural_modal_accelerator_minimum_bytes(
    reduced_dimension: u64,
    stiffness_nonzeros: u64,
) -> Result<u64, ResourcePlanningError> {
    let row_offsets = checked_bytes(
        checked_add(reduced_dimension, 1, "accelerator row offsets")?,
        8,
        "accelerator row offsets",
    )?;
    let sparse_entries = checked_bytes(stiffness_nonzeros, 16, "accelerator sparse entries")?;
    let diagonal = checked_bytes(reduced_dimension, 8, "accelerator diagonal")?;
    let vectors = checked_bytes(
        checked_mul(
            reduced_dimension,
            super::ACCELERATOR_CG_RESIDENT_VECTOR_COUNT,
            "accelerator vectors",
        )?,
        8,
        "accelerator vectors",
    )?;
    [row_offsets, sparse_entries, diagonal, vectors]
        .into_iter()
        .try_fold(0_u64, |total, value| {
            checked_add(total, value, "modal accelerator")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::modal_governance::ModalConvergencePolicy;

    // G26: the drone arm modal study requests 4 modes on 1,135,827 DOFs (378,609 Tet10 nodes).
    // Its inventories follow the 13-vector working subspace, not the 256-vector search ceiling.
    #[test]
    fn arm_modal_inventories_follow_the_requested_working_subspace() {
        let (full, nodes) = (1_135_827_u64, 378_609_u64);
        let convergence = ModalConvergencePolicy::production();
        let subspace = convergence
            .admitted_subspace_dimension(1_135_827, 4)
            .unwrap() as u64;
        assert_eq!(subspace, 13);
        let output = modal_output_value_count(full, subspace, 0).unwrap();
        assert_eq!(output, full * 13 + 2 * 13 * 13 + 64 * 13 + 32);
        assert_eq!(output, 14_766_953);
        let ceiling = modal_output_value_count(full, 256, 0).unwrap();
        assert_eq!(ceiling, 290_919_200);
        let phase = modal_phase_value_count(output, full, subspace, nodes, 0).unwrap();
        let phase_at_ceiling = modal_phase_value_count(ceiling, full, 256, nodes, 0).unwrap();
        assert!(
            phase * 19 < phase_at_ceiling,
            "{phase} vs {phase_at_ceiling}"
        );

        // The eigensolver's working set comes from the solver's own options: a 61-vector
        // Lanczos basis (max(2·13 + 2·6, 13 + 8·6)) and its restart copy, four blocks of 6,
        // two copies of the 13 Ritz vectors, and four 61 × 61 projected matrices.
        let working = modal_eigensolver_working_values(full, subspace, convergence).unwrap();
        assert_eq!(working, full * (2 * 61 + 4 * 6 + 2 * 13) + 4 * 61 * 61);
        let at_ceiling = modal_eigensolver_working_values(full, 256, convergence).unwrap();
        assert!(working * 9 < at_ceiling, "{working} vs {at_ceiling}");
    }

    // G29: the proving factor is released before the Sturm count, so the plan charges the
    // larger of the two, never their sum.
    #[test]
    fn modal_workspace_follows_the_sequential_lifetime_of_factor_and_proof() {
        let (full, reduced, subspace) = (1_135_827_u64, 1_132_980_u64, 13_u64);
        let (stiffness, mass) = (94_600_000_u64, 31_500_000_u64);
        let factor = 28_374_290_490_u64;
        let inertia = 6_000_000_000_u64;
        let eigen = 70_000_000_u64;
        let workspace = |inertia_bytes, residency| {
            modal_workspace_bytes(
                full,
                reduced,
                subspace,
                eigen,
                0,
                stiffness,
                mass,
                inertia_bytes,
                residency,
            )
            .unwrap()
        };
        let direct = |inertia_bytes| {
            workspace(
                inertia_bytes,
                ModalSearchResidency::ShiftInvertFactor {
                    factor_bytes: factor,
                    energy_proof_bytes: 0,
                },
            )
        };
        let base = direct(0) - factor - eigen * 8;
        // The factor phase dominates: the inertia count adds nothing.
        assert_eq!(direct(inertia), direct(0));
        // A proof larger than the factor phase replaces it; it never adds to it.
        assert_eq!(direct(factor * 2), base + factor * 2);
        assert!(direct(factor) < base + 2 * factor);
        // The LOBPCG path holds the multigrid hierarchy during the search instead.
        let amg = 5_000_000_000_u64;
        let iterative = workspace(
            inertia,
            ModalSearchResidency::Multigrid {
                preconditioner_bytes: amg,
            },
        );
        assert_eq!(iterative, base + (amg + eigen * 8).max(inertia));
    }

    #[test]
    fn small_pencils_reserve_the_dense_fallback() {
        let convergence = ModalConvergencePolicy::production();
        let n = 300_u64;
        let working = modal_eigensolver_working_values(n, 13, convergence).unwrap();
        // The densified pencil, its copy and transpose, and the dense eigenvectors.
        assert!(working >= 5 * n * n, "{working}");
    }
}
