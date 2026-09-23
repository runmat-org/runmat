use runmat_meshing_core::SolverMeshArtifact;

use crate::{constraints::CanonicalStructuralConstraintPlan, execution::StructuralModalWorkload};

use super::{checked_add, checked_bytes, checked_mul, count};
use crate::planning::resource::{
    structural_facts::derive_structural_resource_facts, PreparedOutputResourceFacts,
    ResourcePlanningError,
};

const ARTIFACT_OVERHEAD_BYTES: u64 = 64 * 1024;
const MODAL_RECEIPT_PAYLOAD_BYTES_UPPER_BOUND: u64 = 16 * 1024;

pub(super) fn modal_workspace_bytes(
    full_dimension: u64,
    reduced_dimension: u64,
    residual_vector_count: u64,
    reduced_stiffness_nonzeros: u64,
    reduced_mass_nonzeros: u64,
    ilu_bytes: u64,
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
    let vector_values = checked_add(
        checked_mul(full_dimension, reduced_dimension, "expanded modal vectors")?,
        checked_mul(
            checked_mul(
                reduced_dimension,
                reduced_dimension,
                "reduced modal vectors",
            )?,
            3,
            "reduced modal vector blocks",
        )?,
        "modal vector values",
    )?;
    let dense_values = checked_mul(
        checked_mul(reduced_dimension, reduced_dimension, "dense modal matrices")?,
        3,
        "dense modal matrices",
    )?;
    let residual_workspace_values = if residual_vector_count == 0 {
        0
    } else {
        checked_add(
            checked_mul(full_dimension, 6, "residual-vector full workspace")?,
            checked_mul(reduced_dimension, 8, "residual-vector reduced workspace")?,
            "residual-vector workspace",
        )?
    };
    let vector_and_dense = checked_bytes(
        checked_add(
            checked_add(vector_values, dense_values, "modal numerical workspace")?,
            residual_workspace_values,
            "modal and residual-vector workspace",
        )?,
        8,
        "modal numerical workspace",
    )?;
    checked_add(
        checked_add(
            reduced_operator_bytes,
            ilu_bytes,
            "reduced operators and ILU",
        )?,
        vector_and_dense,
        "modal workspace",
    )
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
        checked_mul(retained_modes, 48, "modal scalar metrics")?,
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
    let values = modal_output_value_count(
        count(reduction.full_dof_count, "modal full dimension")?,
        count(reduction.reduced_dof_count, "modal retained modes")?,
        count(
            workload.residual_vector_directions().len(),
            "modal residual-vector count",
        )?,
    )?;
    modal_result_payload_bytes(values, 0)
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
