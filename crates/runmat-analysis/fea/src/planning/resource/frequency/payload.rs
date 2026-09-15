use runmat_meshing_core::SolverMeshArtifact;

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{StructuralFrequencyEntryRequest, StructuralFrequencyWorkload},
};

use super::{add, count, multiply, ResourcePlanningError};

const ARTIFACT_OVERHEAD_BYTES: u64 = 64 * 1024;
// Canonical CBOR writes every finite f64 as one type byte followed by its eight-byte payload.
const CANONICAL_F64_BYTES_UPPER_BOUND: u64 = 9;
// Displacement, reaction, applied, inertia, damping, elastic, and equilibrium residual.
const FREQUENCY_ENTRY_COMPLEX_VECTOR_COUNT: u64 = 7;

/// Returns the pre-execution canonical payload ceiling for one physical frequency result.
pub fn structural_frequency_entry_payload_bytes_upper_bound(
    request: &StructuralFrequencyEntryRequest,
) -> Result<u64, ResourcePlanningError> {
    request
        .scope()
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let dof_count = count(
        request.scope().physical_dof_count(),
        "frequency result physical DOFs",
    )?;
    let frequency_count = count(
        request.scope().plan().frequencies_hz().len(),
        "frequency result plan entries",
    )?;
    let scalar_count = multiply(
        multiply(
            dof_count,
            FREQUENCY_ENTRY_COMPLEX_VECTOR_COUNT,
            "frequency result vector values",
        )?,
        2,
        "frequency result complex components",
    )?;
    let vector_bytes = multiply(
        scalar_count,
        CANONICAL_F64_BYTES_UPPER_BOUND,
        "frequency result encoded scalar bytes",
    )?;
    let embedded_plan_bytes = multiply(
        frequency_count,
        CANONICAL_F64_BYTES_UPPER_BOUND,
        "frequency result embedded plan bytes",
    )?;
    add(
        add(
            vector_bytes,
            embedded_plan_bytes,
            "frequency result variable payload",
        )?,
        ARTIFACT_OVERHEAD_BYTES,
        "frequency result payload ceiling",
    )
}

/// Returns the pre-execution canonical payload ceiling for recovered complex fields.
pub fn structural_frequency_recovery_payload_bytes_upper_bound(
    mesh: &SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralFrequencyWorkload,
) -> Result<u64, ResourcePlanningError> {
    let layout = super::structural_frequency_recovery_layout(mesh, constraints, workload)?;
    recovery_payload_bytes_upper_bound(&layout)
}

pub(super) fn recovery_payload_bytes_upper_bound(
    layout: &crate::post::ComplexStructuralFieldLayout,
) -> Result<u64, ResourcePlanningError> {
    let storage_values = count(layout.storage_value_count, "complex recovery output values")?;
    let partition_entity_count = count(
        crate::results::field::RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
        "recovered field partition entity count",
    )?;
    let (source_indices, partition_count) = layout.definitions.iter().try_fold(
        (0_u64, 0_u64),
        |(indices, partitions), definition| {
            let entities = count(
                layout
                    .entity_counts
                    .entity_count(definition.entity_domain()),
                "complex recovery field entities",
            )?;
            let field_partitions = add(
                entities,
                partition_entity_count - 1,
                "complex recovery partition rounding",
            )? / partition_entity_count;
            Ok::<_, ResourcePlanningError>((
                add(indices, entities, "complex recovery source indices")?,
                add(
                    partitions,
                    field_partitions,
                    "complex recovery field partitions",
                )?,
            ))
        },
    )?;
    let scalar_bytes = multiply(
        add(
            storage_values,
            source_indices,
            "complex recovery encoded scalar count",
        )?,
        CANONICAL_F64_BYTES_UPPER_BOUND,
        "complex recovery encoded scalar bytes",
    )?;
    let metadata_records = add(partition_count, 1, "complex recovery metadata records")?;
    add(
        scalar_bytes,
        multiply(
            metadata_records,
            ARTIFACT_OVERHEAD_BYTES,
            "complex recovery partition and manifest metadata",
        )?,
        "complex recovery output payload ceiling",
    )
}
