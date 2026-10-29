//! Count/profile preflight then exact material-independent complex sparse support.
//! Conservative additive requested heap, never measured RSS or solver acceptance.
use super::*;
use crate::{execution::MaxwellHarmonicWorkload, physics::maxwell::MaxwellSystemPlan};
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
mod payload;
#[cfg(test)]
mod tests;
const LIMITS: CanonicalLimits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedMaxwellHarmonicResourceFacts {
    pub local_execution: super::LocalExecutionPlan,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}
fn invalid(e: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(e.to_string())
}
fn count(n: usize) -> Result<u64, ResourcePlanningError> {
    u64::try_from(n).map_err(|_| ResourcePlanningError::Overflow("Maxwell count"))
}
fn add(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_add(b)
        .ok_or(ResourcePlanningError::Overflow("Maxwell sum"))
}
fn mul(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_mul(b)
        .ok_or(ResourcePlanningError::Overflow("Maxwell product"))
}
fn preflight(
    w: &MaxwellHarmonicWorkload,
    m: &SolverMeshArtifact,
) -> Result<crate::physics::maxwell::resources::MaxwellNumericalReservation, ResourcePlanningError>
{
    // Only borrow the selected policy before original model validation/projection.
    let mut steps = w
        .model
        .sequence
        .steps
        .iter()
        .filter(|s| s.step_id == w.step_id);
    let step = steps
        .next()
        .ok_or_else(|| invalid("selected Maxwell step absent"))?;
    if steps.next().is_some() {
        return Err(invalid("ambiguous Maxwell step"));
    }
    let numerical = crate::physics::maxwell::resources::require_canonical_ceiling(
        &w.model,
        m,
        step.accuracy.resources.resident_byte_budget(),
    )
    .map_err(invalid)?;
    w.validate().map_err(invalid)?;
    Ok(numerical)
}
pub fn maxwell_harmonic_result_payload_bytes_upper_bound(
    w: &MaxwellHarmonicWorkload,
    m: &SolverMeshArtifact,
) -> Result<u64, ResourcePlanningError> {
    preflight(w, m)?;
    Ok(payload::profile(w, m)?.encoded_bytes_upper_bound)
}
pub fn build_maxwell_harmonic_resource_plan(
    w: &MaxwellHarmonicWorkload,
    m: &SolverMeshArtifact,
    prepared: PreparedMaxwellHarmonicResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let numerical = preflight(w, m)?;
    let payload = payload::profile(w, m)?;
    let profiles = w
        .state_publication_profiles_encoding_profile(Digest::from_bytes(
            *m.canonical_digest.bytes(),
        ))
        .map_err(invalid)?;
    // Profiles precede actual canonical encodings; no field/source arrays built.
    let workload_profile = profile_serialized(w, LIMITS).map_err(invalid)?;
    let mesh_profile = profile_serialized(m, LIMITS).map_err(invalid)?;
    let workload_bytes = w.canonical_bytes().map_err(invalid)?;
    let mesh_bytes = m.canonical_encode().map_err(invalid)?;
    let input_bytes = add(count(workload_bytes.len())?, count(mesh_bytes.len())?)?;
    // Prepared transport budgets describe the actual canonical bytes, whereas
    // borrowed Serde profiles conservatively reserve maximum-width encodings.
    // Compare the same actual-wire authority; retain the larger profiles below.
    let actual_decode = add(
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&workload_bytes)
            .map_err(invalid)?,
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&mesh_bytes)
            .map_err(invalid)?,
    )?;
    if prepared.inputs.encoded_payload_bytes.0 != input_bytes
        || prepared.result_output.encoded_payload_bytes_upper_bound.0
            != payload.encoded_bytes_upper_bound
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0
            != crate::execution::MAXWELL_RECEIPT_MAX_ENCODED_BYTES
        || prepared
            .state_profiles_output
            .encoded_payload_bytes_upper_bound
            .0
            != profiles.encoded_bytes_upper_bound
        || prepared.inputs.decoded_resident_bytes_admission_budget.0 < actual_decode
    {
        return Err(invalid(
            "Maxwell original inputs or projected outputs disagree",
        ));
    }
    let input = w.compile_input(m).map_err(invalid)?;
    let system = MaxwellSystemPlan::derive(m, &input.grounded_faces).map_err(invalid)?;
    let current_drives = input
        .drives
        .iter()
        .filter(|d| {
            matches!(
                d.excitation,
                crate::physics::maxwell::MaxwellDriveExcitation::Current { .. }
            )
        })
        .count();
    drop(input);
    let dimension = add(
        count(system.free_edge_indices().len())?,
        count(current_drives)?,
    )?;
    let nonzeros = count(system.pattern().column_indices().len())?;
    let factorization = estimate::maxwell_harmonic::factor_envelope(dimension, nonzeros)?;
    let reservation = add(
        add(
            numerical.total_bytes,
            add(
                workload_profile.materialization_resident_bytes_estimate,
                mesh_profile.materialization_resident_bytes_estimate,
            )?,
        )?,
        add(
            mul(payload.encoded_bytes_upper_bound, 2)?,
            payload.materialization_resident_bytes_estimate,
        )?,
    )?;
    let source = w.identity().map_err(invalid)?;
    let step = &w.model.sequence.steps[0];
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::MaxwellHarmonic(source),
        demand: FeaResourceDemand::MaxwellHarmonic(MaxwellHarmonicResourceDemand {
            workload: source,
            topology_digest: Digest::from_bytes(*m.canonical_digest.bytes()),
            system_plan_identity: system.identity().map_err(invalid)?,
            node_count: Count(count(m.topology.nodes.len())?),
            element_count: Count(count(m.topology.volume_elements.len())?),
            edge_count_upper_bound: Count(numerical.edge_count_upper_bound),
            quadrature_sample_count: Count(numerical.quadrature_sample_count),
            free_edge_count: Count(count(system.free_edge_indices().len())?),
            complex_dimension: Count(dimension),
            nonzeros: Count(nonzeros),
            factorization,
            numerical_reservation_bytes: ByteCount(reservation),
            inputs: prepared.inputs,
            result_output: prepared.result_output,
            receipt_output: prepared.receipt_output,
            state_profiles_output: prepared.state_profiles_output,
            state_profiles_resident_bytes: ByteCount(
                profiles.materialization_resident_bytes_estimate,
            ),
        }),
        output_set: Some(prepared.output_set),
        worker: WorkerResourcePolicy {
            cpu_millicores: prepared
                .local_execution
                .checked_cpu_millicores()
                .map_err(invalid)?,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(16 * 1024 * 1024),
            fixed_scratch_bytes: ByteCount(8 * 1024 * 1024),
            required_capabilities: Default::default(),
            accelerators: Vec::new(),
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}
