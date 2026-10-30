//! Pre-quadrature magnetic CPU admission using shared numerical count preflight,
//! exact sparse edge/gauge support, and codec wire-shape projections. Requested
//! heap is conservative additive lifetime accounting, never a measured RSS claim.
use super::*;
use crate::{execution::MagnetostaticWorkload, physics::magnetic::MagneticSystemPlan};
use runmat_canonical_codec::CanonicalLimits;
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
mod payload;
#[cfg(test)]
mod tests;
const LIMITS: CanonicalLimits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedMagnetostaticResourceFacts {
    pub local_execution: super::LocalExecutionPlan,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}
fn invalid(error: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(error.to_string())
}
fn count(n: usize) -> Result<u64, ResourcePlanningError> {
    u64::try_from(n).map_err(|_| ResourcePlanningError::Overflow("magnetic count"))
}
fn add(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_add(b)
        .ok_or(ResourcePlanningError::Overflow("magnetic sum"))
}
fn mul(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_mul(b)
        .ok_or(ResourcePlanningError::Overflow("magnetic product"))
}
fn preflight(
    w: &MagnetostaticWorkload,
    m: &SolverMeshArtifact,
) -> Result<crate::physics::magnetic::resources::MagneticNumericalReservation, ResourcePlanningError>
{
    w.validate().map_err(invalid)?;
    crate::physics::magnetic::resources::require_canonical_ceiling(
        &w.model,
        m,
        w.model.sequence.steps[0]
            .accuracy
            .resources
            .resident_byte_budget(),
    )
    .map_err(invalid)
}
/// Complete retained result wire, including explicit oriented edges, fifteen-point
/// topology, physical samples, and all original persisted declaration identifiers.
pub fn magnetostatic_result_payload_bytes_upper_bound(
    w: &MagnetostaticWorkload,
    m: &SolverMeshArtifact,
) -> Result<u64, ResourcePlanningError> {
    preflight(w, m)?;
    Ok(payload::profile(w, m)?.encoded_bytes_upper_bound)
}
pub fn build_magnetostatic_resource_plan(
    w: &MagnetostaticWorkload,
    m: &SolverMeshArtifact,
    prepared: PreparedMagnetostaticResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    Ok(build_magnetostatic_resource_plan_with_system(w, m, prepared)?.0)
}

/// The plan together with the compiled input and derived sparse system it was built from:
/// the worker solves on exactly these instead of deriving the topology again.
pub(crate) fn build_magnetostatic_resource_plan_with_system(
    w: &MagnetostaticWorkload,
    m: &SolverMeshArtifact,
    prepared: PreparedMagnetostaticResourceFacts,
) -> Result<
    (
        FeaResourcePlan,
        crate::physics::magnetic::CompiledMagneticInput,
        crate::physics::magnetic::DerivedMagneticSystem,
    ),
    ResourcePlanningError,
> {
    let numerical = preflight(w, m)?;
    let payload = payload::profile(w, m)?;
    let profiles = w
        .state_publication_profiles_encoding_profile(Digest::from_bytes(
            *m.canonical_digest.bytes(),
        ))
        .map_err(invalid)?;
    let input_bytes = add(
        count(w.canonical_bytes().map_err(invalid)?.len())?,
        count(m.canonical_encode().map_err(invalid)?.len())?,
    )?;
    if prepared.inputs.encoded_payload_bytes.0 != input_bytes
        || prepared.result_output.encoded_payload_bytes_upper_bound.0
            != payload.encoded_bytes_upper_bound
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0
            != crate::execution::MAGNETIC_RECEIPT_MAX_ENCODED_BYTES
        || prepared
            .state_profiles_output
            .encoded_payload_bytes_upper_bound
            .0
            != profiles.encoded_bytes_upper_bound
    {
        return Err(invalid(
            "magnetic canonical inputs or projected complete outputs disagree",
        ));
    }
    // Count-only ceiling is checked before either shared geometric projection or
    // sparse plan discovery can allocate. No physical matrices/fields are built.
    let input = crate::physics::magnetic::compile_canonical_input(
        &w.model,
        m,
        w.step_id.as_str(),
        w.relative_residual_target,
    )
    .map_err(invalid)?;
    let derived = MagneticSystemPlan::derive_with_spaces(&input.projection, &input.grounded_faces)
        .map_err(invalid)?;
    let system = &derived.plan;
    let krylov = estimate::magnetic_static::krylov_envelope(
        system.saddle_pattern().rows(),
        system.saddle_pattern().column_indices().len() as u64,
    )?;
    // The numeric preflight already sums support/properties/planned assembly,
    // matrices, shared GMRES and retained sampling/result-clone lifetimes. Add
    // encoded result scratch and exact result wire materialization explicitly.
    let reservation = add(
        numerical.total_bytes,
        add(
            mul(payload.encoded_bytes_upper_bound, 2)?,
            payload.materialization_resident_bytes_estimate,
        )?,
    )?;
    let source = w.identity().map_err(invalid)?;
    let step = &w.model.sequence.steps[0];
    let plan = build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::Magnetostatic(source),
        demand: FeaResourceDemand::Magnetostatic(MagnetostaticResourceDemand {
            workload: source,
            topology_digest: Digest::from_bytes(*m.canonical_digest.bytes()),
            system_plan_identity: system.identity().map_err(invalid)?,
            node_count: Count(count(m.topology.nodes.len())?),
            element_count: Count(count(m.topology.volume_elements.len())?),
            edge_count_upper_bound: Count(numerical.edge_count_upper_bound),
            quadrature_sample_count: Count(numerical.quadrature_sample_count),
            free_edge_count: Count(count(system.free_edge_count())?),
            gauge_node_count: Count(count(system.gauge_node_count())?),
            physical_nonzeros: Count(count(system.physical_pattern().column_indices().len())?),
            saddle_nonzeros: Count(count(system.saddle_pattern().column_indices().len())?),
            krylov,
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
    })?;
    Ok((plan, input, derived))
}
