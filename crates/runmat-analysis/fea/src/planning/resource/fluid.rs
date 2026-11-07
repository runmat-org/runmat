//! Source-authenticated steady finite-volume resource facts for scheduler admission.
//!
//! These are conservative requested-heap bounds, not measured RSS, a replacement
//! numerical reservation, or evidence of distributed/GPU execution. The shared
//! phase/demand contract consumes this projection; projection alone is not a plan.
use super::{ByteCount, PreparedPhaseResourceFacts, ResourcePlanningError};
use crate::{
    execution::{AnalysisWorkloadIdentity, FluidSteadyWorkload, StepIdentity},
    physics::cfd::canonical::CanonicalFluidReservation,
    properties::CompiledPropertyMapIdentity,
};
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

mod admission;
mod governed;
mod recovery;
pub use governed::{
    build_fluid_steady_resource_plan, build_fluid_steady_resource_plan_with,
    fluid_steady_result_payload_bytes_upper_bound, PreparedFluidSteadyResourceFacts,
};
pub use recovery::{
    build_fluid_field_recovery_resource_plan, build_fluid_field_recovery_resource_plan_on,
    fluid_field_recovery_shape,
};
#[cfg(test)]
mod tests;

const LIMITS: CanonicalLimits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);

/// Derived only from the original workload, exact mesh, source property compiler,
/// and execution-owned result wire profile. No solve is run to derive these facts.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidSteadyResourceProjection {
    pub workload: AnalysisWorkloadIdentity,
    pub producer: StepIdentity,
    pub topology_digest: Digest,
    pub projection_digest: Digest,
    pub geometry_digest: Digest,
    pub properties: CompiledPropertyMapIdentity,
    pub property_audit_digest: Digest,
    pub numerical: CanonicalFluidReservation,
    pub source_preparation_resident_bytes: ByteCount,
    /// Finite-volume derivation scratch, released before numerical storage exists.
    pub source_preparation_scratch_bytes: ByteCount,
    pub canonical_input_bytes: ByteCount,
    pub actual_decode_resident_bytes: ByteCount,
    pub result_payload_bytes_upper_bound: ByteCount,
    pub result_materialization_resident_bytes: ByteCount,
    /// Retained source preparation and decoded inputs, plus the largest of the sequential
    /// derivation scratch, numerical solve and result encoding lifetimes (see
    /// `admission::phase_resident_bytes`); scheduler fixed/transport costs are separate.
    pub phase_resident_bytes_upper_bound: ByteCount,
    /// The same lifetimes with the numerical verification bound in place of the solve
    /// reserve: re-preparing and verifying a retained result runs no linear solve.
    pub verification_phase_resident_bytes_upper_bound: ByteCount,
}

fn invalid(error: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(error.to_string())
}

/// Derive authentic local support and wire facts before scheduler task construction.
/// `prepare` admits its owned projection/property expansion under the authored
/// accuracy ceiling; the additional serialization envelope is admitted here.
///
/// The projection is a pure function of the workload's and mesh's content identities, and
/// every phase of one study (plan, worker reauthentication, recovery, publication) asks for
/// it again. The process keeps the last derivation keyed by those identities so the original
/// support is prepared once per process instead of once per caller.
pub fn derive_fluid_steady_resource_projection(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<FluidSteadyResourceProjection, ResourcePlanningError> {
    type Key = (Digest, Digest);
    static LAST: std::sync::Mutex<Option<(Key, FluidSteadyResourceProjection)>> =
        std::sync::Mutex::new(None);
    let key = (
        workload.identity().map_err(invalid)?.digest(),
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    );
    if let Some((cached, projection)) = LAST.lock().ok().as_deref().and_then(Option::as_ref) {
        if *cached == key {
            return Ok(projection.clone());
        }
    }
    let projection = derive_projection(workload, mesh)?;
    if let Ok(mut last) = LAST.lock() {
        *last = Some((key, projection.clone()));
    }
    Ok(projection)
}

fn derive_projection(
    workload: &FluidSteadyWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<FluidSteadyResourceProjection, ResourcePlanningError> {
    let source = workload.prepare(mesh).map_err(invalid)?;
    let problem = source.problem(workload).map_err(invalid)?;
    let numerical = problem.reservation();
    let projection_digest = source.source.projection_digest;
    let geometry_digest = source.topology.geometry_digest;
    let properties = source.property_identity.clone();
    let property_audit_digest = source.property_audit_digest;
    let source_preparation_resident_bytes = source.preparation_resident_bytes;
    let source_preparation_scratch_bytes = source.preparation_scratch_bytes;
    // The execution-owned result profile reads the support prepared here.
    let result = workload
        .result_encoding_profile_prepared(mesh, &source, problem)
        .map_err(invalid)?;
    drop(source);
    let workload_profile = profile_serialized(workload, LIMITS).map_err(invalid)?;
    let mesh_profile = profile_serialized(mesh, LIMITS).map_err(invalid)?;
    let input_profiles = admission::add(
        workload_profile.materialization_resident_bytes_estimate,
        mesh_profile.materialization_resident_bytes_estimate,
    )?;
    let input_buffers = admission::add(
        workload_profile.encoded_bytes_upper_bound,
        mesh_profile.encoded_bytes_upper_bound,
    )?;
    let decoded_inputs = admission::add(input_profiles, input_buffers)?;
    let phase_resident = admission::phase_resident_bytes(
        numerical.resident_bytes_upper_bound,
        numerical.verification_resident_bytes_upper_bound,
        source_preparation_resident_bytes,
        source_preparation_scratch_bytes,
        decoded_inputs,
        result.encoded_bytes_upper_bound,
        result.materialization_resident_bytes_estimate,
    )?;
    let verification_phase_resident = admission::phase_resident_bytes(
        numerical.verification_resident_bytes_upper_bound,
        numerical.verification_resident_bytes_upper_bound,
        source_preparation_resident_bytes,
        source_preparation_scratch_bytes,
        decoded_inputs,
        result.encoded_bytes_upper_bound,
        result.materialization_resident_bytes_estimate,
    )?;
    tracing::debug!(
        cells = numerical.cells,
        faces = numerical.faces,
        numerical = numerical.resident_bytes_upper_bound,
        verification = numerical.verification_resident_bytes_upper_bound,
        preparation = source_preparation_resident_bytes,
        preparation_scratch = source_preparation_scratch_bytes,
        decoded_inputs,
        result_encoded = result.encoded_bytes_upper_bound,
        result_materialization = result.materialization_resident_bytes_estimate,
        phase_resident,
        "steady fluid phase residency terms"
    );
    admission::require_ceiling(
        phase_resident,
        workload.resident_ceiling().map_err(invalid)?,
    )?;
    // Profiles and the additive ceiling are checked before allocating wire bytes.
    let workload_bytes = workload.canonical_bytes().map_err(invalid)?;
    let mesh_bytes = mesh.canonical_encode().map_err(invalid)?;
    let canonical_input_bytes = admission::add(
        u64::try_from(workload_bytes.len())
            .map_err(|_| ResourcePlanningError::Overflow("fluid workload wire length"))?,
        u64::try_from(mesh_bytes.len())
            .map_err(|_| ResourcePlanningError::Overflow("fluid mesh wire length"))?,
    )?;
    let actual_decode_resident_bytes = admission::add(
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&workload_bytes)
            .map_err(invalid)?,
        runmat_canonical_codec::canonical_payload_typed_resident_bytes_estimate(&mesh_bytes)
            .map_err(invalid)?,
    )?;
    drop(workload_bytes);
    drop(mesh_bytes);
    Ok(FluidSteadyResourceProjection {
        workload: workload.identity().map_err(invalid)?,
        producer: workload.producer_identity().map_err(invalid)?,
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        projection_digest,
        geometry_digest,
        properties,
        property_audit_digest,
        numerical,
        source_preparation_resident_bytes: ByteCount(source_preparation_resident_bytes),
        source_preparation_scratch_bytes: ByteCount(source_preparation_scratch_bytes),
        canonical_input_bytes: ByteCount(canonical_input_bytes),
        actual_decode_resident_bytes: ByteCount(actual_decode_resident_bytes),
        result_payload_bytes_upper_bound: ByteCount(result.encoded_bytes_upper_bound),
        result_materialization_resident_bytes: ByteCount(
            result.materialization_resident_bytes_estimate,
        ),
        phase_resident_bytes_upper_bound: ByteCount(phase_resident),
        verification_phase_resident_bytes_upper_bound: ByteCount(verification_phase_resident),
    })
}

/// Compare scheduler transport facts with the same actual canonical wire authority.
/// Maximum-width profiles reserve construction but do not inflate the required
/// actual input decode budget. These checks do not attest execution or usage.
pub fn validate_fluid_steady_prepared_facts(
    projection: &FluidSteadyResourceProjection,
    prepared: &PreparedPhaseResourceFacts,
) -> Result<(), ResourcePlanningError> {
    admission::validate_prepared(
        projection.canonical_input_bytes.0,
        projection.actual_decode_resident_bytes.0,
        projection.result_payload_bytes_upper_bound.0,
        projection.result_materialization_resident_bytes.0,
        prepared,
    )
}
