//! Assembly-free admission for the canonical CPU Helmholtz workload.
//!
//! All reservations are conservative requested-heap estimates, not measured RSS. Raw element
//! incidence bounds real K/M/B storage; no graph operator or numerical assembly is substituted.
use super::*;
use crate::{
    excitation::{
        realize_step_excitations, ExcitationSamplePoint, RealizedExcitationKind,
        SpatialIntegrationRule, SpatialProjectionEvidence,
    },
    execution::AcousticHarmonicWorkload,
    properties::PreparedProperties,
    topology::AnalysisProjection,
};
use runmat_canonical_codec::{profile_serialized, CanonicalLimits, SerializedCanonicalProfile};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

mod sparse_workspace;
#[cfg(test)]
mod tests;
pub(crate) use sparse_workspace::sparse_workspace_bound;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedAcousticResourceFacts {
    pub local_execution: super::LocalExecutionPlan,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}

const PROFILE_LIMITS: CanonicalLimits =
    CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);
// Only the fixed result keys, producer/fixed-width identity/digest values, six vector headers,
// frequency/solver/tolerance/residual/power/iteration scalars and contract prefix. Even counting
// every key at 64 bytes and every summary scalar at 9 bytes, including the factorization
// certificate and its operator digest, fits 8192 bytes. The formulation
// realization is independently profiled below (its vectors are not covered by this constant).
// Authored strings are separately profiled using their declaration and shared evidence wire.
const CLOSED_RESULT_SCHEMA_BYTES: u64 = 8192;

fn invalid(error: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(error.to_string())
}
fn count(value: usize) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| ResourcePlanningError::Overflow("acoustic cardinality"))
}
fn add(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_add(b)
        .ok_or(ResourcePlanningError::Overflow("acoustic byte sum"))
}
fn mul(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_mul(b)
        .ok_or(ResourcePlanningError::Overflow("acoustic byte product"))
}

fn evidence_profile(
    workload: &AcousticHarmonicWorkload,
) -> Result<SerializedCanonicalProfile, ResourcePlanningError> {
    workload.validate().map_err(invalid)?;
    let step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or_else(|| invalid("acoustic step missing"))?;
    let loads = realize_step_excitations(
        &workload.model,
        step,
        ExcitationSamplePoint::frequency(workload.frequency.canonical_value()),
    )
    .map_err(invalid)?;
    let mut total = SerializedCanonicalProfile {
        encoded_bytes_upper_bound: 9,
        materialization_resident_bytes_estimate: 24,
    };
    for load in loads.iter().filter(|load| {
        matches!(
            load.kind,
            RealizedExcitationKind::NormalVelocity(_)
                | RealizedExcitationKind::VolumeVelocity(_)
                | RealizedExcitationKind::AcousticIncidentPressure(_)
        )
    }) {
        let profile = runmat_canonical_codec::profile_serialized_shape(
            &SpatialProjectionEvidence::scalar_projection_serialization(
                &load.excitation_id.0,
                &load.signal_id.0,
                load.quantity,
                &load.sample,
                SpatialIntegrationRule::BoundaryDunavantSeven,
                &load.policy.projection,
            ),
            PROFILE_LIMITS,
        )
        .map_err(invalid)?;
        total.encoded_bytes_upper_bound = add(
            total.encoded_bytes_upper_bound,
            profile.encoded_bytes_upper_bound,
        )?;
        total.materialization_resident_bytes_estimate = add(
            total.materialization_resident_bytes_estimate,
            profile.materialization_resident_bytes_estimate,
        )?;
    }
    Ok(total)
}

/// Canonical payload ceiling, before shared transport chunk/manifest/output-set reservation.
/// No solve or numerical matrix assembly occurs here.
pub fn acoustic_harmonic_result_payload_bytes_upper_bound(
    workload: &AcousticHarmonicWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<u64, ResourcePlanningError> {
    let formulation = workload.formulation(mesh).map_err(invalid)?;
    mesh.validate_canonical().map_err(invalid)?;
    let nodes = count(mesh.topology.nodes.len())?;
    let elements = count(mesh.topology.volume_elements.len())?;
    // ID plus complex pressure map (two named 9-byte scalars) per node. Per element: ID,
    // complex velocity (six scalars/two array keys), intensity (three), energy (one).
    let fields = add(mul(nodes, 80)?, mul(elements, 160)?)?;
    let formulation_bytes = profile_serialized(&formulation, PROFILE_LIMITS)
        .map_err(invalid)?
        .encoded_bytes_upper_bound;
    let declarations = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or_else(|| invalid("acoustic step missing"))?;
    let declaration_bytes = profile_serialized(&declarations.state_outputs, PROFILE_LIMITS)
        .map_err(invalid)?
        .encoded_bytes_upper_bound;
    add(
        add(
            add(add(CLOSED_RESULT_SCHEMA_BYTES, fields)?, formulation_bytes)?,
            declaration_bytes,
        )?,
        evidence_profile(workload)?.encoded_bytes_upper_bound,
    )
}

pub fn build_acoustic_resource_plan(
    workload: &AcousticHarmonicWorkload,
    mesh: &SolverMeshArtifact,
    prepared: PreparedAcousticResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let payload_bound = acoustic_harmonic_result_payload_bytes_upper_bound(workload, mesh)?;
    let state_profiles = workload
        .state_publication_profiles_encoding_profile(Digest::from_bytes(
            *mesh.canonical_digest.bytes(),
        ))
        .map_err(invalid)?;
    let input_bytes = add(
        count(workload.canonical_bytes().map_err(invalid)?.len())?,
        mesh.canonical_encoded_len().map_err(invalid)?,
    )?;
    if prepared.inputs.encoded_payload_bytes.0 != input_bytes
        || prepared.result_output.encoded_payload_bytes_upper_bound.0 != payload_bound
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0
            != crate::execution::ACOUSTIC_RECEIPT_MAX_ENCODED_BYTES
        || prepared
            .state_profiles_output
            .encoded_payload_bytes_upper_bound
            .0
            != state_profiles.encoded_bytes_upper_bound
    {
        return Err(invalid(
            "acoustic canonical input or projected result bytes disagree",
        ));
    }
    let step = workload
        .model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or_else(|| invalid("acoustic step missing"))?;
    let nodes = count(mesh.topology.nodes.len())?;
    let elements = count(mesh.topology.volume_elements.len())?;
    let block = mul(nodes, 2)?;
    // Exact local supports bound discovered sparse nonzeros, with shared entries counted more
    // than once. Each operator owns its row offsets and columns; K/M never share storage.
    let volume_support =
        mesh.topology
            .volume_elements
            .iter()
            .try_fold(0u64, |total, element| {
                add(
                    total,
                    mul(
                        count(element.node_ids.len())?,
                        count(element.node_ids.len())?,
                    )?,
                )
            })?;
    let boundary_support = mesh
        .topology
        .boundary_faces
        .iter()
        .try_fold(0u64, |total, face| {
            add(
                total,
                mul(count(face.node_ids.len())?, count(face.node_ids.len())?)?,
            )
        })?;
    let support = add(mul(volume_support, 2)?, boundary_support)?;
    let projection =
        AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
    let properties = PreparedProperties::prepare(
        &workload.model,
        Some(mesh),
        step.accuracy.resources.resident_byte_budget(),
    )
    .map_err(invalid)?
    .estimated_resident_bytes;
    // BTree discovered row assembly reserves 128 bytes per possible (column,f64) entry,
    // including spare node/key/pointer slots. CSR values+columns add 16 bytes per entry;
    // finalization and discovered storage coexist. Spare capacity is included conservatively.
    let operators = add(mul(support, 160)?, mul(add(nodes, 1)?, 3 * 64)?)?;
    // Full vectors: lifting/real/imaginary/apply's six SpMV buffers, source, prescribed,
    // free indexes, pressure and recovery fields. Boundary ownership sets/faces and source
    // accumulators are separate from projection. These phase lifetimes are summed, not maxed.
    let vectors = add(mul(nodes, 512)?, mul(elements, 256)?)?;
    let boundaries = mul(count(mesh.topology.boundary_faces.len())?, 512)?;
    // Shared unit-phase graph/affine recovery, canonical node lookup and target pairing.
    // At most N matching relations per active authored paired constraint. Include graph,
    // relation-vector spare capacity and pairing metadata simultaneously; the shared map
    // has at most four real coefficients per complex node. No dense reduction is built.
    let periodic_count = count(workload.model.constraints.iter().filter(|constraint| {
        step.object_is_active(&runmat_analysis_core::StepObjectRef::Constraint(
            constraint.constraint_id().into())) && matches!(constraint,
            runmat_analysis_core::AnalysisConstraint::Acoustic(runmat_analysis_core::AcousticConstraint {
                kind: runmat_analysis_core::AcousticConstraintKind::Periodic(_)
                    | runmat_analysis_core::AcousticConstraintKind::PhaseShiftedPeriodic(_), ..
            }))
    }).count())?;
    // Without an active periodic relation, recovery rows are identity or
    // prescribed (one or zero entries), not general two-entry phase rows.
    let recovery_row_support = if periodic_count == 0 { 1 } else { 2 };
    let sparse_solve = sparse_workspace_bound(support, block, recovery_row_support)?;
    let phase_constraints = add(mul(nodes, 512)?, mul(mul(nodes, periodic_count)?, 128)?)?;
    let evidence = evidence_profile(workload)?;
    // Workload clone/source realization metadata is profiled from the complete authored input;
    // property compilation separately owns its selected material clones.
    let metadata = profile_serialized(&workload.model, PROFILE_LIMITS)
        .map_err(invalid)?
        .materialization_resident_bytes_estimate;
    let reservation = [
        // Boundary selectors clone face records and provenance; reserve another complete
        // projection metadata envelope while the authoritative projection remains retained.
        // A disjoint incident-face set can retain at most one complete projection's
        // face metadata for physical net-input power, in addition to selector clones.
        mul(projection, 3)?,
        properties,
        operators,
        sparse_solve,
        vectors,
        boundaries,
        phase_constraints,
        metadata,
        evidence.materialization_resident_bytes_estimate,
    ]
    .into_iter()
    .try_fold(0u64, add)?;
    let source = workload.identity().map_err(invalid)?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::AcousticHarmonic(source),
        demand: FeaResourceDemand::AcousticHarmonic(AcousticHarmonicResourceDemand {
            workload: source,
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            node_count: Count(nodes),
            element_count: Count(elements),
            operator_nonzeros_upper_bound: Count(support),
            block_dimension_upper_bound: Count(block),
            recovery_row_support_upper_bound: Count(recovery_row_support),
            sparse_solve_reservation_bytes: ByteCount(sparse_solve),
            numerical_reservation_bytes: ByteCount(reservation),
            inputs: prepared.inputs,
            result_output: prepared.result_output,
            receipt_output: prepared.receipt_output,
            state_profiles_output: prepared.state_profiles_output,
            state_profiles_resident_bytes: ByteCount(
                state_profiles.materialization_resident_bytes_estimate,
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
