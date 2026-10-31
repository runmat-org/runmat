//! Count-only admission for canonical cavity-mode and pressure-transient solves.
//! Conservative additive requested heap from element incidence, never measured RSS.
use super::*;
use crate::{
    execution::{AcousticResponseSolver, AcousticResponseWorkload},
    topology::AnalysisProjection,
};
use runmat_analysis_core::IndependentDomain;
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

const LIMITS: CanonicalLimits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);
/// Fixed result keys, identities, formulation and scalar evidence.
const RESULT_SCHEMA_BYTES: u64 = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedAcousticResponseResourceFacts {
    /// The admitted local CPU plan: the kernel's thread authority (as for every family).
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
    u64::try_from(n).map_err(|_| ResourcePlanningError::Overflow("acoustic response count"))
}
fn add(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_add(b)
        .ok_or(ResourcePlanningError::Overflow("acoustic response sum"))
}
fn mul(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_mul(b)
        .ok_or(ResourcePlanningError::Overflow("acoustic response product"))
}

/// Retained pressure samples: mode count or every accepted time point including the start.
pub fn acoustic_response_sample_count(
    workload: &AcousticResponseWorkload,
) -> Result<u64, ResourcePlanningError> {
    let step = &workload.model.sequence.steps[0];
    match (&workload.solver, &step.independent_domain) {
        (
            AcousticResponseSolver::Modal { .. },
            IndependentDomain::Eigenvalue { requested_count },
        ) => count(*requested_count),
        (
            AcousticResponseSolver::Transient { .. },
            IndependentDomain::TimeSeconds { interval_count, .. },
        ) => add(count(*interval_count)?, 1),
        _ => Err(invalid(
            "acoustic response domain differs from its procedure",
        )),
    }
}

pub fn acoustic_response_result_payload_bytes_upper_bound(
    workload: &AcousticResponseWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<u64, ResourcePlanningError> {
    workload.validate().map_err(invalid)?;
    let nodes = count(mesh.topology.nodes.len())?;
    let samples = acoustic_response_sample_count(workload)?;
    let step = &workload.model.sequence.steps[0];
    if samples > count(step.accuracy.resources.maximum_samples)? {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "acoustic response retained samples",
            required: samples,
            maximum: count(step.accuracy.resources.maximum_samples)?,
        });
    }
    let formulation = workload.formulation(mesh).map_err(invalid)?;
    let formulation_bytes = profile_serialized(&formulation, LIMITS)
        .map_err(invalid)?
        .encoded_bytes_upper_bound;
    // Canonical f64 and u64 encodings are at most nine bytes; reserve sixteen.
    let elements = count(mesh.topology.volume_elements.len())?;
    let values = mul(add(add(mul(samples, nodes)?, nodes)?, elements)?, 16)?;
    let per_sample = mul(samples, 96)?;
    // The result repeats the typed authored declarations so detached cold decoding can reject
    // missing or substituted output authority. Charge their canonical workload envelope too.
    let authored_bytes = count(workload.canonical_bytes().map_err(invalid)?.len())?;
    add(
        add(add(RESULT_SCHEMA_BYTES, formulation_bytes)?, authored_bytes)?,
        add(values, per_sample)?,
    )
}

/// Element-incidence upper bound of one assembled K/M/B CSR pattern.
fn operator_support(mesh: &SolverMeshArtifact) -> Result<u64, ResourcePlanningError> {
    let volume = mesh
        .topology
        .volume_elements
        .iter()
        .try_fold(0_u64, |total, element| {
            let n = count(element.node_ids.len())?;
            add(total, mul(n, n)?)
        })?;
    let boundary = mesh
        .topology
        .boundary_faces
        .iter()
        .try_fold(0_u64, |total, face| {
            let n = count(face.node_ids.len())?;
            add(total, mul(n, n)?)
        })?;
    add(volume, boundary)
}

pub fn build_acoustic_response_resource_plan(
    workload: &AcousticResponseWorkload,
    mesh: &SolverMeshArtifact,
    prepared: PreparedAcousticResponseResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    let payload = acoustic_response_result_payload_bytes_upper_bound(workload, mesh)?;
    let profiles = workload
        .state_publication_profiles_encoding_profile(Digest::from_bytes(
            *mesh.canonical_digest.bytes(),
        ))
        .map_err(invalid)?;
    let workload_bytes = workload.canonical_bytes().map_err(invalid)?;
    let input_bytes = add(
        count(workload_bytes.len())?,
        mesh.canonical_encoded_len().map_err(invalid)?,
    )?;
    if prepared.inputs.encoded_payload_bytes.0 != input_bytes
        || prepared.result_output.encoded_payload_bytes_upper_bound.0 != payload
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0
            != crate::execution::ACOUSTIC_RESPONSE_RECEIPT_MAX_ENCODED_BYTES
        || prepared
            .state_profiles_output
            .encoded_payload_bytes_upper_bound
            .0
            != profiles.encoded_bytes_upper_bound
    {
        return Err(invalid(
            "acoustic response original inputs or projected outputs disagree",
        ));
    }
    let nodes = count(mesh.topology.nodes.len())?;
    let support = operator_support(mesh)?;
    let samples = acoustic_response_sample_count(workload)?;
    // Assembly maps plus CSR values/columns for each simultaneously live operator.
    let (live_operators, factors, vectors) = match workload.solver {
        AcousticResponseSolver::Modal { .. } => {
            // K, M, B, K + sM, free K and free M plus the sparse Cholesky factor of K + sM;
            // V, KV, MV basis blocks of capacity 2m + 6b + 20 with b <= 4, plus Rayleigh-Ritz
            // dense work.
            let capacity = add(mul(samples, 2)?, 44)?;
            let dense = mul(mul(capacity, capacity)?, 24)?;
            (
                6,
                1,
                add(mul(mul(capacity, 3)?, nodes)?, add(dense, mul(nodes, 16)?)?)?,
            )
        }
        AcousticResponseSolver::Transient { .. } => {
            // K, M, B, effective, free effective and free mass plus the sparse Cholesky
            // factors of the free effective and free mass operators; the retained history plus
            // about twenty state/work vectors.
            (6, 2, add(mul(samples, nodes)?, mul(nodes, 24)?)?)
        }
    };
    let operators = mul(mul(support, live_operators)?, 160)?;
    // Each factor is quoted by the shared direct-solver reservation model from the operator's
    // order and its element-incidence support (an upper bound of its nonzeros).
    let factor = runmat_sparse_solvers::estimate::direct_reservation_bytes(
        usize::try_from(nodes).map_err(|_| invalid("acoustic node count"))?,
        usize::try_from(support).map_err(|_| invalid("acoustic operator support"))?,
        false,
    );
    if factor == u64::MAX {
        return Err(ResourcePlanningError::Overflow(
            "acoustic factor reservation",
        ));
    }
    let operators = add(operators, mul(factor, factors)?)?;
    let projection =
        AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
    let reservation = add(
        add(operators, mul(vectors, 8)?)?,
        add(
            mul(projection, 2)?,
            add(mul(payload, 2)?, add(input_bytes, mul(nodes, 512)?)?)?,
        )?,
    )?;
    let source = workload.identity().map_err(invalid)?;
    let step = &workload.model.sequence.steps[0];
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::AcousticResponse(source),
        demand: FeaResourceDemand::AcousticResponse(AcousticResponseResourceDemand {
            workload: source,
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            node_count: Count(nodes),
            element_count: Count(count(mesh.topology.volume_elements.len())?),
            sample_count: Count(samples),
            operator_support_upper_bound: Count(support),
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
