//! Assembly-free CPU admission of canonical scalar electrical FEM.
//! Reservations sum phase lifetimes conservatively; they are requested heap, not measured RSS.
use super::*;
use crate::{
    algebra::{CoercivityWorkEnvelope, SparsePattern},
    execution::ElectricalStaticWorkload,
    properties::PreparedProperties,
    topology::AnalysisProjection,
};
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedElectricalStaticResourceFacts {
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
}

const LIMITS: CanonicalLimits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);
fn invalid(error: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(error.to_string())
}
fn count(value: usize) -> Result<u64, ResourcePlanningError> {
    u64::try_from(value).map_err(|_| ResourcePlanningError::Overflow("electrical cardinality"))
}
fn add(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_add(b)
        .ok_or(ResourcePlanningError::Overflow("electrical byte sum"))
}
fn mul(a: u64, b: u64) -> Result<u64, ResourcePlanningError> {
    a.checked_mul(b)
        .ok_or(ResourcePlanningError::Overflow("electrical byte product"))
}

fn support_preparation_bytes(mesh: &SolverMeshArtifact) -> Result<u64, ResourcePlanningError> {
    let nodes = count(mesh.topology.nodes.len())?;
    let entries = mesh
        .topology
        .volume_elements
        .iter()
        .try_fold(nodes, |total, element| {
            let width = count(element.node_ids.len())?;
            add(total, mul(width, width)?)
        })?;
    // Raw ID index, row headers/diagonals, BTree support/undirected proof graph,
    // simultaneously retained CSR and later geometric projection construction.
    add(
        add(mul(nodes, 256)?, mul(entries, 384)?)?,
        AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?,
    )
}

fn require_support_preparation_ceiling(
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<(), ResourcePlanningError> {
    workload.validate().map_err(invalid)?;
    let required = support_preparation_bytes(mesh)?;
    let maximum = workload.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes;
    if required > maximum {
        return Err(ResourcePlanningError::AccuracyCeiling {
            resource: "electrical support preparation bytes",
            required,
            maximum,
        });
    }
    Ok(())
}

/// Source-bound canonical result payload bound, including arbitrary terminal identifiers.
/// Receipt and publication-profile payloads are reserved separately.
pub fn electrical_static_result_payload_bytes_upper_bound(
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<u64, ResourcePlanningError> {
    require_support_preparation_ceiling(workload, mesh)?;
    let formulation = workload.formulation(mesh).map_err(invalid)?;
    mesh.validate_canonical().map_err(invalid)?;
    let nodes = count(mesh.topology.nodes.len())?;
    let elements = count(mesh.topology.volume_elements.len())?;
    let terminals = count(workload.model.constraints.len())?;
    // IDs, potential and physical flux per node; ID, E and constitutive vector/energy per
    // element. Nine bytes per scalar plus array framing fits these intentionally loose bounds.
    let fields = add(mul(nodes, 64)?, mul(elements, 160)?)?;
    let terminal_fields = mul(terminals, add(128, mul(nodes, 16)?)?)?;
    let formulation = profile_serialized(&formulation, LIMITS).map_err(invalid)?;
    let constraints = profile_serialized(&workload.model.constraints, LIMITS).map_err(invalid)?;
    let declarations = profile_serialized(&workload.model.sequence.steps[0].state_outputs, LIMITS)
        .map_err(invalid)?;
    // Closed schema keys, fixed-width producer/source/projection/mesh identities, order,
    // residual/iteration/energy/conservation summaries and framing (not authored strings).
    [
        4096,
        fields,
        terminal_fields,
        formulation.encoded_bytes_upper_bound,
        constraints.encoded_bytes_upper_bound,
        declarations.encoded_bytes_upper_bound,
    ]
    .into_iter()
    .try_fold(0, add)
}

/// Same material-independent complete local-support graph as scalar assembly. The shared
/// work envelope bounds sparse subgraphs; a successful upper-graph symbolic factorization
/// alone would not bound the numerical operator's different elimination ordering.
pub fn electrical_static_solver_admission(
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<SolverAdmissionResourcePolicy, ResourcePlanningError> {
    require_support_preparation_ceiling(workload, mesh)?;
    workload.formulation(mesh).map_err(invalid)?;
    let n = mesh.topology.nodes.len();
    // No physical quadrature projection is needed to establish scalar incidence support.
    let node_indices = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut rows = (0..n).map(|row| BTreeSet::from([row])).collect::<Vec<_>>();
    for element in &mesh.topology.volume_elements {
        for row in &element.node_ids {
            let row = *node_indices
                .get(row)
                .ok_or_else(|| invalid("missing electrical support node"))?;
            for column in &element.node_ids {
                rows[row].insert(
                    *node_indices
                        .get(column)
                        .ok_or_else(|| invalid("missing electrical support node"))?,
                );
            }
        }
    }
    let mut offsets = vec![0];
    let mut columns = Vec::new();
    for row in rows {
        for column in row {
            columns.push(count(column)?);
        }
        offsets.push(count(columns.len())?);
    }
    let support =
        SparsePattern::try_new(count(n)?, count(n)?, offsets, columns).map_err(invalid)?;
    let envelope = CoercivityWorkEnvelope::derive(&support).map_err(invalid)?;
    Ok(SolverAdmissionResourcePolicy {
        dimension: Count(envelope.dimension()),
        factor_nonzero_upper_bound: Count(envelope.factor_nonzero_limit()),
        update_count_upper_bound: Count(envelope.update_limit()),
        resident_bytes_upper_bound: ByteCount(envelope.resident_bytes_limit()),
    })
}

pub fn build_electrical_static_resource_plan(
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
    prepared: PreparedElectricalStaticResourceFacts,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    require_support_preparation_ceiling(workload, mesh)?;
    let payload = electrical_static_result_payload_bytes_upper_bound(workload, mesh)?;
    let state_profiles = workload
        .state_publication_profiles_encoding_profile(Digest::from_bytes(
            *mesh.canonical_digest.bytes(),
        ))
        .map_err(invalid)?;
    let inputs = add(
        count(workload.canonical_bytes().map_err(invalid)?.len())?,
        count(mesh.canonical_encode().map_err(invalid)?.len())?,
    )?;
    if prepared.inputs.encoded_payload_bytes.0 != inputs
        || prepared.result_output.encoded_payload_bytes_upper_bound.0 != payload
        || prepared.receipt_output.encoded_payload_bytes_upper_bound.0
            != crate::execution::ELECTRICAL_RECEIPT_MAX_ENCODED_BYTES
        || prepared
            .state_profiles_output
            .encoded_payload_bytes_upper_bound
            .0
            != state_profiles.encoded_bytes_upper_bound
    {
        return Err(invalid(
            "electrical canonical input or projected result bytes disagree",
        ));
    }
    let step = &workload.model.sequence.steps[0];
    let nodes = count(mesh.topology.nodes.len())?;
    let elements = count(mesh.topology.volume_elements.len())?;
    let support = mesh
        .topology
        .volume_elements
        .iter()
        .try_fold(nodes, |total, element| {
            add(
                total,
                mul(
                    count(element.node_ids.len())?,
                    count(element.node_ids.len())?,
                )?,
            )
        })?;
    let admission = electrical_static_solver_admission(workload, mesh)?;
    let projection =
        AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
    let properties = PreparedProperties::prepare(
        &workload.model,
        Some(mesh),
        step.accuracy.resources.maximum_resident_bytes,
    )
    .map_err(invalid)?
    .estimated_resident_bytes;
    let metadata = profile_serialized(&workload.model, LIMITS)
        .map_err(invalid)?
        .materialization_resident_bytes_estimate;
    // Physical assembled K plus normalized K and prepared constrained K can coexist.
    // BTree discovery/finalization allowance is added separately. Zero M/C own CSR storage.
    let matrices = add(mul(support, 224)?, mul(add(nodes, 1)?, 128)?)?;
    // Proof construction/support sets/CSR/edge dedup and both symbolic+numeric proof buffers.
    let proof = add(
        mul(admission.resident_bytes_upper_bound.0, 2)?,
        mul(support, 192)?,
    )?;
    // Gauge/component lists, lifting/RHS, Jacobi, PCG temporaries, residual, potentials,
    // physical flux, terminal node vectors/ownership maps and result validation scratch.
    let vectors = add(
        mul(nodes, 512)?,
        mul(mul(nodes, count(workload.model.constraints.len())?)?, 64)?,
    )?;
    // Per-element gradient recovery, physical product integration, retained constitutive
    // arrays/coefficient data and result clone. All lifetimes are summed, not maxed.
    let recovery = mul(elements, 512)?;
    let reservation = [
        support_preparation_bytes(mesh)?,
        mul(projection, 3)?,
        properties,
        metadata,
        matrices,
        proof,
        vectors,
        recovery,
        payload,
    ]
    .into_iter()
    .try_fold(0, add)?;
    let source = workload.identity().map_err(invalid)?;
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::ElectricalStatic(source),
        demand: FeaResourceDemand::ElectricalStatic(ElectricalStaticResourceDemand {
            workload: source,
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            node_count: Count(nodes),
            element_count: Count(elements),
            operator_nonzeros_upper_bound: Count(support),
            numerical_reservation_bytes: ByteCount(reservation),
            solver_admission: admission,
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
            cpu_millicores: 1000,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(16 * 1024 * 1024),
            fixed_scratch_bytes: ByteCount(8 * 1024 * 1024),
            required_capabilities: Default::default(),
            accelerators: Vec::new(),
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}
