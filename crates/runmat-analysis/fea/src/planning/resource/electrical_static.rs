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
    pub local_execution: super::LocalExecutionPlan,
    pub inputs: PreparedInputResourceFacts,
    pub result_output: PreparedOutputResourceFacts,
    pub receipt_output: PreparedOutputResourceFacts,
    pub state_profiles_output: PreparedOutputResourceFacts,
    pub output_set: PreparedOutputSetResourceFacts,
    /// Memory this machine admits for work (the shared machine-capacity authority), or, when a
    /// plan is replayed, the solver budget the plan recorded.
    pub machine_memory_bytes: u64,
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
    let entries = element_couplings(mesh)?;
    // Raw ID index and row sets of the support graph, its CSR copy, the undirected proof graph
    // and the nested-dissection symbolic analysis. Calibrated against measured peak RSS of the
    // planner on Tet4 and Tet10 meshes (about 70 bytes per element coupling) with headroom.
    add(mul(nodes, 128)?, mul(entries, 96)?)
}

/// Preparation and execution count the same original element contributions, before
/// assembly merges repeated node pairs. Prescribed rows use one bool per node.
fn local_energy_proof_demand(
    mesh: &SolverMeshArtifact,
    envelope: CoercivityWorkEnvelope,
) -> Result<crate::algebra::anchored_energy::SourceEnergyDemand, ResourcePlanningError> {
    crate::algebra::anchored_energy::SourceEnergyDemand::derive(
        envelope,
        mesh.topology.nodes.len(),
        mul(
            count(mesh.topology.nodes.len())?,
            std::mem::size_of::<bool>() as u64,
        )?,
        mesh.topology
            .volume_elements
            .iter()
            .map(|element| element.node_ids.len()),
    )
    .map_err(invalid)
}

/// Element node pairs, with repeats: the sum of squared element widths plus the diagonal.
fn element_couplings(mesh: &SolverMeshArtifact) -> Result<u64, ResourcePlanningError> {
    mesh.topology.volume_elements.iter().try_fold(
        count(mesh.topology.nodes.len())?,
        |total, element| {
            let width = count(element.node_ids.len())?;
            add(total, mul(width, width)?)
        },
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
        .resident_byte_budget();
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
    // Retained gauge increments and optional nodal volume charge are separate arrays.
    let fields = add(mul(nodes, 96)?, mul(elements, 160)?)?;
    let terminal_fields = mul(terminals, add(128, mul(nodes, 16)?)?)?;
    let formulation = profile_serialized(&formulation, LIMITS).map_err(invalid)?;
    let constraints = profile_serialized(&workload.model.constraints, LIMITS).map_err(invalid)?;
    let declarations = profile_serialized(&workload.model.sequence.steps[0].state_outputs, LIMITS)
        .map_err(invalid)?;
    let charge_metadata =
        profile_serialized(&workload.model.excitations, LIMITS).map_err(invalid)?;
    // Per-action projection evidence carries copied source/signal names and fixed scalar
    // balance metadata. Its topology cardinalities are counts, not another connectivity array.
    let charge_evidence = add(
        mul(charge_metadata.encoded_bytes_upper_bound, 2)?,
        mul(count(workload.model.excitations.len())?, 2048)?,
    )?;
    // Closed schema keys, fixed-width producer/source/projection/mesh identities, order,
    // residual/iteration/energy/conservation summaries and framing (not authored strings).
    [
        4096,
        fields,
        terminal_fields,
        formulation.encoded_bytes_upper_bound,
        constraints.encoded_bytes_upper_bound,
        declarations.encoded_bytes_upper_bound,
        charge_evidence,
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
    electrical_static_solver_envelope(workload, mesh).map(|(policy, _)| policy)
}

fn electrical_static_solver_envelope(
    workload: &ElectricalStaticWorkload,
    mesh: &SolverMeshArtifact,
) -> Result<(SolverAdmissionResourcePolicy, CoercivityWorkEnvelope), ResourcePlanningError> {
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
    let energy = local_energy_proof_demand(mesh, envelope)?;
    Ok((
        SolverAdmissionResourcePolicy {
            dimension: Count(envelope.dimension()),
            factor_nonzero_upper_bound: Count(envelope.factor_nonzero_limit()),
            update_count_upper_bound: Count(
                envelope.update_limit().max(energy.scalar_operations()),
            ),
            // The local-energy proof runs first; when its bound is indeterminate the sparse
            // Cholesky proof decides. The admitted resident bound covers either proof.
            resident_bytes_upper_bound: ByteCount(
                energy
                    .resident_bytes()
                    .max(envelope.factor_resident_limit()),
            ),
        },
        envelope,
    ))
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
        mesh.canonical_encoded_len().map_err(invalid)?,
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
    let support = element_couplings(mesh)?;
    let (admission, envelope) = electrical_static_solver_envelope(workload, mesh)?;
    let projection =
        AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
    let properties = PreparedProperties::prepare(
        &workload.model,
        Some(mesh),
        step.accuracy.resources.resident_byte_budget(),
    )
    .map_err(invalid)?
    .estimated_resident_bytes;
    let metadata = profile_serialized(&workload.model, LIMITS)
        .map_err(invalid)?
        .materialization_resident_bytes_estimate;
    // Physical, normalized and constrained K coexist: three CSR copies of at most `support`
    // entries (value and column index) plus row offsets.
    let matrices = add(mul(support, 64)?, mul(add(nodes, 1)?, 24)?)?;
    // Gauge/component lists, lifting/RHS, PCG temporaries, residual, potentials, physical flux,
    // terminal node vectors/ownership maps and result validation scratch.
    let vectors = add(
        mul(nodes, 512)?,
        mul(mul(nodes, count(workload.model.constraints.len())?)?, 64)?,
    )?;
    // Per-element gradient recovery, physical product integration, retained constitutive
    // arrays/coefficient data and result clone.
    let recovery = mul(elements, 512)?;
    // Source realization + retained evidence/metadata, gauge increments, source load and
    // cold source rederivation, raw-volume identity index and scalar projection scratch.
    let charge_sources = add(
        add(mul(metadata, 3)?, mul(nodes, 48)?)?,
        mul(elements, 128)?,
    )?;
    // Original once-captured local contributions remain live through the local-energy proof.
    let energy_source =
        crate::algebra::anchored_energy::AnchoredEnergySource::resident_bytes_for_widths(
            mesh.topology
                .volume_elements
                .iter()
                .map(|element| element.node_ids.len()),
        )
        .map_err(invalid)?;
    // Live through assembly, proof, solve and recovery. The two geometric projections (operator
    // assembly and result recovery) are derived in sequence, so at most two coexist.
    let persistent = [
        mul(projection, 2)?,
        properties,
        metadata,
        matrices,
        add(energy_source, nodes)?,
        vectors,
        recovery,
        charge_sources,
        payload,
    ]
    .into_iter()
    .try_fold(0, add)?;
    // The coercivity proof is the anchored local-energy proof (its graph and block scratch);
    // the solve then factors directly when the exact symbolic factor fits beside the persistent
    // state within the ceiling, and otherwise runs algebraic-multigrid preconditioned CG.
    let proof = admission.resident_bytes_upper_bound.0;
    // The effective budget: the authored ceiling, bounded by what this machine admits.
    let budget = step
        .accuracy
        .resources
        .resident_byte_budget()
        .min(prepared.machine_memory_bytes);
    let iterative = iterative_solve_bytes(nodes, support)?;
    let direct = electrical_direct_solve(
        persistent,
        &envelope,
        iterative_solve_flops(nodes, support),
        budget,
    );
    let solve = if direct {
        envelope.factor_resident_limit()
    } else {
        iterative
    };
    tracing::info!(
        nodes,
        elements,
        couplings = support,
        projection_each = projection,
        properties,
        matrices,
        vectors,
        recovery,
        charge_sources,
        energy_source,
        payload,
        support_preparation = support_preparation_bytes(mesh)?,
        proof,
        solve,
        direct,
        "electrical static reservation components (bytes)"
    );
    // Support preparation precedes assembly; the proof completes and releases its graph before
    // the solve starts, so the three are successive phases.
    let reservation = add(
        persistent,
        support_preparation_bytes(mesh)?.max(proof).max(solve),
    )?;
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
            direct_solve: direct,
            solver_memory_budget: ByteCount(budget),
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

/// The electrical solve factors directly when the factor and the persistent state fit within
/// 70 % of the effective budget and the factorization's estimated work is below that of
/// AMG-preconditioned CG; otherwise it runs AMG-preconditioned CG. The plan records the choice and
/// the budget, so a replay solves exactly as planned, whatever the executing host's memory.
fn electrical_direct_solve(
    persistent: u64,
    envelope: &CoercivityWorkEnvelope,
    iterative_flops: u64,
    budget: u64,
) -> bool {
    persistent.saturating_add(envelope.factor_resident_limit()) as f64 <= 0.7 * budget as f64
        && envelope.factor_flop_limit() <= iterative_flops
}

/// AMG-PCG work: about 100 iterations, each applying the operator and a V-cycle of roughly
/// four operator-complexity applications (2 flops per stored coupling).
fn iterative_solve_flops(nodes: u64, entries: u64) -> u64 {
    entries
        .saturating_mul(2 * 5)
        .saturating_add(nodes.saturating_mul(20))
        .saturating_mul(100)
}

/// AMG hierarchy (operator complexity, prolongators and level vectors) plus CG workspace for
/// `entries` couplings, with 25 % headroom over the shared solver's estimate.
fn iterative_solve_bytes(nodes: u64, entries: u64) -> Result<u64, ResourcePlanningError> {
    let n =
        usize::try_from(nodes).map_err(|_| ResourcePlanningError::Overflow("electrical nodes"))?;
    let nnz = usize::try_from(entries)
        .map_err(|_| ResourcePlanningError::Overflow("electrical couplings"))?;
    let bytes = runmat_sparse_solvers::estimate::amg_bytes(n, nnz, 1).saturating_add(
        runmat_sparse_solvers::estimate::krylov_workspace_bytes(
            runmat_sparse_solvers::estimate::KrylovMethod::Pcg,
            n,
            8,
        ),
    );
    Ok(bytes.saturating_add(bytes / 4))
}
