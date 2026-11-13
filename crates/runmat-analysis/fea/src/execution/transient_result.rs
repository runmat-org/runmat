use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    NumericalReceiptIdentity, SolverResultIdentity, StructuralTransientBlockRequestIdentity,
    StructuralTransientCheckpointIdentity, StructuralTransientCheckpointScope,
    StructuralTransientPhysicalSample, StructuralTransientPhysicalSampleIdentity,
    StructuralTransientResultAssemblyRequestIdentity, StructuralTransientSampleStep,
    StructuralTransientSegment, StructuralTransientSegmentIdentity,
    StructuralTransientSolverCounters,
};
use crate::contracts::{SolverMethod, SolverPreconditioner};
use crate::solve::{
    accelerator::AcceleratorSolveEvidence, backend::kind::LinearAlgebraBackendKind,
};

const TRANSIENT_RESULT_PREFIX: &[u8] = b"runmat-structural-transient-result/v1\0";
const TRANSIENT_RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralTransientResultAssemblyRequestWire"
)]
pub struct StructuralTransientResultAssemblyRequest {
    pub schema_version: u16,
    pub identity: StructuralTransientResultAssemblyRequestIdentity,
    pub scope: StructuralTransientCheckpointScope,
    pub segments: Vec<StructuralTransientSegmentIdentity>,
    pub terminal_checkpoint: StructuralTransientCheckpointIdentity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientResultAssemblyRequestWire {
    schema_version: u16,
    identity: StructuralTransientResultAssemblyRequestIdentity,
    scope: StructuralTransientCheckpointScope,
    segments: Vec<StructuralTransientSegmentIdentity>,
    terminal_checkpoint: StructuralTransientCheckpointIdentity,
}

impl TryFrom<StructuralTransientResultAssemblyRequestWire>
    for StructuralTransientResultAssemblyRequest
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientResultAssemblyRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            segments: wire.segments,
            terminal_checkpoint: wire.terminal_checkpoint,
        };
        request.validate_identity()?;
        Ok(request)
    }
}

impl StructuralTransientResultAssemblyRequest {
    pub fn new(segments: &[StructuralTransientSegment]) -> AnalysisExecutionResult<Self> {
        validate_segment_chain(segments)?;
        let first = &segments[0];
        let terminal = &segments[segments.len() - 1];
        let segment_identities = segments
            .iter()
            .map(|segment| segment.manifest().identity)
            .collect::<Vec<_>>();
        let terminal_checkpoint = terminal.manifest().terminal_checkpoint.identity;
        let scope = first.manifest().request.scope.clone();
        let identity =
            assembly_request_identity(1, &scope, &segment_identities, terminal_checkpoint)?;
        Ok(Self {
            schema_version: 1,
            identity,
            scope,
            segments: segment_identities,
            terminal_checkpoint,
        })
    }

    pub fn validate_for(
        &self,
        segments: &[StructuralTransientSegment],
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        validate_segment_chain(segments)?;
        if self.scope != segments[0].manifest().request.scope
            || self.segments
                != segments
                    .iter()
                    .map(|segment| segment.manifest().identity)
                    .collect::<Vec<_>>()
            || self.terminal_checkpoint
                != segments[segments.len() - 1]
                    .manifest()
                    .terminal_checkpoint
                    .identity
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient result assembly inputs differ from their request",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_identity()?;
        Ok(encode_contract(
            TRANSIENT_RESULT_PREFIX,
            "structural-transient-result-assembly-request",
            self,
            TRANSIENT_RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            TRANSIENT_RESULT_PREFIX,
            "structural-transient-result-assembly-request",
            bytes,
            TRANSIENT_RESULT_LIMITS,
        )?)
    }

    fn validate_identity(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        if self.schema_version != 1
            || self.segments.is_empty()
            || self.identity
                != assembly_request_identity(
                    self.schema_version,
                    &self.scope,
                    &self.segments,
                    self.terminal_checkpoint,
                )?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient result assembly request identity does not converge",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientResultEntry {
    pub schedule_ordinal: u64,
    pub sample: StructuralTransientPhysicalSampleIdentity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralTransientSolverResultManifestWire"
)]
pub struct StructuralTransientSolverResultManifest {
    pub schema_version: u16,
    pub identity: SolverResultIdentity,
    pub source_workload: AnalysisWorkloadIdentity,
    pub scope: StructuralTransientCheckpointScope,
    pub samples: Vec<StructuralTransientResultEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientSolverResultManifestWire {
    schema_version: u16,
    identity: SolverResultIdentity,
    source_workload: AnalysisWorkloadIdentity,
    scope: StructuralTransientCheckpointScope,
    samples: Vec<StructuralTransientResultEntry>,
}

impl TryFrom<StructuralTransientSolverResultManifestWire>
    for StructuralTransientSolverResultManifest
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientSolverResultManifestWire) -> Result<Self, Self::Error> {
        let manifest = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_workload: wire.source_workload,
            scope: wire.scope,
            samples: wire.samples,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

impl StructuralTransientSolverResultManifest {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        let expected_count = self.scope.schedule.points.len();
        if self.schema_version != 1
            || self.source_workload != self.scope.source_workload
            || self.samples.len() != expected_count
            || self
                .samples
                .iter()
                .enumerate()
                .any(|(ordinal, entry)| u64::try_from(ordinal).ok() != Some(entry.schedule_ordinal))
            || self.identity != solver_result_identity(&self.scope, &self.samples)?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient solver-result manifest does not converge",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            TRANSIENT_RESULT_PREFIX,
            "structural-transient-solver-result-manifest",
            self,
            TRANSIENT_RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            TRANSIENT_RESULT_PREFIX,
            "structural-transient-solver-result-manifest",
            bytes,
            TRANSIENT_RESULT_LIMITS,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StructuralTransientSolverResult {
    manifest: StructuralTransientSolverResultManifest,
    pub(crate) samples: Vec<StructuralTransientPhysicalSample>,
}

impl StructuralTransientSolverResult {
    pub fn assemble(
        request: &StructuralTransientResultAssemblyRequest,
        segments: &[StructuralTransientSegment],
    ) -> AnalysisExecutionResult<Self> {
        request.validate_for(segments)?;
        let initial = segments[0].initial_sample().cloned().ok_or({
            AnalysisExecutionError::Identity(
                "structural transient result is missing its initial physical sample",
            )
        })?;
        let mut samples = Vec::with_capacity(request.scope.schedule.points.len());
        samples.push(initial);
        samples.extend(
            segments
                .iter()
                .flat_map(|segment| segment.samples())
                .map(|sample| sample.physical.clone()),
        );
        let entries = samples
            .iter()
            .map(|sample| StructuralTransientResultEntry {
                schedule_ordinal: sample.schedule_ordinal,
                sample: sample.identity,
            })
            .collect::<Vec<_>>();
        let manifest = StructuralTransientSolverResultManifest {
            schema_version: 1,
            identity: solver_result_identity(&request.scope, &entries)?,
            source_workload: request.scope.source_workload,
            scope: request.scope.clone(),
            samples: entries,
        };
        Self::from_manifest_and_samples(manifest, samples)
    }

    pub fn from_manifest_and_samples(
        manifest: StructuralTransientSolverResultManifest,
        samples: Vec<StructuralTransientPhysicalSample>,
    ) -> AnalysisExecutionResult<Self> {
        let result = Self { manifest, samples };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.manifest.validate()?;
        if self.samples.len() != self.manifest.samples.len() {
            return Err(AnalysisExecutionError::Identity(
                "structural transient solver-result payload count differs from its manifest",
            ));
        }
        for (ordinal, (entry, sample)) in
            self.manifest.samples.iter().zip(&self.samples).enumerate()
        {
            sample.validate_against(&self.manifest.scope)?;
            if u64::try_from(ordinal).ok() != Some(sample.schedule_ordinal)
                || entry.schedule_ordinal != sample.schedule_ordinal
                || entry.sample != sample.identity
                || (ordinal == 0) != matches!(sample.step, StructuralTransientSampleStep::Initial)
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural transient solver-result sample order or identity is invalid",
                ));
            }
        }
        Ok(())
    }

    pub fn manifest(&self) -> &StructuralTransientSolverResultManifest {
        &self.manifest
    }

    pub fn samples(&self) -> &[StructuralTransientPhysicalSample] {
        &self.samples
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientBlockReceipt {
    pub segment: StructuralTransientSegmentIdentity,
    pub request: StructuralTransientBlockRequestIdentity,
    pub terminal_checkpoint: StructuralTransientCheckpointIdentity,
    pub solver_method: SolverMethod,
    pub solver_backend: LinearAlgebraBackendKind,
    pub preconditioner: SolverPreconditioner,
    pub counters: StructuralTransientSolverCounters,
    pub accelerator_evidence: Vec<AcceleratorSolveEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralTransientNumericalReceiptWire"
)]
pub struct StructuralTransientNumericalReceipt {
    pub schema_version: u16,
    pub identity: NumericalReceiptIdentity,
    pub source_result: SolverResultIdentity,
    pub assembly_request: StructuralTransientResultAssemblyRequestIdentity,
    pub blocks: Vec<StructuralTransientBlockReceipt>,
    pub terminal_counters: StructuralTransientSolverCounters,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralTransientNumericalReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_result: SolverResultIdentity,
    assembly_request: StructuralTransientResultAssemblyRequestIdentity,
    blocks: Vec<StructuralTransientBlockReceipt>,
    terminal_counters: StructuralTransientSolverCounters,
}

impl TryFrom<StructuralTransientNumericalReceiptWire> for StructuralTransientNumericalReceipt {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralTransientNumericalReceiptWire) -> Result<Self, Self::Error> {
        let receipt = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_result: wire.source_result,
            assembly_request: wire.assembly_request,
            blocks: wire.blocks,
            terminal_counters: wire.terminal_counters,
        };
        receipt.validate_identity()?;
        Ok(receipt)
    }
}

impl StructuralTransientNumericalReceipt {
    pub fn new(
        result: &StructuralTransientSolverResult,
        request: &StructuralTransientResultAssemblyRequest,
        segments: &[StructuralTransientSegment],
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        request.validate_for(segments)?;
        let blocks = segments
            .iter()
            .map(|segment| {
                let manifest = segment.manifest();
                StructuralTransientBlockReceipt {
                    segment: manifest.identity,
                    request: manifest.request.identity,
                    terminal_checkpoint: manifest.terminal_checkpoint.identity,
                    solver_method: manifest.solver_evidence.solver_method,
                    solver_backend: manifest.solver_evidence.solver_backend,
                    preconditioner: manifest.solver_evidence.preconditioner,
                    counters: manifest.solver_evidence.counters.clone(),
                    accelerator_evidence: manifest.solver_evidence.accelerator_evidence.clone(),
                }
            })
            .collect::<Vec<_>>();
        let terminal_counters = segments[segments.len() - 1]
            .manifest()
            .terminal_checkpoint
            .counters
            .clone();
        let identity = receipt_identity(
            result.manifest.identity,
            request.identity,
            &blocks,
            &terminal_counters,
        )?;
        Ok(Self {
            schema_version: 1,
            identity,
            source_result: result.manifest.identity,
            assembly_request: request.identity,
            blocks,
            terminal_counters,
        })
    }

    pub fn validate_for(
        &self,
        result: &StructuralTransientSolverResult,
        request: &StructuralTransientResultAssemblyRequest,
        segments: &[StructuralTransientSegment],
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        let expected = Self::new(result, request, segments)?;
        if self != &expected {
            return Err(AnalysisExecutionError::Identity(
                "structural transient numerical receipt differs from its exact inputs",
            ));
        }
        Ok(())
    }

    /// Validates the self-contained numerical evidence retained with an imported solver result.
    ///
    /// The result-assembly task has already checked the full segment/request chain before it emits
    /// this receipt. Result consumers do not need to retain those potentially large intermediate
    /// segments, but they must still bind the receipt to the exact solver result and reject
    /// inconsistent aggregate accounting.
    pub fn validate_for_result(
        &self,
        result: &StructuralTransientSolverResult,
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        result.validate()?;
        if self.source_result != result.manifest().identity {
            return Err(AnalysisExecutionError::Identity(
                "structural transient numerical receipt names a different solver result",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_identity()?;
        Ok(encode_contract(
            TRANSIENT_RESULT_PREFIX,
            "structural-transient-numerical-receipt",
            self,
            TRANSIENT_RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            TRANSIENT_RESULT_PREFIX,
            "structural-transient-numerical-receipt",
            bytes,
            TRANSIENT_RESULT_LIMITS,
        )?)
    }

    fn validate_identity(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1 || self.blocks.is_empty() {
            return Err(AnalysisExecutionError::Identity(
                "structural transient numerical receipt identity does not converge",
            ));
        }
        let mut aggregate = StructuralTransientSolverCounters::default();
        let first = &self.blocks[0];
        for block in &self.blocks {
            super::StructuralTransientBlockSolverEvidence {
                solver_method: block.solver_method,
                solver_backend: block.solver_backend,
                preconditioner: block.preconditioner,
                counters: block.counters.clone(),
                accelerator_evidence: block.accelerator_evidence.clone(),
            }
            .validate()?;
            if block.solver_method != first.solver_method
                || block.solver_backend != first.solver_backend
                || block.preconditioner != first.preconditioner
            {
                return Err(AnalysisExecutionError::Invalid(
                    "structural transient numerical receipt changes solver semantics between blocks"
                        .into(),
                ));
            }
            aggregate.host_sync_count = aggregate
                .host_sync_count
                .checked_add(block.counters.host_sync_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient receipt host synchronization count overflow",
                ))?;
            aggregate.device_apply_k_count = aggregate
                .device_apply_k_count
                .checked_add(block.counters.device_apply_k_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient receipt device apply count overflow",
                ))?;
            aggregate.device_apply_k_attempt_count = aggregate
                .device_apply_k_attempt_count
                .checked_add(block.counters.device_apply_k_attempt_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient receipt device apply-attempt count overflow",
                ))?;
            aggregate.dense_factorization_count = aggregate
                .dense_factorization_count
                .checked_add(block.counters.dense_factorization_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient receipt dense factorization count overflow",
                ))?;
            aggregate.dense_solve_count = aggregate
                .dense_solve_count
                .checked_add(block.counters.dense_solve_count)
                .ok_or(AnalysisExecutionError::Limit(
                    "structural transient receipt dense solve count overflow",
                ))?;
        }
        if aggregate != self.terminal_counters
            || self.identity
                != receipt_identity(
                    self.source_result,
                    self.assembly_request,
                    &self.blocks,
                    &self.terminal_counters,
                )?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient numerical receipt identity or aggregate accounting does not converge",
            ));
        }
        Ok(())
    }
}

fn validate_segment_chain(segments: &[StructuralTransientSegment]) -> AnalysisExecutionResult<()> {
    let first = segments.first().ok_or_else(|| {
        AnalysisExecutionError::Invalid(
            "structural transient result assembly requires at least one segment".into(),
        )
    })?;
    first.validate()?;
    let scope = &first.manifest().request.scope;
    if first.manifest().request.block_ordinal != 0
        || first.manifest().request.start_schedule_ordinal != 0
        || first.manifest().request.predecessor.is_some()
        || first.initial_sample().is_none()
    {
        return Err(AnalysisExecutionError::Identity(
            "structural transient result chain does not begin at block and schedule ordinal zero",
        ));
    }
    for (index, segment) in segments.iter().enumerate() {
        segment.validate()?;
        let manifest = segment.manifest();
        if manifest.request.scope != *scope
            || u64::try_from(index).ok() != Some(manifest.request.block_ordinal)
            || (index == 0) != segment.initial_sample().is_some()
        {
            return Err(AnalysisExecutionError::Identity(
                "structural transient result segments differ in scope, order, or initial ownership",
            ));
        }
        if let Some(previous) = index.checked_sub(1).map(|value| &segments[value]) {
            let previous_checkpoint = &previous.manifest().terminal_checkpoint;
            if manifest.request.predecessor != Some(previous_checkpoint.identity)
                || manifest.request.start_schedule_ordinal
                    != previous_checkpoint.continuation.schedule_ordinal
                || manifest.request.start_counters != previous_checkpoint.counters
            {
                return Err(AnalysisExecutionError::Identity(
                    "structural transient result segment lineage is not contiguous",
                ));
            }
        }
    }
    let terminal = &segments[segments.len() - 1].manifest().terminal_checkpoint;
    let expected_terminal = u64::try_from(scope.schedule.points.len() - 1).map_err(|_| {
        AnalysisExecutionError::Limit(
            "structural transient result schedule exceeds platform ordinal limits",
        )
    })?;
    if terminal.continuation.schedule_ordinal != expected_terminal {
        return Err(AnalysisExecutionError::Identity(
            "structural transient result chain does not cover its complete schedule",
        ));
    }
    Ok(())
}

fn assembly_request_identity(
    schema_version: u16,
    scope: &StructuralTransientCheckpointScope,
    segments: &[StructuralTransientSegmentIdentity],
    terminal_checkpoint: StructuralTransientCheckpointIdentity,
) -> AnalysisExecutionResult<StructuralTransientResultAssemblyRequestIdentity> {
    StructuralTransientResultAssemblyRequestIdentity::derive(
        "structural-transient-result-assembly-request",
        &(schema_version, scope, segments, terminal_checkpoint),
    )
}

fn solver_result_identity(
    scope: &StructuralTransientCheckpointScope,
    samples: &[StructuralTransientResultEntry],
) -> AnalysisExecutionResult<SolverResultIdentity> {
    let content = encode_contract(
        TRANSIENT_RESULT_PREFIX,
        "structural-transient-solver-result-content",
        &(scope, samples),
        TRANSIENT_RESULT_LIMITS,
    )?;
    SolverResultIdentity::new(scope.source_workload, Digest::sha256(&content))
}

fn receipt_identity(
    source_result: SolverResultIdentity,
    assembly_request: StructuralTransientResultAssemblyRequestIdentity,
    blocks: &[StructuralTransientBlockReceipt],
    terminal_counters: &StructuralTransientSolverCounters,
) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
    let content = encode_contract(
        TRANSIENT_RESULT_PREFIX,
        "structural-transient-numerical-receipt-content",
        &(assembly_request, blocks, terminal_counters),
        TRANSIENT_RESULT_LIMITS,
    )?;
    NumericalReceiptIdentity::new(source_result, Digest::sha256(&content))
}

#[cfg(test)]
mod tests {
    use crate::contracts::StructuralDynamicBalanceSnapshot;
    use crate::execution::transient_checkpoint::tests::{checkpoint, scope};

    use super::*;
    use crate::execution::{
        StructuralTransientBlockRequest, StructuralTransientBlockSolverEvidence,
        StructuralTransientContinuationState, StructuralTransientPhysicalSampleInput,
        StructuralTransientSegmentSample, StructuralTransientSegmentSampleInput,
    };

    fn evidence(host_sync_count: u64) -> StructuralTransientBlockSolverEvidence {
        StructuralTransientBlockSolverEvidence {
            solver_method: SolverMethod::NewmarkAverageAccelerationPcg,
            solver_backend: LinearAlgebraBackendKind::CpuReference,
            preconditioner: SolverPreconditioner::None,
            counters: StructuralTransientSolverCounters {
                host_sync_count,
                ..StructuralTransientSolverCounters::default()
            },
            accelerator_evidence: Vec::new(),
        }
    }

    fn balance(value: f64) -> StructuralDynamicBalanceSnapshot {
        StructuralDynamicBalanceSnapshot {
            applied_load: vec![value, 0.0],
            inertia_force: vec![0.25 * value, 0.0],
            damping_force: vec![0.25 * value, 0.0],
            elastic_force: vec![0.5 * value, 0.0],
            equilibrium_residual: vec![0.0, 0.0],
        }
    }

    fn initial() -> StructuralTransientPhysicalSample {
        StructuralTransientPhysicalSample::new(
            &scope(),
            StructuralTransientPhysicalSampleInput {
                schedule_ordinal: 0,
                time_s: runmat_types::physical::Time::new(0.0),
                step: StructuralTransientSampleStep::Initial,
                state: StructuralTransientContinuationState {
                    displacement: vec![0.0, 0.0],
                    velocity: vec![0.0, 0.0],
                    acceleration: vec![0.0, 0.0],
                },
                projected_residual_norm: 0.0,
                reaction: vec![0.0, 0.0],
                strain_energy_j: runmat_types::physical::Energy::new(0.0),
                kinetic_energy_j: runmat_types::physical::Energy::new(0.0),
                applied_load_work_j: runmat_types::physical::Energy::new(0.0),
                support_work_j: runmat_types::physical::Energy::new(0.0),
                damping_dissipation_j: runmat_types::physical::Energy::new(0.0),
                energy_balance_residual_j: runmat_types::physical::Energy::new(0.0),
                relative_energy_balance_error: 0.0,
                dynamic_balance: balance(0.0),
            },
        )
        .unwrap()
    }

    fn accepted(
        request: &StructuralTransientBlockRequest,
        ordinal: u64,
    ) -> StructuralTransientSegmentSample {
        StructuralTransientSegmentSample::new(
            request,
            StructuralTransientSegmentSampleInput {
                schedule_ordinal: ordinal,
                time_s: runmat_types::physical::Time::new(ordinal as f64 * 0.5),
                accepted_time_step_s: runmat_types::physical::Time::new(0.5),
                state: StructuralTransientContinuationState {
                    displacement: vec![ordinal as f64, 0.0],
                    velocity: vec![0.5, 0.0],
                    acceleration: vec![if ordinal == 1 { 7.0 } else { 1.0 }, 0.0],
                },
                projected_residual_norm: 1.0e-12,
                reaction: vec![2.0, 0.0],
                strain_energy_j: runmat_types::physical::Energy::new(0.6),
                kinetic_energy_j: runmat_types::physical::Energy::new(0.4),
                applied_load_work_j: runmat_types::physical::Energy::new(ordinal as f64),
                support_work_j: runmat_types::physical::Energy::new(-0.25 * ordinal as f64),
                damping_dissipation_j: runmat_types::physical::Energy::new(0.1 * ordinal as f64),
                energy_balance_residual_j: runmat_types::physical::Energy::new(0.0),
                relative_energy_balance_error: 0.0,
                dynamic_balance: balance(1.0),
            },
        )
        .unwrap()
    }

    fn one_block() -> Vec<StructuralTransientSegment> {
        let request = StructuralTransientBlockRequest::new(scope(), None, 2).unwrap();
        vec![StructuralTransientSegment::new(
            request.clone(),
            checkpoint(None, 2),
            evidence(2),
            Some(initial()),
            vec![accepted(&request, 1), accepted(&request, 2)],
        )
        .unwrap()]
    }

    fn two_blocks() -> Vec<StructuralTransientSegment> {
        let first_request = StructuralTransientBlockRequest::new(scope(), None, 1).unwrap();
        let first_checkpoint = checkpoint(None, 1);
        let first = StructuralTransientSegment::new(
            first_request.clone(),
            first_checkpoint.clone(),
            evidence(1),
            Some(initial()),
            vec![accepted(&first_request, 1)],
        )
        .unwrap();
        let second_request =
            StructuralTransientBlockRequest::new(scope(), Some(&first_checkpoint), 2).unwrap();
        let second = StructuralTransientSegment::new(
            second_request.clone(),
            checkpoint(Some(&first_checkpoint), 2),
            evidence(1),
            None,
            vec![accepted(&second_request, 2)],
        )
        .unwrap();
        vec![first, second]
    }

    #[test]
    fn physical_result_is_canonical_across_block_layouts_while_receipt_retains_lineage() {
        let one = one_block();
        let one_request = StructuralTransientResultAssemblyRequest::new(&one).unwrap();
        let one_result = StructuralTransientSolverResult::assemble(&one_request, &one).unwrap();
        let one_receipt =
            StructuralTransientNumericalReceipt::new(&one_result, &one_request, &one).unwrap();

        let two = two_blocks();
        let two_request = StructuralTransientResultAssemblyRequest::new(&two).unwrap();
        let two_result = StructuralTransientSolverResult::assemble(&two_request, &two).unwrap();
        let two_receipt =
            StructuralTransientNumericalReceipt::new(&two_result, &two_request, &two).unwrap();

        assert_eq!(one_result, two_result);
        assert_eq!(
            one_result.manifest().canonical_bytes().unwrap(),
            two_result.manifest().canonical_bytes().unwrap()
        );
        assert_ne!(one_request.identity, two_request.identity);
        assert_ne!(one_receipt.identity, two_receipt.identity);
        assert_eq!(one_receipt.terminal_counters, two_receipt.terminal_counters);
    }

    #[test]
    fn result_assembly_rejects_missing_and_reordered_segments() {
        let segments = two_blocks();
        assert!(StructuralTransientResultAssemblyRequest::new(&segments[..1]).is_err());
        assert!(StructuralTransientResultAssemblyRequest::new(&[
            segments[1].clone(),
            segments[0].clone(),
        ])
        .is_err());
    }

    #[test]
    fn imported_receipt_rejects_a_self_consistent_different_result_binding() {
        let segments = one_block();
        let request = StructuralTransientResultAssemblyRequest::new(&segments).unwrap();
        let result = StructuralTransientSolverResult::assemble(&request, &segments).unwrap();
        let mut receipt =
            StructuralTransientNumericalReceipt::new(&result, &request, &segments).unwrap();
        receipt.source_result = SolverResultIdentity::new(
            result.manifest().scope.source_workload,
            Digest::sha256(b"substituted-transient-result"),
        )
        .unwrap();
        receipt.identity = receipt_identity(
            receipt.source_result,
            receipt.assembly_request,
            &receipt.blocks,
            &receipt.terminal_counters,
        )
        .unwrap();

        receipt.validate_identity().unwrap();
        assert!(receipt.validate_for_result(&result).is_err());
    }
}
