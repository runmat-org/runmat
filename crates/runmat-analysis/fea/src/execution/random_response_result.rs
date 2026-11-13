use std::collections::BTreeSet;

use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::solve::dynamics::{
    assemble_described_random_response_intervals, ModalRandomResponseResult,
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    NumericalReceiptIdentity, SolverResultIdentity,
    StructuralRandomResponseAssemblyRequestIdentity, StructuralRandomResponseIntervalResult,
    StructuralRandomResponseScope, StructuralRandomResponseWorkload,
};

const RESULT_PREFIX: &[u8] = b"runmat-structural-random-response-result/v1\0";
const RESULT_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));

/// Deterministic fan-in request for canonical frequency-interval tasks.
///
/// Task completion order is discarded. Each task may carry several contiguous canonical
/// intervals, but assembly always reduces the individual intervals in global ordinal order so
/// task sizing, retries, and worker placement cannot change the numerical result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralRandomResponseAssemblyRequestWire"
)]
pub struct StructuralRandomResponseAssemblyRequest {
    schema_version: u16,
    identity: StructuralRandomResponseAssemblyRequestIdentity,
    scope: StructuralRandomResponseScope,
    tasks: Vec<StructuralRandomResponseIntervalResult>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralRandomResponseAssemblyRequestWire {
    schema_version: u16,
    identity: StructuralRandomResponseAssemblyRequestIdentity,
    scope: StructuralRandomResponseScope,
    tasks: Vec<StructuralRandomResponseIntervalResult>,
}

impl TryFrom<StructuralRandomResponseAssemblyRequestWire>
    for StructuralRandomResponseAssemblyRequest
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralRandomResponseAssemblyRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            scope: wire.scope,
            tasks: wire.tasks,
        };
        request.validate()?;
        Ok(request)
    }
}

impl StructuralRandomResponseAssemblyRequest {
    pub fn new(
        scope: StructuralRandomResponseScope,
        mut tasks: Vec<StructuralRandomResponseIntervalResult>,
    ) -> AnalysisExecutionResult<Self> {
        tasks.sort_by_key(|task| task.request().task_ordinal());
        let mut request = Self {
            schema_version: 1,
            identity: StructuralRandomResponseAssemblyRequestIdentity::derive(
                "structural-random-response-assembly-request-placeholder",
                &1_u8,
            )?,
            scope,
            tasks,
        };
        request.identity = request.derive_identity()?;
        request.validate()?;
        Ok(request)
    }

    pub const fn identity(&self) -> StructuralRandomResponseAssemblyRequestIdentity {
        self.identity
    }

    pub fn scope(&self) -> &StructuralRandomResponseScope {
        &self.scope
    }

    pub fn tasks(&self) -> &[StructuralRandomResponseIntervalResult] {
        &self.tasks
    }

    pub fn assemble(&self) -> AnalysisExecutionResult<ModalRandomResponseResult> {
        self.validate()?;
        let intervals = self
            .tasks
            .iter()
            .flat_map(|task| task.intervals().iter().cloned())
            .collect();
        assemble_described_random_response_intervals(self.scope.plan(), intervals)
            .map_err(|error| invalid(error.to_string()))
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        let expected_task_count = self.scope.task_ranges().len();
        let task_ordinals = self
            .tasks
            .iter()
            .map(|task| task.request().task_ordinal())
            .collect::<BTreeSet<_>>();
        if self.schema_version != 1
            || self.tasks.len() != expected_task_count
            || task_ordinals.len() != self.tasks.len()
            || self.tasks.iter().enumerate().any(|(ordinal, task)| {
                task.validate().is_err()
                    || task.request().task_ordinal() != ordinal
                    || task.request().scope() != &self.scope
            })
            || self.identity != self.derive_identity()?
        {
            return Err(invalid(
                "random-response assembly request is incomplete, unordered, or inconsistent",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            RESULT_PREFIX,
            "structural-random-response-assembly-request",
            self,
            RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            RESULT_PREFIX,
            "structural-random-response-assembly-request",
            bytes,
            RESULT_LIMITS,
        )?)
    }

    fn derive_identity(
        &self,
    ) -> AnalysisExecutionResult<StructuralRandomResponseAssemblyRequestIdentity> {
        StructuralRandomResponseAssemblyRequestIdentity::derive(
            "structural-random-response-assembly-request",
            &(
                self.schema_version,
                self.scope.identity(),
                self.tasks
                    .iter()
                    .map(StructuralRandomResponseIntervalResult::identity)
                    .collect::<Vec<_>>(),
            ),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralRandomResponseSolverResultWire"
)]
pub struct StructuralRandomResponseSolverResult {
    schema_version: u16,
    assembly: StructuralRandomResponseAssemblyRequest,
    numerical: ModalRandomResponseResult,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralRandomResponseSolverResultWire {
    schema_version: u16,
    assembly: StructuralRandomResponseAssemblyRequest,
    numerical: ModalRandomResponseResult,
}

impl TryFrom<StructuralRandomResponseSolverResultWire> for StructuralRandomResponseSolverResult {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralRandomResponseSolverResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            assembly: wire.assembly,
            numerical: wire.numerical,
        };
        result.validate()?;
        Ok(result)
    }
}

impl StructuralRandomResponseSolverResult {
    pub fn new(
        workload: &StructuralRandomResponseWorkload,
        assembly: StructuralRandomResponseAssemblyRequest,
    ) -> AnalysisExecutionResult<Self> {
        assembly.scope().validate_for(workload)?;
        let numerical = assembly.assemble()?;
        let result = Self {
            schema_version: 1,
            assembly,
            numerical,
        };
        result.validate_for(workload)?;
        Ok(result)
    }

    pub fn assembly(&self) -> &StructuralRandomResponseAssemblyRequest {
        &self.assembly
    }

    pub fn numerical(&self) -> &ModalRandomResponseResult {
        &self.numerical
    }

    pub fn validate_for(
        &self,
        workload: &StructuralRandomResponseWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate()?;
        self.assembly.scope().validate_for(workload)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.assembly.validate()?;
        if self.schema_version != 1 || self.numerical != self.assembly.assemble()? {
            return Err(invalid(
                "random-response solver result differs from its canonical interval assembly",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            RESULT_PREFIX,
            "structural-random-response-result",
            self,
            RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            RESULT_PREFIX,
            "structural-random-response-result",
            bytes,
            RESULT_LIMITS,
        )?)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    try_from = "StructuralRandomResponseNumericalReceiptWire"
)]
pub struct StructuralRandomResponseNumericalReceipt {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    source_payload_digest: Digest,
    task_count: usize,
    interval_count: usize,
    selected_mode_count: usize,
    minimum_captured_mass_ratio: f64,
    evaluated_frequency_count: usize,
    maximum_refinement_level_used: u8,
    maximum_estimated_relative_variance_error: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralRandomResponseNumericalReceiptWire {
    schema_version: u16,
    identity: NumericalReceiptIdentity,
    source_workload: AnalysisWorkloadIdentity,
    source_result: SolverResultIdentity,
    source_payload_digest: Digest,
    task_count: usize,
    interval_count: usize,
    selected_mode_count: usize,
    minimum_captured_mass_ratio: f64,
    evaluated_frequency_count: usize,
    maximum_refinement_level_used: u8,
    maximum_estimated_relative_variance_error: f64,
}

impl TryFrom<StructuralRandomResponseNumericalReceiptWire>
    for StructuralRandomResponseNumericalReceipt
{
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralRandomResponseNumericalReceiptWire) -> Result<Self, Self::Error> {
        let receipt = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            source_workload: wire.source_workload,
            source_result: wire.source_result,
            source_payload_digest: wire.source_payload_digest,
            task_count: wire.task_count,
            interval_count: wire.interval_count,
            selected_mode_count: wire.selected_mode_count,
            minimum_captured_mass_ratio: wire.minimum_captured_mass_ratio,
            evaluated_frequency_count: wire.evaluated_frequency_count,
            maximum_refinement_level_used: wire.maximum_refinement_level_used,
            maximum_estimated_relative_variance_error: wire
                .maximum_estimated_relative_variance_error,
        };
        receipt.validate_source()?;
        Ok(receipt)
    }
}

impl StructuralRandomResponseNumericalReceipt {
    pub fn new(
        source_workload: AnalysisWorkloadIdentity,
        result: &StructuralRandomResponseSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        let source_payload_digest = Digest::sha256(result.canonical_bytes()?);
        let source_result = SolverResultIdentity::new(source_workload, source_payload_digest)?;
        let numerical = result.numerical();
        let evidence = numerical.evidence();
        let mut receipt = Self {
            schema_version: 1,
            identity: NumericalReceiptIdentity::new(source_result, Digest::sha256([]))?,
            source_workload,
            source_result,
            source_payload_digest,
            task_count: result.assembly().tasks().len(),
            interval_count: result.assembly().scope().plan().interval_count(),
            selected_mode_count: numerical.selected_mode_count(),
            minimum_captured_mass_ratio: numerical
                .captured_mass_ratio_by_channel()
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
            evaluated_frequency_count: evidence.evaluated_frequency_count,
            maximum_refinement_level_used: evidence.maximum_refinement_level_used,
            maximum_estimated_relative_variance_error: evidence
                .maximum_estimated_relative_variance_error,
        };
        receipt.identity = receipt.derive_identity()?;
        receipt.validate_source()?;
        Ok(receipt)
    }

    pub const fn identity(&self) -> NumericalReceiptIdentity {
        self.identity
    }

    pub const fn source_result(&self) -> SolverResultIdentity {
        self.source_result
    }

    pub const fn source_workload(&self) -> AnalysisWorkloadIdentity {
        self.source_workload
    }

    pub const fn minimum_captured_mass_ratio(&self) -> f64 {
        self.minimum_captured_mass_ratio
    }

    pub const fn maximum_estimated_relative_variance_error(&self) -> f64 {
        self.maximum_estimated_relative_variance_error
    }

    pub fn validate_against(
        &self,
        source_result: SolverResultIdentity,
        result: &StructuralRandomResponseSolverResult,
    ) -> AnalysisExecutionResult<()> {
        if self != &Self::new(self.source_workload, result)? || self.source_result != source_result
        {
            return Err(AnalysisExecutionError::Identity(
                "random-response receipt differs from its source result",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_source()?;
        Ok(encode_contract(
            RESULT_PREFIX,
            "structural-random-response-receipt",
            self,
            RESULT_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            RESULT_PREFIX,
            "structural-random-response-receipt",
            bytes,
            RESULT_LIMITS,
        )?)
    }

    pub fn validate_source(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || self.task_count == 0
            || self.interval_count < self.task_count
            || self.selected_mode_count == 0
            || !self.minimum_captured_mass_ratio.is_finite()
            || !(0.0..=1.0 + 1.0e-12).contains(&self.minimum_captured_mass_ratio)
            || self.evaluated_frequency_count < 3 * self.interval_count
            || self.maximum_refinement_level_used == 0
            || !nonnegative_finite(self.maximum_estimated_relative_variance_error)
            || self.source_result
                != SolverResultIdentity::new(self.source_workload, self.source_payload_digest)?
            || self.identity != self.derive_identity()?
        {
            return Err(invalid("random-response numerical receipt is invalid"));
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<NumericalReceiptIdentity> {
        let evidence = Digest::sha256(encode_contract(
            RESULT_PREFIX,
            "structural-random-response-receipt-evidence",
            &(
                self.task_count,
                self.interval_count,
                self.selected_mode_count,
                self.minimum_captured_mass_ratio,
                self.evaluated_frequency_count,
                self.maximum_refinement_level_used,
                self.maximum_estimated_relative_variance_error,
            ),
            RESULT_LIMITS,
        )?);
        NumericalReceiptIdentity::new(self.source_result, evidence)
    }
}

/// Compact terminal statistics state for downstream consumers and publication.
///
/// Interval task artifacts remain in the solver result. This state retains the exact physical
/// scope and assembled RMS/variance fields, avoiding a second copy of every interval artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRandomStatistics {
    scope: StructuralRandomResponseScope,
    numerical: ModalRandomResponseResult,
}

impl StructuralRandomStatistics {
    pub fn from_result(
        result: &StructuralRandomResponseSolverResult,
    ) -> AnalysisExecutionResult<Self> {
        result.validate()?;
        let statistics = Self {
            scope: result.assembly().scope().clone(),
            numerical: result.numerical().clone(),
        };
        statistics.validate()?;
        Ok(statistics)
    }

    pub fn scope(&self) -> &StructuralRandomResponseScope {
        &self.scope
    }

    pub fn numerical(&self) -> &ModalRandomResponseResult {
        &self.numerical
    }

    pub fn validate_for(
        &self,
        result: &StructuralRandomResponseSolverResult,
    ) -> AnalysisExecutionResult<()> {
        if self != &Self::from_result(result)? {
            return Err(AnalysisExecutionError::Identity(
                "structural random statistics differ from their solver result",
            ));
        }
        Ok(())
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        self.numerical
            .canonical_bytes()
            .map_err(|error| invalid(error.to_string()))?;
        let plan = self.scope.plan();
        if self.numerical.basis() != plan.basis()
            || self.numerical.excitation() != plan.excitation()
            || self.numerical.excitation_digest() != plan.excitation_digest()
            || self.numerical.policy() != plan.policy()
            || self.numerical.selected_mode_count() != plan.selected_mode_count()
            || self.numerical.captured_mass_ratio_by_channel()
                != plan.captured_mass_ratio_by_channel()
            || self.numerical.dof_variance().len() != plan.physical_dof_count()
        {
            return Err(invalid(
                "structural random statistics differ from their canonical scope",
            ));
        }
        Ok(())
    }
}

fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}
