use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use serde::{Deserialize, Serialize};

use crate::frequency::FrequencyResponsePlanEntry;

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, ExecutionPlanIdentity, StepIdentity,
    StructuralFrequencyEntryRequestIdentity, StructuralFrequencyEntryResultIdentity,
    StructuralFrequencyEntrySolverEvidence, StructuralFrequencyPhysicalSample,
    StructuralFrequencyScope, StructuralFrequencySolutionSpace, StructuralFrequencySolverPlan,
    StructuralFrequencyWorkload, StructuralFrequencyWorkloadIdentity, StudyIdentity,
};

const FREQUENCY_ENTRY_PREFIX: &[u8] = b"runmat-structural-frequency-entry/v1\0";
const FREQUENCY_ENTRY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64);

/// One independently executable member of a structural frequency plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralFrequencyEntryRequestWire")]
pub struct StructuralFrequencyEntryRequest {
    schema_version: u16,
    identity: StructuralFrequencyEntryRequestIdentity,
    study: StudyIdentity,
    step: StepIdentity,
    execution_plan: ExecutionPlanIdentity,
    workload: StructuralFrequencyWorkloadIdentity,
    scope: StructuralFrequencyScope,
    entry: FrequencyResponsePlanEntry,
    solver: StructuralFrequencySolverPlan,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyEntryRequestWire {
    schema_version: u16,
    identity: StructuralFrequencyEntryRequestIdentity,
    study: StudyIdentity,
    step: StepIdentity,
    execution_plan: ExecutionPlanIdentity,
    workload: StructuralFrequencyWorkloadIdentity,
    scope: StructuralFrequencyScope,
    entry: FrequencyResponsePlanEntry,
    solver: StructuralFrequencySolverPlan,
}

impl TryFrom<StructuralFrequencyEntryRequestWire> for StructuralFrequencyEntryRequest {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencyEntryRequestWire) -> Result<Self, Self::Error> {
        let request = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            study: wire.study,
            step: wire.step,
            execution_plan: wire.execution_plan,
            workload: wire.workload,
            scope: wire.scope,
            entry: wire.entry,
            solver: wire.solver,
        };
        request.validate_identity()?;
        Ok(request)
    }
}

impl StructuralFrequencyEntryRequest {
    pub fn new(
        workload: &StructuralFrequencyWorkload,
        study: StudyIdentity,
        step: StepIdentity,
        execution_plan: ExecutionPlanIdentity,
        scope: StructuralFrequencyScope,
        entry: FrequencyResponsePlanEntry,
    ) -> AnalysisExecutionResult<Self> {
        workload.validate()?;
        scope.validate()?;
        if workload.plan() != scope.plan() {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency workload and physical scope use different plans",
            ));
        }
        scope
            .plan()
            .validate_entry(entry)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let mut request = Self {
            schema_version: 1,
            identity: StructuralFrequencyEntryRequestIdentity::derive(
                "structural-frequency-entry-request-placeholder",
                &1_u8,
            )?,
            study,
            step,
            execution_plan,
            workload: workload.identity(),
            scope,
            entry,
            solver: workload.solver().clone(),
        };
        request.identity = request.derive_identity()?;
        request.validate_for_workload(workload)?;
        Ok(request)
    }

    pub const fn identity(&self) -> StructuralFrequencyEntryRequestIdentity {
        self.identity
    }

    pub const fn workload(&self) -> StructuralFrequencyWorkloadIdentity {
        self.workload
    }

    pub const fn study(&self) -> StudyIdentity {
        self.study
    }

    pub const fn step(&self) -> StepIdentity {
        self.step
    }

    pub const fn execution_plan(&self) -> ExecutionPlanIdentity {
        self.execution_plan
    }

    pub fn scope(&self) -> &StructuralFrequencyScope {
        &self.scope
    }

    pub const fn entry(&self) -> FrequencyResponsePlanEntry {
        self.entry
    }

    pub fn solver(&self) -> &StructuralFrequencySolverPlan {
        &self.solver
    }

    pub fn validate_for_workload(
        &self,
        workload: &StructuralFrequencyWorkload,
    ) -> AnalysisExecutionResult<()> {
        self.validate_identity()?;
        workload.validate()?;
        if self.workload != workload.identity()
            || self.solver != *workload.solver()
            || self.scope.plan() != workload.plan()
        {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency entry request differs from its workload",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate_identity()?;
        Ok(encode_contract(
            FREQUENCY_ENTRY_PREFIX,
            "structural-frequency-entry-request",
            self,
            FREQUENCY_ENTRY_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            FREQUENCY_ENTRY_PREFIX,
            "structural-frequency-entry-request",
            bytes,
            FREQUENCY_ENTRY_LIMITS,
        )?)
    }

    fn validate_identity(&self) -> AnalysisExecutionResult<()> {
        self.scope.validate()?;
        self.scope
            .plan()
            .validate_entry(self.entry)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        self.solver.validate()?;
        if self.schema_version != 1 || self.identity != self.derive_identity()? {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency entry request identity does not converge",
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StructuralFrequencyEntryRequestIdentity> {
        StructuralFrequencyEntryRequestIdentity::derive(
            "structural-frequency-entry-request",
            &(
                self.schema_version,
                self.study,
                self.step,
                self.execution_plan,
                self.workload,
                &self.scope,
                self.entry,
                &self.solver,
            ),
        )
    }
}

/// Worker output for one frequency; final plan assembly remains a separate task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralFrequencyEntryResultWire")]
pub struct StructuralFrequencyEntryResult {
    schema_version: u16,
    identity: StructuralFrequencyEntryResultIdentity,
    request: StructuralFrequencyEntryRequest,
    sample: StructuralFrequencyPhysicalSample,
    solution_space: StructuralFrequencySolutionSpace,
    evidence: StructuralFrequencyEntrySolverEvidence,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralFrequencyEntryResultWire {
    schema_version: u16,
    identity: StructuralFrequencyEntryResultIdentity,
    request: StructuralFrequencyEntryRequest,
    sample: StructuralFrequencyPhysicalSample,
    solution_space: StructuralFrequencySolutionSpace,
    evidence: StructuralFrequencyEntrySolverEvidence,
}

impl TryFrom<StructuralFrequencyEntryResultWire> for StructuralFrequencyEntryResult {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralFrequencyEntryResultWire) -> Result<Self, Self::Error> {
        let result = Self {
            schema_version: wire.schema_version,
            identity: wire.identity,
            request: wire.request,
            sample: wire.sample,
            solution_space: wire.solution_space,
            evidence: wire.evidence,
        };
        result.validate()?;
        Ok(result)
    }
}

impl StructuralFrequencyEntryResult {
    pub fn new(
        request: StructuralFrequencyEntryRequest,
        sample: StructuralFrequencyPhysicalSample,
        solution_space: StructuralFrequencySolutionSpace,
        evidence: StructuralFrequencyEntrySolverEvidence,
    ) -> AnalysisExecutionResult<Self> {
        let mut result = Self {
            schema_version: 1,
            identity: StructuralFrequencyEntryResultIdentity::derive(
                "structural-frequency-entry-result-placeholder",
                &1_u8,
            )?,
            request,
            sample,
            solution_space,
            evidence,
        };
        result.identity = result.derive_identity()?;
        result.validate()?;
        Ok(result)
    }

    pub const fn identity(&self) -> StructuralFrequencyEntryResultIdentity {
        self.identity
    }

    pub fn request(&self) -> &StructuralFrequencyEntryRequest {
        &self.request
    }

    pub fn sample(&self) -> &StructuralFrequencyPhysicalSample {
        &self.sample
    }

    pub fn solution_space(&self) -> &StructuralFrequencySolutionSpace {
        &self.solution_space
    }

    pub fn evidence(&self) -> &StructuralFrequencyEntrySolverEvidence {
        &self.evidence
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        self.request.validate_identity()?;
        self.sample.validate_against(self.request.scope())?;
        self.solution_space.validate(self.request.scope())?;
        self.evidence
            .validate_for_sample(&self.sample, &self.solution_space)?;
        let method_matches = match (&self.request.solver, &self.solution_space) {
            (
                StructuralFrequencySolverPlan::Direct { .. },
                StructuralFrequencySolutionSpace::Direct,
            ) => true,
            (
                StructuralFrequencySolverPlan::ModalReduced {
                    basis_state: expected,
                },
                StructuralFrequencySolutionSpace::ModalReduced { basis_state, .. },
            ) => expected == basis_state,
            _ => false,
        };
        if self.schema_version != 1
            || !method_matches
            || self.request.entry != self.sample.entry()
            || self.identity != self.derive_identity()?
        {
            return Err(AnalysisExecutionError::Identity(
                "structural frequency entry result differs from its request or physical sample",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            FREQUENCY_ENTRY_PREFIX,
            "structural-frequency-entry-result",
            self,
            FREQUENCY_ENTRY_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        Ok(decode_contract(
            FREQUENCY_ENTRY_PREFIX,
            "structural-frequency-entry-result",
            bytes,
            FREQUENCY_ENTRY_LIMITS,
        )?)
    }

    fn derive_identity(&self) -> AnalysisExecutionResult<StructuralFrequencyEntryResultIdentity> {
        StructuralFrequencyEntryResultIdentity::derive(
            "structural-frequency-entry-result",
            &(
                self.schema_version,
                &self.request,
                &self.sample,
                &self.solution_space,
                &self.evidence,
            ),
        )
    }
}
