use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, AnalysisWorkloadIdentity, ExecutionPlanIdentity,
    StepIdentity, StudyIdentity,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::resource::Capability;
use runmat_execution::value::ValueRefKind;
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{AnalysisBridgeError, AnalysisBridgeResult, ANALYSIS_HOST_ABI};

const WORK_CONTRACT_PREFIX: &[u8] = b"runmat-analysis-work-contract/v1\0";
const WORK_CONTRACT_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 65_536, 4096, 32);
const MAX_INPUTS: usize = 4096;
const MAX_OUTPUTS: usize = 4096;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AnalysisCapabilityRequirement {
    Host {
        abi: String,
    },
    ProcessIsolation,
    DistributedRuntime,
    Accelerator {
        class: runmat_execution::resource::AcceleratorClass,
    },
    Formulation {
        identity: Digest,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisWorkloadInput {
    pub kind: AnalysisNumericalObjectKind,
    pub logical_identity: Digest,
    pub payload_digest: Digest,
    pub manifest_digest: Digest,
    pub manifest_encoded_length: u64,
    pub reference_kind: ValueRefKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisWorkloadRequest {
    pub schema_version: u16,
    pub study: StudyIdentity,
    pub step: StepIdentity,
    pub execution_plan: ExecutionPlanIdentity,
    pub identity: AnalysisWorkloadIdentity,
    pub kind: AnalysisWorkKind,
    pub inputs: Vec<AnalysisWorkloadInput>,
    pub outputs: Vec<AnalysisNumericalObjectKind>,
    pub required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
}

impl AnalysisWorkloadRequest {
    pub fn new(
        study: StudyIdentity,
        step: StepIdentity,
        execution_plan: ExecutionPlanIdentity,
        kind: AnalysisWorkKind,
        inputs: Vec<AnalysisWorkloadInput>,
        outputs: Vec<AnalysisNumericalObjectKind>,
        required_capabilities: BTreeSet<AnalysisCapabilityRequirement>,
    ) -> AnalysisBridgeResult<Self> {
        let contract_digest = work_contract_digest(&inputs, &outputs, &required_capabilities)?;
        let identity = AnalysisWorkloadIdentity::new(
            study,
            step,
            execution_plan,
            kind,
            contract_digest,
            inputs.iter().map(|input| input.payload_digest).collect(),
        )?;
        let request = Self {
            schema_version: 1,
            study,
            step,
            execution_plan,
            identity,
            kind,
            inputs,
            outputs,
            required_capabilities,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        if self.schema_version != 1
            || self.inputs.is_empty()
            || self.inputs.len() > MAX_INPUTS
            || self.outputs.is_empty()
            || self.outputs.len() > MAX_OUTPUTS
        {
            return Err(AnalysisBridgeError::Invalid(
                "unsupported schema or invalid input/output count".into(),
            ));
        }
        let host_requirements = self
            .required_capabilities
            .iter()
            .filter(|requirement| {
                matches!(
                    requirement,
                    AnalysisCapabilityRequirement::Host { abi } if abi == ANALYSIS_HOST_ABI
                )
            })
            .count();
        if host_requirements != 1
            || self.inputs.iter().any(|input| {
                input.manifest_encoded_length == 0
                    || matches!(
                        input.reference_kind,
                        ValueRefKind::ResidentObject | ValueRefKind::SlicedObject
                    )
            })
        {
            return Err(AnalysisBridgeError::Invalid(
                "workload lacks its exact host capability or has an invalid input root".into(),
            ));
        }
        for capability in &self.required_capabilities {
            validate_capability(capability)?;
        }
        let expected = Self::new_unchecked_identity(
            self.study,
            self.step,
            self.execution_plan,
            self.kind,
            &self.inputs,
            &self.outputs,
            &self.required_capabilities,
        )?;
        if self.identity != expected {
            return Err(AnalysisBridgeError::Identity(
                "analysis workload identity does not converge",
            ));
        }
        Ok(())
    }

    fn new_unchecked_identity(
        study: StudyIdentity,
        step: StepIdentity,
        execution_plan: ExecutionPlanIdentity,
        kind: AnalysisWorkKind,
        inputs: &[AnalysisWorkloadInput],
        outputs: &[AnalysisNumericalObjectKind],
        capabilities: &BTreeSet<AnalysisCapabilityRequirement>,
    ) -> AnalysisBridgeResult<AnalysisWorkloadIdentity> {
        Ok(AnalysisWorkloadIdentity::new(
            study,
            step,
            execution_plan,
            kind,
            work_contract_digest(inputs, outputs, capabilities)?,
            inputs.iter().map(|input| input.payload_digest).collect(),
        )?)
    }

    pub fn generic_capabilities(&self) -> AnalysisBridgeResult<BTreeSet<Capability>> {
        self.validate()?;
        self.required_capabilities
            .iter()
            .map(map_capability)
            .collect()
    }
}

pub fn analysis_worker_capabilities(
    formulation_identities: impl IntoIterator<Item = Digest>,
    accelerator_classes: impl IntoIterator<Item = runmat_execution::resource::AcceleratorClass>,
    distributed: bool,
) -> BTreeSet<Capability> {
    let mut capabilities = BTreeSet::from([Capability::Custom(format!(
        "runmat.analysis.host:{ANALYSIS_HOST_ABI}"
    ))]);
    capabilities.extend(
        formulation_identities
            .into_iter()
            .map(|identity| Capability::Custom(format!("runmat.analysis.formulation:{identity}"))),
    );
    capabilities.extend(accelerator_classes.into_iter().map(Capability::Accelerator));
    if distributed {
        capabilities.insert(Capability::Custom(
            "runmat.analysis.distributed-runtime".into(),
        ));
    }
    capabilities
}

fn work_contract_digest(
    inputs: &[AnalysisWorkloadInput],
    outputs: &[AnalysisNumericalObjectKind],
    capabilities: &BTreeSet<AnalysisCapabilityRequirement>,
) -> AnalysisBridgeResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        WORK_CONTRACT_PREFIX,
        "work-contract",
        &(
            inputs
                .iter()
                .map(|input| (input.kind, input.logical_identity, input.payload_digest))
                .collect::<Vec<_>>(),
            outputs,
            capabilities,
        ),
        WORK_CONTRACT_LIMITS,
    )?))
}

fn validate_capability(requirement: &AnalysisCapabilityRequirement) -> AnalysisBridgeResult<()> {
    let text = match requirement {
        // Accelerator classes are validated tokens by construction.
        AnalysisCapabilityRequirement::Host { abi } => Some(abi),
        _ => None,
    };
    if text.is_some_and(|value| {
        value.is_empty()
            || value.len() > 256
            || !value.is_ascii()
            || value.chars().any(char::is_control)
    }) {
        return Err(AnalysisBridgeError::Invalid(
            "analysis capability token is invalid".into(),
        ));
    }
    Ok(())
}

fn map_capability(requirement: &AnalysisCapabilityRequirement) -> AnalysisBridgeResult<Capability> {
    validate_capability(requirement)?;
    Ok(match requirement {
        AnalysisCapabilityRequirement::Host { abi } => {
            Capability::Custom(format!("runmat.analysis.host:{abi}"))
        }
        AnalysisCapabilityRequirement::ProcessIsolation => Capability::ProcessIsolation,
        AnalysisCapabilityRequirement::DistributedRuntime => {
            Capability::Custom("runmat.analysis.distributed-runtime".into())
        }
        AnalysisCapabilityRequirement::Accelerator { class } => {
            Capability::Accelerator(class.clone())
        }
        AnalysisCapabilityRequirement::Formulation { identity } => {
            Capability::Custom(format!("runmat.analysis.formulation:{identity}"))
        }
    })
}
