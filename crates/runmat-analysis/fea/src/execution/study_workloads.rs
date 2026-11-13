use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{validate_model_sequence, AnalysisModel, ProcedureStepId};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StructuralBucklingTemplate,
    StructuralFrequencyWorkload, StructuralLinearWorkload, StructuralModalFrequencyTemplate,
    StructuralModalResponseSpectrumTemplate, StructuralModalTransientTemplate,
    StructuralModalWorkload, StructuralPrestressedFrequencySolverTemplate,
    StructuralPrestressedFrequencyTemplate, StructuralPrestressedModalTemplate,
    StructuralRandomResponseTemplate, StructuralTransientWorkload, ThermalWorkload,
};

const STUDY_WORKLOADS_PREFIX: &[u8] = b"runmat-structural-study-workloads\0";
const STUDY_WORKLOADS_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 64));
const MAXIMUM_TRANSIENT_BLOCK_INTERVALS: u64 = 1_000_000;

mod prestress_prefix;
pub use prestress_prefix::StructuralPrestressPrefix;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientBlockPlan {
    maximum_intervals_per_block: u64,
}

impl StructuralTransientBlockPlan {
    pub fn new(maximum_intervals_per_block: u64) -> AnalysisExecutionResult<Self> {
        let plan = Self {
            maximum_intervals_per_block,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub const fn maximum_intervals_per_block(self) -> u64 {
        self.maximum_intervals_per_block
    }

    pub fn terminal_schedule_ordinals(
        self,
        terminal_schedule_ordinal: u64,
    ) -> AnalysisExecutionResult<Vec<u64>> {
        self.validate()?;
        if terminal_schedule_ordinal == 0 {
            return Err(invalid(
                "structural transient block plan requires a nonempty schedule",
            ));
        }
        let block_count = terminal_schedule_ordinal
            .checked_add(self.maximum_intervals_per_block - 1)
            .and_then(|value| value.checked_div(self.maximum_intervals_per_block))
            .ok_or_else(|| invalid("structural transient block count overflow"))?;
        let capacity = usize::try_from(block_count)
            .map_err(|_| invalid("structural transient block count exceeds addressable memory"))?;
        let mut terminals = Vec::with_capacity(capacity);
        let mut terminal = self.maximum_intervals_per_block;
        while terminal < terminal_schedule_ordinal {
            terminals.push(terminal);
            terminal = terminal
                .checked_add(self.maximum_intervals_per_block)
                .ok_or_else(|| invalid("structural transient block ordinal overflow"))?;
        }
        terminals.push(terminal_schedule_ordinal);
        Ok(terminals)
    }

    fn validate(self) -> AnalysisExecutionResult<()> {
        if self.maximum_intervals_per_block == 0
            || self.maximum_intervals_per_block > MAXIMUM_TRANSIENT_BLOCK_INTERVALS
        {
            return Err(invalid(
                "structural transient block plan is empty or exceeds its bound",
            ));
        }
        Ok(())
    }
}

/// One unresolved authored structural procedure in a heterogeneous sequence.
///
/// Producer results and state identities are deliberately absent. Modal transient and buckling
/// consumers remain templates until the sequence coordinator publishes and binds the upstream
/// modal basis or geometric-stiffness state. This inventory is numerical intent, not an execution
/// or placement plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StructuralStepWorkload {
    Thermal {
        workload: Box<ThermalWorkload>,
    },
    Linear {
        workload: Box<StructuralLinearWorkload>,
    },
    ModalExtraction {
        workload: Box<StructuralModalWorkload>,
    },
    ModalTransient {
        template: Box<StructuralModalTransientTemplate>,
        block_plan: StructuralTransientBlockPlan,
    },
    ModalFrequency {
        template: Box<StructuralModalFrequencyTemplate>,
    },
    ModalResponseSpectrum {
        template: Box<StructuralModalResponseSpectrumTemplate>,
    },
    RandomResponse {
        template: Box<StructuralRandomResponseTemplate>,
    },
    PrestressedModal {
        template: Box<StructuralPrestressedModalTemplate>,
    },
    Buckling {
        template: Box<StructuralBucklingTemplate>,
    },
    PrestressedFrequency {
        template: Box<StructuralPrestressedFrequencyTemplate>,
    },
}

impl StructuralStepWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        match self {
            Self::Thermal { workload } => workload
                .validate_selected_study_intent()
                .map_err(|error| invalid(error.to_string())),
            Self::Linear { workload } => workload.validate_sequence_template(),
            Self::ModalExtraction { workload } => workload.validate(),
            Self::ModalTransient {
                template,
                block_plan,
            } => {
                template.validate()?;
                block_plan.validate()
            }
            Self::ModalFrequency { template } => template.validate(),
            Self::ModalResponseSpectrum { template } => template.validate(),
            Self::RandomResponse { template } => template.validate(),
            Self::PrestressedModal { template } => template.validate(),
            Self::Buckling { template } => template.validate(),
            Self::PrestressedFrequency { template } => template.validate(),
        }
    }

    pub fn authored_model(&self) -> &AnalysisModel {
        match self {
            Self::Thermal { workload } => &workload.model,
            Self::Linear { workload } => &workload.model,
            Self::ModalExtraction { workload } => workload.model(),
            Self::ModalTransient { template, .. } => &template.model,
            Self::ModalFrequency { template } => &template.model,
            Self::ModalResponseSpectrum { template } => &template.model,
            Self::RandomResponse { template } => &template.model,
            Self::PrestressedModal { template } => &template.model,
            Self::Buckling { template } => &template.model,
            Self::PrestressedFrequency { template } => &template.model,
        }
    }

    pub fn step_id(&self) -> &ProcedureStepId {
        match self {
            Self::Thermal { workload } => &workload.step_id,
            Self::Linear { workload } => &workload.step_id,
            Self::ModalExtraction { workload } => workload.step_id(),
            Self::ModalTransient { template, .. } => &template.step_id,
            Self::ModalFrequency { template } => &template.step_id,
            Self::ModalResponseSpectrum { template } => &template.step_id,
            Self::RandomResponse { template } => &template.step_id,
            Self::PrestressedModal { template } => &template.step_id,
            Self::Buckling { template } => &template.step_id,
            Self::PrestressedFrequency { template } => &template.step_id,
        }
    }

    pub const fn backend(&self) -> crate::ComputeBackend {
        match self {
            Self::Thermal { workload } => workload.backend,
            Self::Linear { workload } => workload.backend,
            Self::ModalExtraction { workload } => workload.backend(),
            Self::ModalTransient { template, .. } => template.backend,
            Self::ModalFrequency { template } => template.backend,
            Self::ModalResponseSpectrum { template } => template.backend,
            Self::RandomResponse { template } => template.backend,
            Self::PrestressedModal { template } => template.backend,
            Self::Buckling { template } => template.backend,
            Self::PrestressedFrequency { template } => template.backend,
        }
    }

    fn replace_model(&mut self, model: AnalysisModel) -> AnalysisExecutionResult<()> {
        match self {
            Self::Thermal { workload } => workload.model = model,
            Self::Linear { workload } => workload.model = model,
            Self::ModalExtraction { workload } => workload.replace_model(model)?,
            Self::ModalTransient { template, .. } => template.model = model,
            Self::ModalFrequency { template } => template.model = model,
            Self::ModalResponseSpectrum { template } => template.model = model,
            Self::RandomResponse { template } => template.model = model,
            Self::PrestressedModal { template } => template.model = model,
            Self::Buckling { template } => template.model = model,
            Self::PrestressedFrequency { template } => template.model = model,
        }
        self.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StructuralStudyWorkloads {
    HeterogeneousSequence {
        workloads: BTreeMap<ProcedureStepId, StructuralStepWorkload>,
    },
    LinearSequence {
        workloads: BTreeMap<ProcedureStepId, StructuralLinearWorkload>,
    },
    StructuralTransient {
        workload: Box<StructuralTransientWorkload>,
        block_plan: StructuralTransientBlockPlan,
    },
    StructuralFrequency {
        workload: Box<StructuralFrequencyWorkload>,
    },
    StructuralModal {
        workload: Box<StructuralModalWorkload>,
    },
}

impl StructuralStudyWorkloads {
    pub fn validate_linear_sequence_workloads(
        workloads: &BTreeMap<ProcedureStepId, StructuralLinearWorkload>,
    ) -> AnalysisExecutionResult<()> {
        validate_linear_sequence(workloads)
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        match self {
            Self::HeterogeneousSequence { workloads } => {
                validate_heterogeneous_sequence(workloads).map(|_| ())
            }
            Self::LinearSequence { workloads } => validate_linear_sequence(workloads),
            Self::StructuralTransient {
                workload,
                block_plan,
            } => {
                workload.validate()?;
                block_plan.validate()?;
                if !matches!(
                    workload.solver,
                    super::StructuralTransientSolverPlan::Direct
                ) || workload.model.sequence.steps.len() != 1
                    || workload.model.sequence.steps[0].step_id != workload.step_id
                {
                    return Err(invalid(
                        "standalone structural transient execution requires one direct authored step; modal response must resolve its basis through an authored sequence",
                    ));
                }
                Ok(())
            }
            Self::StructuralFrequency { workload } => {
                workload.validate()?;
                if workload.model().sequence.steps.len() != 1
                    || workload.model().sequence.steps[0].step_id != *workload.step_id()
                {
                    return Err(invalid(
                        "standalone structural frequency execution requires exactly one authored step",
                    ));
                }
                Ok(())
            }
            Self::StructuralModal { workload } => {
                workload.validate()?;
                if workload.model().sequence.steps.len() != 1
                    || workload.model().sequence.steps[0].step_id != *workload.step_id()
                {
                    return Err(invalid(
                        "standalone structural modal execution requires exactly one authored step",
                    ));
                }
                Ok(())
            }
        }
    }

    pub fn authored_model(&self) -> AnalysisExecutionResult<&AnalysisModel> {
        self.validate()?;
        match self {
            Self::HeterogeneousSequence { workloads } => workloads
                .values()
                .next()
                .map(StructuralStepWorkload::authored_model)
                .ok_or_else(|| invalid("structural heterogeneous sequence has no workload")),
            Self::LinearSequence { workloads } => workloads
                .values()
                .next()
                .map(|workload| &workload.model)
                .ok_or_else(|| invalid("structural linear sequence has no workload")),
            Self::StructuralTransient { workload, .. } => Ok(&workload.model),
            Self::StructuralFrequency { workload } => Ok(workload.model()),
            Self::StructuralModal { workload } => Ok(workload.model()),
        }
    }

    pub fn replace_model(&mut self, model: AnalysisModel) -> AnalysisExecutionResult<()> {
        let mut replacement = self.clone();
        replacement.replace_model_in_place(model)?;
        replacement.validate()?;
        *self = replacement;
        Ok(())
    }

    fn replace_model_in_place(&mut self, model: AnalysisModel) -> AnalysisExecutionResult<()> {
        match self {
            Self::HeterogeneousSequence { workloads } => {
                for workload in workloads.values_mut() {
                    workload.replace_model(model.clone())?;
                }
            }
            Self::LinearSequence { workloads } => {
                for workload in workloads.values_mut() {
                    workload.model = model.clone();
                }
            }
            Self::StructuralTransient { workload, .. } => workload.model = model,
            Self::StructuralFrequency { workload } => workload.replace_model(model)?,
            Self::StructuralModal { workload } => workload.replace_model(model)?,
        }
        Ok(())
    }

    pub fn step_ids(&self) -> AnalysisExecutionResult<Vec<ProcedureStepId>> {
        self.validate()?;
        Ok(match self {
            Self::HeterogeneousSequence { workloads } => {
                authored_step_order(validate_heterogeneous_sequence(workloads)?)?
            }
            Self::LinearSequence { workloads } => workloads.keys().cloned().collect(),
            Self::StructuralTransient { workload, .. } => vec![workload.step_id.clone()],
            Self::StructuralFrequency { workload } => vec![workload.step_id().clone()],
            Self::StructuralModal { workload } => vec![workload.step_id().clone()],
        })
    }

    pub fn backends(&self) -> AnalysisExecutionResult<Vec<crate::ComputeBackend>> {
        self.validate()?;
        Ok(match self {
            Self::HeterogeneousSequence { workloads } => {
                authored_step_order(validate_heterogeneous_sequence(workloads)?)?
                    .iter()
                    .map(|step_id| {
                        workloads
                            .get(step_id)
                            .map(StructuralStepWorkload::backend)
                            .ok_or({
                                AnalysisExecutionError::Identity(
                            "validated heterogeneous step is absent from its workload inventory",
                        )
                            })
                    })
                    .collect::<AnalysisExecutionResult<Vec<_>>>()?
            }
            Self::LinearSequence { workloads } => workloads
                .values()
                .map(|workload| workload.backend)
                .collect(),
            Self::StructuralTransient { workload, .. } => vec![workload.backend],
            Self::StructuralFrequency { workload } => vec![workload.backend()],
            Self::StructuralModal { workload } => vec![workload.backend()],
        })
    }

    pub fn canonical_digest(&self) -> AnalysisExecutionResult<Digest> {
        Ok(Digest::sha256(self.canonical_bytes()?))
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            STUDY_WORKLOADS_PREFIX,
            "structural-study-workloads",
            self,
            STUDY_WORKLOADS_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let workloads: Self = runmat_canonical_codec::decode_contract(
            STUDY_WORKLOADS_PREFIX,
            "structural-study-workloads",
            bytes,
            STUDY_WORKLOADS_LIMITS,
        )?;
        workloads.validate()?;
        Ok(workloads)
    }
}

fn validate_linear_sequence(
    workloads: &BTreeMap<ProcedureStepId, StructuralLinearWorkload>,
) -> AnalysisExecutionResult<()> {
    for (step_id, workload) in workloads {
        workload.validate_sequence_template()?;
        if workload.state_input.is_some() {
            return Err(invalid(
                "sequence preparation does not accept a prebound state transition",
            ));
        }
        if workload.step_id != *step_id {
            return Err(AnalysisExecutionError::Identity(
                "structural sequence workload differs from its inventory key",
            ));
        }
    }
    validate_exact_sequence_coverage(
        workloads
            .iter()
            .map(|(key, workload)| (key, &workload.model, &workload.step_id)),
    )
    .map(|_| ())
}

fn validate_heterogeneous_sequence(
    workloads: &BTreeMap<ProcedureStepId, StructuralStepWorkload>,
) -> AnalysisExecutionResult<&AnalysisModel> {
    for workload in workloads.values() {
        workload.validate()?;
    }
    let model = validate_exact_sequence_coverage(
        workloads
            .iter()
            .map(|(key, workload)| (key, workload.authored_model(), workload.step_id())),
    )?;
    let order = authored_step_order(model)?;
    let ordered = order
        .iter()
        .map(|step_id| {
            workloads
                .get(step_id)
                .ok_or(AnalysisExecutionError::Identity(
                    "validated heterogeneous step is absent from its workload inventory",
                ))
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let mechanical = prestress_prefix::mechanical_suffix(model, &ordered)?;
    prestress_prefix::validate_consumer_chain(mechanical)?;
    Ok(model)
}

fn validate_exact_sequence_coverage<'a>(
    entries: impl IntoIterator<Item = (&'a ProcedureStepId, &'a AnalysisModel, &'a ProcedureStepId)>,
) -> AnalysisExecutionResult<&'a AnalysisModel> {
    let entries = entries.into_iter().collect::<Vec<_>>();
    let (_, model, _) = entries
        .first()
        .copied()
        .ok_or_else(|| invalid("structural sequence preparation has no step workloads"))?;
    let sequence = validate_model_sequence(model)
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let expected_steps = sequence
        .sequence()
        .steps
        .iter()
        .map(|step| step.step_id.clone())
        .collect::<BTreeSet<_>>();
    if entries
        .iter()
        .map(|(key, _, _)| (*key).clone())
        .collect::<BTreeSet<_>>()
        != expected_steps
    {
        return Err(invalid(
            "structural sequence workloads do not exactly cover the authored steps",
        ));
    }
    if entries
        .iter()
        .any(|(key, candidate, step_id)| *candidate != model || *key != *step_id)
    {
        return Err(AnalysisExecutionError::Identity(
            "structural sequence workload differs from its authored model or inventory key",
        ));
    }
    Ok(model)
}

fn authored_step_order(model: &AnalysisModel) -> AnalysisExecutionResult<Vec<ProcedureStepId>> {
    Ok(validate_model_sequence(model)
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?
        .topological_order()
        .to_vec())
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}

#[cfg(test)]
mod tests;
