use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{validate_model_sequence, AnalysisModel, ProcedureStepId};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StructuralFrequencyWorkload,
    StructuralLinearWorkload, StructuralTransientWorkload,
};

const STUDY_WORKLOADS_PREFIX: &[u8] = b"runmat-structural-study-workloads\0";
const STUDY_WORKLOADS_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
const MAXIMUM_TRANSIENT_BLOCK_INTERVALS: u64 = 1_000_000;

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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StructuralStudyWorkloads {
    LinearSequence {
        workloads: BTreeMap<ProcedureStepId, StructuralLinearWorkload>,
    },
    DirectTransient {
        workload: Box<StructuralTransientWorkload>,
        block_plan: StructuralTransientBlockPlan,
    },
    StructuralFrequency {
        workload: Box<StructuralFrequencyWorkload>,
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
            Self::LinearSequence { workloads } => validate_linear_sequence(workloads),
            Self::DirectTransient {
                workload,
                block_plan,
            } => {
                workload.validate()?;
                block_plan.validate()?;
                if workload.model.sequence.steps.len() != 1
                    || workload.model.sequence.steps[0].step_id != workload.step_id
                {
                    return Err(invalid(
                        "direct structural transient execution requires exactly one authored step",
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
        }
    }

    pub fn authored_model(&self) -> AnalysisExecutionResult<&AnalysisModel> {
        self.validate()?;
        match self {
            Self::LinearSequence { workloads } => workloads
                .values()
                .next()
                .map(|workload| &workload.model)
                .ok_or_else(|| invalid("structural linear sequence has no workload")),
            Self::DirectTransient { workload, .. } => Ok(&workload.model),
            Self::StructuralFrequency { workload } => Ok(workload.model()),
        }
    }

    pub fn replace_model(&mut self, model: AnalysisModel) -> AnalysisExecutionResult<()> {
        match self {
            Self::LinearSequence { workloads } => {
                for workload in workloads.values_mut() {
                    workload.model = model.clone();
                }
            }
            Self::DirectTransient { workload, .. } => workload.model = model,
            Self::StructuralFrequency { workload } => workload.replace_model(model)?,
        }
        self.validate()
    }

    pub fn step_ids(&self) -> AnalysisExecutionResult<Vec<ProcedureStepId>> {
        self.validate()?;
        Ok(match self {
            Self::LinearSequence { workloads } => workloads.keys().cloned().collect(),
            Self::DirectTransient { workload, .. } => vec![workload.step_id.clone()],
            Self::StructuralFrequency { workload } => vec![workload.step_id().clone()],
        })
    }

    pub fn backends(&self) -> AnalysisExecutionResult<Vec<crate::ComputeBackend>> {
        self.validate()?;
        Ok(match self {
            Self::LinearSequence { workloads } => workloads
                .values()
                .map(|workload| workload.backend)
                .collect(),
            Self::DirectTransient { workload, .. } => vec![workload.backend],
            Self::StructuralFrequency { workload } => vec![workload.backend()],
        })
    }

    pub fn canonical_digest(&self) -> AnalysisExecutionResult<Digest> {
        self.validate()?;
        let bytes = encode_contract(
            STUDY_WORKLOADS_PREFIX,
            "structural-study-workloads",
            self,
            STUDY_WORKLOADS_LIMITS,
        )?;
        Ok(Digest::sha256(bytes))
    }
}

fn validate_linear_sequence(
    workloads: &BTreeMap<ProcedureStepId, StructuralLinearWorkload>,
) -> AnalysisExecutionResult<()> {
    let model = workloads
        .values()
        .next()
        .ok_or_else(|| invalid("structural sequence preparation has no step workloads"))?
        .model
        .clone();
    let sequence = validate_model_sequence(&model)
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let expected_steps = sequence
        .sequence()
        .steps
        .iter()
        .map(|step| step.step_id.clone())
        .collect::<BTreeSet<_>>();
    if workloads.keys().cloned().collect::<BTreeSet<_>>() != expected_steps {
        return Err(invalid(
            "structural sequence workloads do not exactly cover the authored steps",
        ));
    }
    for (step_id, workload) in workloads {
        workload.validate_sequence_template()?;
        if workload.model != model || workload.step_id != *step_id {
            return Err(AnalysisExecutionError::Identity(
                "structural sequence workload differs from its authored step or model",
            ));
        }
        if workload.initial_guess_transition.is_some() {
            return Err(invalid(
                "sequence preparation does not accept a prebound state transition",
            ));
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{IndependentDomain, ProcedureStep};

    use crate::execution::{
        StructuralFrequencySolverPlan, StructuralFrequencyWorkload,
        StructuralFrequencyWorkloadInput, StructuralTransientWorkload,
    };
    use crate::fixtures::{fixture_model, FixtureId};
    use crate::frequency::FrequencyResponsePlan;
    use crate::results::observation::ObservationExecutionPolicy;
    use crate::solve::dynamics::DirectHarmonicPolicy;
    use crate::solve::transient::StructuralTimeStepPolicy;
    use crate::ComputeBackend;

    use super::*;

    #[test]
    fn transient_block_plan_is_bounded_and_closes_the_terminal_schedule() {
        let plan = StructuralTransientBlockPlan::new(3).unwrap();
        assert_eq!(plan.terminal_schedule_ordinals(8).unwrap(), vec![3, 6, 8]);
        assert!(plan.terminal_schedule_ordinals(0).is_err());
        assert!(StructuralTransientBlockPlan::new(0).is_err());
        assert!(StructuralTransientBlockPlan::new(MAXIMUM_TRANSIENT_BLOCK_INTERVALS + 1).is_err());
    }

    #[test]
    fn direct_transient_workloads_are_strongly_typed_and_model_replaceable() {
        let model = fixture_model(FixtureId::StructuralTransientTet4Reference);
        let workload = StructuralTransientWorkload {
            schema_version: 1,
            step_id: model.sequence.steps[0].step_id.clone(),
            model,
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 8 },
            residual_target: 1.0e-10,
        };
        let mut workloads = StructuralStudyWorkloads::DirectTransient {
            workload: Box::new(workload),
            block_plan: StructuralTransientBlockPlan::new(3).unwrap(),
        };
        let digest = workloads.canonical_digest().unwrap();
        let mut replacement = workloads.authored_model().unwrap().clone();
        replacement.model_id.0 = "replacement-model".into();
        workloads.replace_model(replacement.clone()).unwrap();
        assert_eq!(workloads.authored_model().unwrap(), &replacement);
        assert_ne!(workloads.canonical_digest().unwrap(), digest);
    }

    #[test]
    fn structural_frequency_workload_is_canonical_and_model_replaceable() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let mut step = ProcedureStep::structural_harmonic("frequency");
        step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
        model.sequence.steps = vec![step];
        let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
            model,
            step_id: ProcedureStepId("frequency".into()),
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            observation_execution: ObservationExecutionPolicy::scalable(),
            plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
            solver: StructuralFrequencySolverPlan::Direct {
                policy: DirectHarmonicPolicy::default(),
            },
        })
        .unwrap();
        let mut workloads = StructuralStudyWorkloads::StructuralFrequency {
            workload: Box::new(workload),
        };
        let canonical = workloads.canonical_digest().unwrap();
        assert_eq!(
            workloads.step_ids().unwrap(),
            [ProcedureStepId("frequency".into())]
        );
        assert_eq!(workloads.backends().unwrap(), [ComputeBackend::Cpu]);

        let mut replacement = workloads.authored_model().unwrap().clone();
        replacement.model_id.0 = "resolved-frequency-model".into();
        workloads.replace_model(replacement.clone()).unwrap();
        assert_eq!(workloads.authored_model().unwrap(), &replacement);
        assert_ne!(workloads.canonical_digest().unwrap(), canonical);
    }

    #[test]
    fn standalone_structural_frequency_rejects_a_mixed_sequence() {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let mut frequency = ProcedureStep::structural_harmonic("frequency");
        frequency.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0]);
        model.sequence.steps.push(frequency.clone());
        let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
            model,
            step_id: frequency.step_id,
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            observation_execution: ObservationExecutionPolicy::scalable(),
            plan: FrequencyResponsePlan::new(20.0, &[], &[]).unwrap(),
            solver: StructuralFrequencySolverPlan::Direct {
                policy: DirectHarmonicPolicy::default(),
            },
        })
        .unwrap();
        assert!(StructuralStudyWorkloads::StructuralFrequency {
            workload: Box::new(workload),
        }
        .validate()
        .is_err());
    }
}
