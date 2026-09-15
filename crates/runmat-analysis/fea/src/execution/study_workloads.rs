use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{validate_model_sequence, AnalysisModel, ProcedureStepId};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, StructuralBucklingTemplate,
    StructuralFrequencyWorkload, StructuralLinearWorkload, StructuralModalTransientTemplate,
    StructuralModalWorkload, StructuralTransientWorkload,
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

/// One unresolved authored structural procedure in a heterogeneous sequence.
///
/// Producer results and state identities are deliberately absent. Modal transient and buckling
/// consumers remain templates until the sequence coordinator publishes and binds the upstream
/// modal basis or geometric-stiffness state. This inventory is numerical intent, not an execution
/// or placement plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StructuralStepWorkload {
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
    Buckling {
        template: Box<StructuralBucklingTemplate>,
    },
}

impl StructuralStepWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        match self {
            Self::Linear { workload } => workload.validate_sequence_template(),
            Self::ModalExtraction { workload } => workload.validate(),
            Self::ModalTransient {
                template,
                block_plan,
            } => {
                template.validate()?;
                block_plan.validate()
            }
            Self::Buckling { template } => template.validate(),
        }
    }

    pub fn authored_model(&self) -> &AnalysisModel {
        match self {
            Self::Linear { workload } => &workload.model,
            Self::ModalExtraction { workload } => workload.model(),
            Self::ModalTransient { template, .. } => &template.model,
            Self::Buckling { template } => &template.model,
        }
    }

    pub fn step_id(&self) -> &ProcedureStepId {
        match self {
            Self::Linear { workload } => &workload.step_id,
            Self::ModalExtraction { workload } => workload.step_id(),
            Self::ModalTransient { template, .. } => &template.step_id,
            Self::Buckling { template } => &template.step_id,
        }
    }

    pub const fn backend(&self) -> crate::ComputeBackend {
        match self {
            Self::Linear { workload } => workload.backend,
            Self::ModalExtraction { workload } => workload.backend(),
            Self::ModalTransient { template, .. } => template.backend,
            Self::Buckling { template } => template.backend,
        }
    }

    fn replace_model(&mut self, model: AnalysisModel) -> AnalysisExecutionResult<()> {
        match self {
            Self::Linear { workload } => workload.model = model,
            Self::ModalExtraction { workload } => workload.replace_model(model)?,
            Self::ModalTransient { template, .. } => template.model = model,
            Self::Buckling { template } => template.model = model,
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
        if workload.initial_guess_transition.is_some() {
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
    let [producer_id, consumer_id] = order.as_slice() else {
        return Err(invalid(
            "heterogeneous structural execution requires exactly one producer and one state-dependent consumer",
        ));
    };
    let producer = workloads.get(producer_id).ok_or({
        AnalysisExecutionError::Identity(
            "validated heterogeneous producer is absent from its workload inventory",
        )
    })?;
    let consumer = workloads.get(consumer_id).ok_or({
        AnalysisExecutionError::Identity(
            "validated heterogeneous consumer is absent from its workload inventory",
        )
    })?;
    if !matches!(
        (producer, consumer),
        (
            StructuralStepWorkload::ModalExtraction { .. },
            StructuralStepWorkload::ModalTransient { .. }
        ) | (
            StructuralStepWorkload::Linear { .. },
            StructuralStepWorkload::Buckling { .. }
        )
    ) {
        return Err(invalid(
            "heterogeneous structural execution supports modal extraction followed by modal transient response, or linear preload followed by buckling",
        ));
    }
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
mod tests {
    use runmat_analysis_core::{
        FormulationRequirement, IndependentDomain, ProcedureStep, ReferenceStatePolicy,
        SequenceDependency, SequenceDependencyReason, StateId, StateTransition,
        StateTransitionMethod, TopologyTransitionPolicy, TransitionId,
    };

    use crate::execution::{
        StructuralFrequencySolverPlan, StructuralFrequencyWorkload,
        StructuralFrequencyWorkloadInput, StructuralModalTransientTemplate,
        StructuralModalWorkloadInput, StructuralTransientWorkload,
    };
    use crate::fixtures::{fixture_model, FixtureId};
    use crate::frequency::FrequencyResponsePlan;
    use crate::results::observation::ObservationExecutionPolicy;
    use crate::solve::dynamics::DirectHarmonicPolicy;
    use crate::solve::transient::StructuralTimeStepPolicy;
    use crate::ComputeBackend;

    use super::*;

    fn heterogeneous_modal_sequence() -> BTreeMap<ProcedureStepId, StructuralStepWorkload> {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let mut modal = ProcedureStep::structural_modal("z_modes");
        modal.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
        let transient =
            ProcedureStep::structural_modal_transient("a_response", "z_modes", "modal_basis");
        model.sequence.steps = vec![modal, transient];
        model.sequence.dependencies = vec![SequenceDependency {
            predecessor: ProcedureStepId("z_modes".into()),
            successor: ProcedureStepId("a_response".into()),
            reason: SequenceDependencyReason::StateTransfer,
        }];
        model.sequence.transitions = vec![StateTransition {
            transition_id: TransitionId("modal-basis-transfer".into()),
            producer_step: ProcedureStepId("z_modes".into()),
            consumer_step: ProcedureStepId("a_response".into()),
            state_id: StateId("modal_basis".into()),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        }];
        let modal = StructuralModalWorkload::new(StructuralModalWorkloadInput {
            model: model.clone(),
            step_id: ProcedureStepId("z_modes".into()),
            backend: ComputeBackend::Cpu,
            formulation_requirement: FormulationRequirement::Experimental,
            observation_execution: ObservationExecutionPolicy::scalable(),
            truncation: crate::ModalTruncationPolicy::requested(3),
            convergence: crate::ModalConvergencePolicy::exploratory(),
            prestress_transition: None,
        })
        .unwrap();
        let transient = StructuralModalTransientTemplate {
            schema_version: 1,
            model,
            step_id: ProcedureStepId("a_response".into()),
            backend: ComputeBackend::Cpu,
            formulation_requirement: FormulationRequirement::Experimental,
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 8 },
            residual_target: 1.0e-10,
        };
        BTreeMap::from([
            (
                ProcedureStepId("a_response".into()),
                StructuralStepWorkload::ModalTransient {
                    template: Box::new(transient),
                    block_plan: StructuralTransientBlockPlan::new(3).unwrap(),
                },
            ),
            (
                ProcedureStepId("z_modes".into()),
                StructuralStepWorkload::ModalExtraction {
                    workload: Box::new(modal),
                },
            ),
        ])
    }

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
            solver: crate::execution::StructuralTransientSolverPlan::Direct,
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 8 },
            residual_target: 1.0e-10,
        };
        let mut workloads = StructuralStudyWorkloads::StructuralTransient {
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

    #[test]
    fn heterogeneous_inventory_round_trips_and_reports_authored_order() {
        let study = StructuralStudyWorkloads::HeterogeneousSequence {
            workloads: heterogeneous_modal_sequence(),
        };
        assert_eq!(
            study.step_ids().unwrap(),
            [
                ProcedureStepId("z_modes".into()),
                ProcedureStepId("a_response".into())
            ]
        );
        let bytes = study.canonical_bytes().unwrap();
        let restored = StructuralStudyWorkloads::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(restored, study);
        assert_eq!(restored.canonical_digest().unwrap(), Digest::sha256(bytes));
    }

    #[test]
    fn failed_model_replacement_preserves_the_complete_workload_inventory() {
        let mut study = StructuralStudyWorkloads::HeterogeneousSequence {
            workloads: heterogeneous_modal_sequence(),
        };
        let original = study.clone();
        let mut invalid_model = study.authored_model().unwrap().clone();
        invalid_model.sequence.steps[0].state_outputs.clear();

        assert!(study.replace_model(invalid_model).is_err());
        assert_eq!(study, original);
        study.validate().unwrap();
    }

    #[test]
    fn heterogeneous_inventory_rejects_missing_and_substituted_steps() {
        let mut missing = heterogeneous_modal_sequence();
        missing.remove(&ProcedureStepId("a_response".into()));
        assert!(
            StructuralStudyWorkloads::HeterogeneousSequence { workloads: missing }
                .validate()
                .is_err()
        );

        let mut substituted = heterogeneous_modal_sequence();
        let response = substituted
            .remove(&ProcedureStepId("a_response".into()))
            .unwrap();
        substituted.insert(ProcedureStepId("z_modes".into()), response);
        assert!(StructuralStudyWorkloads::HeterogeneousSequence {
            workloads: substituted,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn modal_transient_inventory_is_an_unbound_template() {
        let workloads = heterogeneous_modal_sequence();
        let StructuralStepWorkload::ModalTransient { template, .. } =
            &workloads[&ProcedureStepId("a_response".into())]
        else {
            panic!("response step must remain a modal-transient template");
        };
        template.validate().unwrap();
        let encoded = serde_json::to_value(&workloads).unwrap();
        let encoded = encoded.to_string();
        assert!(!encoded.contains("basis_transition"));
        assert!(!encoded.contains("source_state"));
    }
}
