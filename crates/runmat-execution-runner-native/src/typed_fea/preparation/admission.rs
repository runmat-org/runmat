use runmat_analysis_core::{
    AnalysisProcedure, AnalysisSequence, PhysicsFamily, RestartPolicy, SolutionSpace,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedFamily {
    Acoustic,
    Structural,
    Thermal,
}

/// Family dispatch only. Exact material/source/mesh/numerical admission stays with
/// the existing compiled study product and canonical worker, not this pre-mesh gate.
pub(super) fn select_family(sequence: &AnalysisSequence) -> Result<PreparedFamily, String> {
    if sequence
        .steps
        .iter()
        .all(|step| step.family == PhysicsFamily::Structural)
    {
        return Ok(PreparedFamily::Structural);
    }
    let [step] = sequence.steps.as_slice() else {
        return Err(unsupported());
    };
    if step.family == PhysicsFamily::Acoustic
        && step.procedure == AnalysisProcedure::HarmonicResponse
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        return Ok(PreparedFamily::Acoustic);
    }
    if step.family != PhysicsFamily::Thermal
        || !matches!(
            step.procedure,
            AnalysisProcedure::Static | AnalysisProcedure::TransientResponse
        )
        || step.solution_space != SolutionSpace::Direct
        || step.restart != RestartPolicy::Fresh
        || !runmat_analysis_fea::study::thermal_step_initial_selection_supported(step)
        || !sequence.dependencies.is_empty()
        || !sequence.transitions.is_empty()
        || !sequence.branches.is_empty()
    {
        return Err(unsupported());
    }
    Ok(PreparedFamily::Thermal)
}

fn unsupported() -> String {
    "RM.FEA.TYPED_EXECUTION.UNSUPPORTED_PHYSICS_FAMILY: typed native execution admits structural studies or one fresh independent direct thermal steady/transient or acoustic harmonic step".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::ProcedureStep;

    #[test]
    fn fresh_acoustic_harmonic_selects_without_admitting_restart_or_cross_physics() {
        let acoustic = ProcedureStep::acoustic_harmonic("sound");
        assert_eq!(
            select_family(&AnalysisSequence::from_steps(
                "sound",
                vec![acoustic.clone()]
            ))
            .unwrap(),
            PreparedFamily::Acoustic
        );
        let mut resumed = acoustic.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "checkpoint".into(),
            required_state_digest: "digest".into(),
        };
        let mut modal = acoustic.clone();
        modal.solution_space = SolutionSpace::ModalReduced;
        for steps in [
            vec![resumed],
            vec![modal],
            vec![acoustic, ProcedureStep::thermal_steady("heat")],
        ] {
            assert!(select_family(&AnalysisSequence::from_steps("unsupported", steps)).is_err());
        }
    }

    #[test]
    fn standalone_thermal_and_existing_structural_families_select_without_cad() {
        for step in [
            ProcedureStep::thermal_steady("heat"),
            ProcedureStep::thermal_transient("heat"),
        ] {
            assert_eq!(
                select_family(&AnalysisSequence::from_steps("heat", vec![step])).unwrap(),
                PreparedFamily::Thermal
            );
        }
        assert_eq!(
            select_family(&AnalysisSequence::from_steps(
                "solid",
                vec![ProcedureStep::structural_static("solid")]
            ))
            .unwrap(),
            PreparedFamily::Structural
        );
    }

    #[test]
    fn thermal_checkpoint_mixed_and_duplicate_steps_stay_closed() {
        let thermal = ProcedureStep::thermal_steady("heat");
        let mut restart = thermal.clone();
        restart.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "checkpoint".into(),
            required_state_digest: "digest".into(),
        };
        for steps in [
            vec![restart],
            vec![thermal.clone(), ProcedureStep::structural_static("solid")],
            vec![thermal.clone(), ProcedureStep::thermal_transient("other")],
        ] {
            assert!(
                select_family(&AnalysisSequence::from_steps("unsupported", steps))
                    .unwrap_err()
                    .contains("UNSUPPORTED_PHYSICS_FAMILY")
            );
        }
        let mut wrong = thermal;
        wrong.family = PhysicsFamily::Acoustic;
        assert!(select_family(&AnalysisSequence::from_steps("acoustic", vec![wrong])).is_err());
    }

    #[test]
    fn physical_thermal_initial_selection_is_admitted_but_other_state_lifecycles_are_not() {
        use runmat_analysis_core::{
            ActivationLifecycle, InitialStateConflictPolicy, ProcedureStepId, StateId, StateInput,
            StateKind, StateSource, StateUse, StepObjectActivation, StepObjectRef,
        };
        let mut selected = ProcedureStep::thermal_transient("heat");
        selected.state_inputs.push(StateInput {
            input_id: "temperature".into(),
            required_kind: StateKind::Thermal,
            source: StateSource::InitialConditions,
            use_as: StateUse::PhysicalInitialState,
            conflict: InitialStateConflictPolicy::Reject,
        });
        selected.activations.push(StepObjectActivation {
            object: StepObjectRef::InitialCondition("chosen".into()),
            lifecycle: ActivationLifecycle::Activate,
        });
        assert_eq!(
            select_family(&AnalysisSequence::from_steps(
                "heat",
                vec![selected.clone()]
            ))
            .unwrap(),
            PreparedFamily::Thermal
        );
        let mut variants = Vec::new();
        let mut wrong = selected.clone();
        wrong.state_inputs[0].source = StateSource::Upstream {
            producer_step: ProcedureStepId("other".into()),
            state_id: StateId("state".into()),
        };
        variants.push(wrong);
        for kind in [StateKind::StructuralPrimary, StateKind::AcousticComplex] {
            let mut wrong = selected.clone();
            wrong.state_inputs[0].required_kind = kind;
            variants.push(wrong);
        }
        for use_as in [StateUse::InitialGuess, StateUse::Restart] {
            let mut wrong = selected.clone();
            wrong.state_inputs[0].use_as = use_as;
            variants.push(wrong);
        }
        let mut wrong = selected.clone();
        wrong.state_inputs[0].conflict = InitialStateConflictPolicy::AugmentDisjointQuantities;
        variants.push(wrong);
        for lifecycle in [
            ActivationLifecycle::Continue {
                from_step: ProcedureStepId("other".into()),
            },
            ActivationLifecycle::Deactivate {
                from_step: ProcedureStepId("other".into()),
            },
            ActivationLifecycle::Modify {
                prior_step: ProcedureStepId("other".into()),
                prior_object: StepObjectRef::InitialCondition("prior".into()),
            },
        ] {
            let mut wrong = selected.clone();
            wrong.activations[0].lifecycle = lifecycle;
            variants.push(wrong);
        }
        let mut wrong = selected;
        wrong.activations[0].object = StepObjectRef::MaterialParameter("parameter".into());
        variants.push(wrong);
        for step in variants {
            assert!(select_family(&AnalysisSequence::from_steps("heat", vec![step])).is_err());
        }
    }
}
