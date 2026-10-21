use runmat_analysis_core::{
    AnalysisProcedure, AnalysisSequence, ElectromagneticDomain, PhysicsFamily, RestartPolicy,
    SolutionSpace,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PreparedFamily {
    Acoustic,
    Electrical,
    Fluid,
    Nonlinear,
    Magnetic,
    Maxwell,
    Structural,
    Thermal,
}

/// Family dispatch only. Exact material/source/mesh/numerical admission stays with
/// the existing compiled study product and canonical worker, not this pre-mesh gate.
pub(super) fn select_family(
    sequence: &AnalysisSequence,
    electromagnetic: Option<&ElectromagneticDomain>,
) -> Result<PreparedFamily, String> {
    if let [step] = sequence.steps.as_slice() {
        if step.family == PhysicsFamily::Structural
            && step.procedure == AnalysisProcedure::Nonlinear
            && step.solution_space == SolutionSpace::Direct
            && step.restart == RestartPolicy::Fresh
            && step.state_inputs.is_empty()
            && step.activations.is_empty()
            && sequence.dependencies.is_empty()
            && sequence.transitions.is_empty()
            && sequence.branches.is_empty()
        {
            return Ok(PreparedFamily::Nonlinear);
        }
    }
    if sequence.steps.iter().any(|step| {
        step.family == PhysicsFamily::Structural && step.procedure == AnalysisProcedure::Nonlinear
    }) {
        return Err(unsupported());
    }
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
    if step.family == PhysicsFamily::Fluid
        && step.procedure == AnalysisProcedure::Static
        && step.independent_domain == runmat_analysis_core::IndependentDomain::Static
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        return Ok(PreparedFamily::Fluid);
    }
    if step.family == PhysicsFamily::Electromagnetic
        && step.procedure == AnalysisProcedure::HarmonicResponse
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        if !matches!(
            electromagnetic,
            None | Some(ElectromagneticDomain::Harmonic { .. })
        ) {
            return Err(unsupported());
        }
        // Multi-frequency studies are fanned out into one case per frequency before
        // preparation, so each prepared harmonic step carries exactly one frequency.
        let runmat_analysis_core::IndependentDomain::FrequencyHertz(frequencies) =
            &step.independent_domain
        else {
            return Err(unsupported());
        };
        if frequencies.len() != 1 || !frequencies[0].is_finite() || frequencies[0] <= 0.0 {
            return Err(unsupported());
        }
        return Ok(PreparedFamily::Maxwell);
    }
    if step.family == PhysicsFamily::Electromagnetic
        && step.procedure == AnalysisProcedure::Static
        && step.solution_space == SolutionSpace::Direct
        && step.restart == RestartPolicy::Fresh
        && step.state_inputs.is_empty()
        && step.activations.is_empty()
        && sequence.dependencies.is_empty()
        && sequence.transitions.is_empty()
        && sequence.branches.is_empty()
    {
        return match electromagnetic {
            Some(ElectromagneticDomain::Static { .. }) => Ok(PreparedFamily::Electrical),
            Some(ElectromagneticDomain::Magnetostatic { .. }) => Ok(PreparedFamily::Magnetic),
            Some(ElectromagneticDomain::Harmonic { .. }) | None => Err(unsupported()),
        };
    }
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

/// One fresh electrothermal (Joule heating) analysis, steady or transient.
pub(in crate::typed_fea) fn is_electrothermal(sequence: &AnalysisSequence) -> bool {
    matches!(sequence.steps.as_slice(), [step]
        if step.family == PhysicsFamily::Coupled(runmat_analysis_core::CoupledPhysicsFamily::Electrothermal)
            && matches!(step.procedure, AnalysisProcedure::Static | AnalysisProcedure::TransientResponse)
            && step.solution_space == SolutionSpace::Direct
            && step.restart == RestartPolicy::Fresh
            && sequence.dependencies.is_empty()
            && sequence.transitions.is_empty()
            && sequence.branches.is_empty())
}

fn unsupported() -> String {
    "RM.FEA.TYPED_EXECUTION.UNSUPPORTED_PHYSICS_FAMILY: typed native execution admits structural studies or one fresh independent direct thermal steady/transient, fluid steady, equation-selected static electrical/magnetic, singleton Maxwell harmonic, or acoustic harmonic step".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_analysis_core::ProcedureStep;

    #[test]
    fn nonlinear_dispatch_preserves_fresh_standalone_authority() {
        let step = ProcedureStep::structural_nonlinear("plastic");
        assert_eq!(
            super::select_family(
                &AnalysisSequence::from_steps("plastic", vec![step.clone()]),
                None
            )
            .unwrap(),
            PreparedFamily::Nonlinear
        );
        let mut resumed = step.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "foreign".into(),
            required_state_digest: "foreign".into(),
        };
        let mut reduced = step.clone();
        reduced.solution_space = SolutionSpace::ModalReduced;
        for steps in [vec![resumed], vec![reduced], vec![step.clone(), step]] {
            assert!(
                super::select_family(&AnalysisSequence::from_steps("plastic", steps), None)
                    .is_err()
            );
        }
    }

    #[test]
    fn fluid_dispatch_requires_one_fresh_independent_direct_static_step() {
        let step = ProcedureStep::fluid_static("flow");
        let sequence = AnalysisSequence::from_steps("flow", vec![step.clone()]);
        assert_eq!(
            super::select_family(&sequence, None).unwrap(),
            PreparedFamily::Fluid
        );
        let mut transient = step.clone();
        transient.procedure = AnalysisProcedure::TransientResponse;
        let mut resumed = step.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "foreign".into(),
            required_state_digest: "foreign".into(),
        };
        let mut reduced = step.clone();
        reduced.solution_space = SolutionSpace::ModalReduced;
        for steps in [
            vec![transient],
            vec![resumed],
            vec![reduced],
            vec![step.clone(), step],
        ] {
            assert!(
                super::select_family(&AnalysisSequence::from_steps("flow", steps), None).is_err()
            );
        }
    }

    fn select_family(sequence: &AnalysisSequence) -> Result<PreparedFamily, String> {
        super::select_family(
            sequence,
            Some(&ElectromagneticDomain::Static {
                equation: runmat_analysis_core::StaticElectricalEquation::Electrostatics,
            }),
        )
    }

    #[test]
    fn maxwell_dispatch_requires_exact_singleton_fresh_harmonic_intent() {
        let domain = ElectromagneticDomain::Harmonic {
            reference_frequency: runmat_analysis_core::Frequency::new(2000.0),
        };
        let mut step = ProcedureStep::electromagnetic_harmonic("field");
        step.independent_domain =
            runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![2000.0]);
        let sequence = AnalysisSequence::from_steps("field", vec![step.clone()]);
        assert_eq!(
            super::select_family(&sequence, Some(&domain)).unwrap(),
            PreparedFamily::Maxwell
        );
        assert!(super::select_family(&sequence, None).is_err());
        assert!(super::select_family(
            &sequence,
            Some(&ElectromagneticDomain::Magnetostatic {
                formulation:
                    runmat_analysis_core::MagnetostaticFormulation::MagneticVectorPotential,
            })
        )
        .is_err());
        for frequencies in [
            vec![],
            vec![1.0],
            vec![2000.0, 3000.0],
            vec![f64::NAN],
            vec![0.0],
        ] {
            let mut changed = step.clone();
            changed.independent_domain =
                runmat_analysis_core::IndependentDomain::FrequencyHertz(frequencies);
            assert!(super::select_family(
                &AnalysisSequence::from_steps("field", vec![changed]),
                Some(&domain)
            )
            .is_err());
        }
        let mut resumed = step.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "missing".into(),
            required_state_digest: "missing".into(),
        };
        let mut reduced = step.clone();
        reduced.solution_space = SolutionSpace::ModalReduced;
        let mut static_step = step.clone();
        static_step.procedure = AnalysisProcedure::Static;
        for steps in [
            vec![resumed],
            vec![reduced],
            vec![static_step],
            vec![step.clone(), step],
        ] {
            assert!(super::select_family(
                &AnalysisSequence::from_steps("field", steps),
                Some(&domain)
            )
            .is_err());
        }
    }

    #[test]
    fn static_em_dispatch_requires_exact_authored_equation() {
        let sequence = AnalysisSequence::from_steps(
            "field",
            vec![ProcedureStep::electromagnetic_static("field")],
        );
        assert!(super::select_family(&sequence, None).is_err());
        assert!(super::select_family(
            &sequence,
            Some(&ElectromagneticDomain::Harmonic {
                reference_frequency: runmat_analysis_core::Frequency::new(1.0),
            })
        )
        .is_err());
        assert_eq!(
            super::select_family(
                &sequence,
                Some(&ElectromagneticDomain::Magnetostatic {
                    formulation:
                        runmat_analysis_core::MagnetostaticFormulation::MagneticVectorPotential,
                })
            )
            .unwrap(),
            PreparedFamily::Magnetic
        );
        for equation in [
            runmat_analysis_core::StaticElectricalEquation::Electrostatics,
            runmat_analysis_core::StaticElectricalEquation::SteadyConduction,
        ] {
            assert_eq!(
                super::select_family(&sequence, Some(&ElectromagneticDomain::Static { equation }))
                    .unwrap(),
                PreparedFamily::Electrical
            );
        }
    }

    #[test]
    fn static_electrical_family_selection_keeps_restart_and_mixed_sequences_closed() {
        let step = ProcedureStep::electromagnetic_static("potential");
        assert_eq!(
            select_family(&AnalysisSequence::from_steps(
                "potential",
                vec![step.clone()]
            ))
            .unwrap(),
            PreparedFamily::Electrical
        );
        let mut resumed = step.clone();
        resumed.restart = RestartPolicy::ResumeCheckpoint {
            checkpoint_id: "checkpoint".into(),
            required_state_digest: "digest".into(),
        };
        for steps in [
            vec![resumed],
            vec![step.clone(), step],
            vec![ProcedureStep::electromagnetic_harmonic("field")],
        ] {
            assert!(select_family(&AnalysisSequence::from_steps("unsupported", steps)).is_err());
        }
    }

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
        // Non-initial objects are validated by shared model/sequence planning, not
        // interpreted as thermal state selections by this pre-mesh family gate.
        let mut parameter = selected;
        parameter.activations[0].object = StepObjectRef::MaterialParameter("parameter".into());
        assert_eq!(
            select_family(&AnalysisSequence::from_steps("heat", vec![parameter])).unwrap(),
            PreparedFamily::Thermal
        );
        for step in variants {
            assert!(select_family(&AnalysisSequence::from_steps("heat", vec![step])).is_err());
        }
    }
}
