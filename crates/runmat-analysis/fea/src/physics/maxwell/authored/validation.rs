use super::*;

pub fn validate_canonical_input<'a>(
    model: &'a AnalysisModel,
    step_id: &str,
    requirement: FormulationRequirement,
    residual_target: f64,
) -> Result<&'a ProcedureStep, MaxwellError> {
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(invalid("relative residual target"));
    }
    if requirement != FormulationRequirement::Experimental {
        return Err(MaxwellError::Unsupported(
            "production Maxwell qualification".into(),
        ));
    }
    validate_model(model).map_err(invalid)?;
    let Some(ElectromagneticDomain::Harmonic {
        reference_frequency,
    }) = &model.electromagnetic
    else {
        return Err(invalid("distinct harmonic electromagnetic intent required"));
    };
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == step_id)
        .ok_or_else(|| invalid("harmonic step not found"))?;
    let frequency_matches = matches!(&step.independent_domain,
        IndependentDomain::FrequencyHertz(values) if values.len() == 1
            && values[0].is_finite() && values[0] > 0.0
            && values[0].to_bits() == reference_frequency.hertz.to_bits());
    if !frequency_matches {
        return Err(MaxwellError::Unsupported(
            "one exact step/domain harmonic frequency required; sweeps are not admitted".into(),
        ));
    }
    crate::results::observation::validate_sampled_requests(step)
        .map_err(|error| MaxwellError::Unsupported(error.to_string()))?;
    if step.family != PhysicsFamily::Electromagnetic
        || step.procedure != AnalysisProcedure::HarmonicResponse
        || step.solution_space != SolutionSpace::Direct
        || step.settings != ProcedureSettings::None
        || step.restart != RestartPolicy::Fresh
        || !step.state_inputs.is_empty()
        || step.state_outputs.iter().any(|output| {
            output.kind != StateKind::ElectromagneticComplex
                || output.completeness != StateCompleteness::Complete
        })
        || !step.activations.is_empty()
        || model.sequence.steps.len() != 1
        || !model.sequence.dependencies.is_empty()
        || !model.sequence.transitions.is_empty()
        || !model.sequence.branches.is_empty()
        || model.structural.is_some()
        || model.thermal.is_some()
        || model.thermo_mechanical.is_some()
        || model.electro_thermal.is_some()
        || model.cfd.is_some()
        || !model.connectors.is_empty()
        || !model.interfaces.is_empty()
        || !model.initial_conditions.is_empty()
        || !model.boundaries.is_empty()
        || !model.response_spectra.is_empty()
        || !model.random_excitations.is_empty()
    {
        return Err(MaxwellError::Unsupported(
            "standalone fresh direct harmonic step required; lifecycle/coupling not admitted"
                .into(),
        ));
    }
    if model
        .excitations
        .iter()
        .any(|action| !matches!(action, ExcitationAction::MagneticVolumeCurrentDensity(_)))
    {
        return Err(MaxwellError::Unsupported(
            "only impressed volume current actions are admitted".into(),
        ));
    }
    Ok(step)
}
