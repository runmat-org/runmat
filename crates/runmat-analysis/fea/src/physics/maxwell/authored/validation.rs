use super::*;

pub fn validate_canonical_input<'a>(
    model: &'a AnalysisModel,
    step_id: &str,
    residual_target: f64,
) -> Result<&'a ProcedureStep, MaxwellError> {
    if !residual_target.is_finite() || residual_target <= 0.0 || residual_target >= 1.0 {
        return Err(invalid("relative residual target"));
    }
    validate_model(model).map_err(invalid)?;
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id.as_str() == step_id)
        .ok_or_else(|| invalid("harmonic step not found"))?;
    let IndependentDomain::FrequencyHertz(frequencies) = &step.independent_domain else {
        return Err(invalid("harmonic step requires authored frequencies"));
    };
    let mut seen = std::collections::BTreeSet::new();
    if frequencies.is_empty()
        || frequencies
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || !seen.insert(value.to_bits()))
    {
        return Err(invalid(
            "harmonic frequencies must be finite, positive and distinct",
        ));
    }
    // The step's frequencies are the analysis authority (each runs as its own case). An
    // authored harmonic domain is optional; its reference frequency only names the domain.
    match &model.electromagnetic {
        None => {}
        Some(ElectromagneticDomain::Harmonic {
            reference_frequency,
        }) if reference_frequency.canonical_value().is_finite()
            && reference_frequency.canonical_value() > 0.0 => {}
        Some(ElectromagneticDomain::Harmonic { .. }) => {
            return Err(invalid(
                "the model's harmonic reference frequency must be finite and positive",
            ))
        }
        Some(_) => {
            return Err(invalid(
                "a harmonic analysis needs a harmonic (or no) electromagnetic model domain",
            ))
        }
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
    if model.excitations.iter().any(|action| {
        !matches!(
            action,
            ExcitationAction::MagneticVolumeCurrentDensity(_)
                | ExcitationAction::ElectricCurrent(_)
                | ExcitationAction::TerminalVoltage(_)
        )
    }) {
        return Err(MaxwellError::Unsupported(
            "harmonic analysis admits impressed volume current density and terminal currents"
                .into(),
        ));
    }
    Ok(step)
}
