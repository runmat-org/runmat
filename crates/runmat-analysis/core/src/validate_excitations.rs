use std::collections::{HashMap, HashSet};

use crate::{
    problem::quantity::FiniteQuantity, AnalysisExcitation, AnalysisModel, AnalysisValidationError,
    ExcitationAction, ExcitationCaseActivation, ExcitationDistribution, ExcitationKinematics,
    ExcitationTarget, InitialCondition, InitialConditionTarget, InitialConditionValue,
    InitialValue, PhysicalQuantityKind, PhysicalSignal, ResponseSpectrum, SignalId, SignalShape,
};

pub(crate) fn validate_model_excitations(
    model: &AnalysisModel,
) -> Result<(), AnalysisValidationError> {
    let signals = model
        .signals
        .iter()
        .map(|signal| (signal.signal_id(), signal))
        .collect::<HashMap<_, _>>();
    let steps = model
        .sequence
        .steps
        .iter()
        .map(|step| step.step_id.as_str())
        .collect::<HashSet<_>>();
    validate_excitations(&model.excitations, &signals, &steps)?;
    validate_cases(&model.excitation_cases, &model.excitations, &steps)?;
    validate_case_selections(&model.excitation_cases, &model.sequence.steps)?;
    validate_initial_conditions(&model.initial_conditions, &signals, &steps)?;
    validate_response_spectra(&model.response_spectra, &signals)?;
    validate_random_excitations(&model.random_excitations)?;
    Ok(())
}

fn validate_case_selections(
    cases: &[crate::ExcitationCase],
    steps: &[crate::ProcedureStep],
) -> Result<(), AnalysisValidationError> {
    let known = cases
        .iter()
        .map(|case| &case.case_id)
        .collect::<HashSet<_>>();
    for step in steps {
        if let crate::ExcitationCaseSelection::Only(ids) = &step.excitation_case_selection {
            let mut selected = HashSet::new();
            for id in ids {
                if !known.contains(id) || !selected.insert(id) {
                    return Err(AnalysisValidationError::InvalidProcedureStep {
                        step_id: step.step_id.0.clone(),
                        detail: format!(
                            "selected excitation case `{}` must exist and occur only once",
                            id.0
                        ),
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_random_excitations(
    excitations: &[crate::RandomExcitationSpectrum],
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for excitation in excitations {
        let id = excitation.excitation_id.0.as_str();
        if !ids.insert(id) {
            return Err(AnalysisValidationError::InvalidRandomExcitation {
                excitation_id: id.to_string(),
                detail: "random-excitation ids must be unique".into(),
            });
        }
        if let Err(detail) = excitation.validate_intrinsic() {
            return Err(AnalysisValidationError::InvalidRandomExcitation {
                excitation_id: id.to_string(),
                detail,
            });
        }
    }
    Ok(())
}

fn validate_excitations(
    actions: &[ExcitationAction],
    signals: &HashMap<&SignalId, &PhysicalSignal>,
    steps: &HashSet<&str>,
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for action in actions {
        let excitation = action.excitation();
        let id = excitation.excitation_id.0.as_str();
        let result = (|| {
            if id.trim().is_empty() || !ids.insert(id) {
                return Err("excitation ids must be non-empty and unique".to_string());
            }
            let signal = signals
                .get(&excitation.signal_id)
                .ok_or_else(|| "referenced signal does not exist".to_string())?;
            if signal.quantity() != action.expected_quantity() {
                return Err(format!(
                    "action requires {:?}, but signal provides {:?}",
                    action.expected_quantity(),
                    signal.quantity()
                ));
            }
            if !(matches!(signal.shape(), SignalShape::Scalar)
                || signal.shape().topology_component_count() == Some(1) && signal.frame().is_none())
                && signal.frame() != Some(&excitation.frame)
            {
                return Err(
                    "vector or tensor signal frame must match the excitation frame".to_string(),
                );
            }
            validate_excitation_target(&excitation.target)?;
            crate::validate_authored::validate_coordinate_frame(&excitation.frame)
                .map_err(|error| error.to_string())?;
            if let Some(direction) = excitation.direction {
                validate_direction(direction.vector)?;
            }
            if !excitation.projection.relative_tolerance.is_finite()
                || excitation.projection.relative_tolerance <= 0.0
            {
                return Err("projection tolerance must be finite and positive".to_string());
            }
            if excitation
                .active_step_ids
                .iter()
                .any(|step| step.trim().is_empty() || !steps.contains(step.as_str()))
            {
                return Err("active steps must reference existing procedure steps".to_string());
            }
            if excitation
                .combination_group
                .as_ref()
                .is_some_and(|group| group.trim().is_empty())
            {
                return Err("combination group cannot be empty".to_string());
            }
            validate_action_policy(action, excitation, signal)
        })();
        if let Err(detail) = result {
            return Err(AnalysisValidationError::InvalidExcitation {
                excitation_id: id.to_string(),
                detail,
            });
        }
    }
    Ok(())
}

fn validate_excitation_target(target: &ExcitationTarget) -> Result<(), String> {
    match target {
        ExcitationTarget::Region(target) => crate::validate_authored::validate_target(target),
        ExcitationTarget::ReferencePoint(point) => {
            if point.reference_point_id.trim().is_empty() || !point.position.is_finite() {
                Err("reference-point target must have finite position and identity".to_string())
            } else {
                Ok(())
            }
        }
        ExcitationTarget::Port { port_id } if port_id.trim().is_empty() => {
            Err("port target requires an id".to_string())
        }
        ExcitationTarget::Interface { interface_id } if interface_id.trim().is_empty() => {
            Err("interface target requires an id".to_string())
        }
        _ => Ok(()),
    }
}

fn validate_direction(vector: [f64; 3]) -> Result<(), String> {
    if vector.into_iter().any(|component| !component.is_finite()) {
        return Err("direction must be finite".to_string());
    }
    let norm = vector
        .into_iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON || (norm - 1.0).abs() > 1.0e-10 {
        Err("direction must be a normalized nonzero vector".to_string())
    } else {
        Ok(())
    }
}

fn validate_action_policy(
    action: &ExcitationAction,
    excitation: &AnalysisExcitation,
    signal: &PhysicalSignal,
) -> Result<(), String> {
    let distribution_valid = match action {
        ExcitationAction::Force(_) | ExcitationAction::Moment(_) => matches!(
            excitation.distribution,
            ExcitationDistribution::Concentrated | ExcitationDistribution::PreserveResultant
        ),
        ExcitationAction::Traction(_)
        | ExcitationAction::Pressure(_)
        | ExcitationAction::HeatFlux(_) => matches!(
            excitation.distribution,
            ExcitationDistribution::IntensityPerArea | ExcitationDistribution::TopologyField
        ),
        ExcitationAction::BodyForce(_) | ExcitationAction::HeatGeneration(_) => matches!(
            excitation.distribution,
            ExcitationDistribution::IntensityPerVolume | ExcitationDistribution::TopologyField
        ),
        ExcitationAction::ElectricChargeDensity(_) => {
            excitation.distribution == ExcitationDistribution::IntensityPerVolume
                && signal.shape() == &SignalShape::Scalar
                && excitation.direction.is_none()
        }
        ExcitationAction::MagneticVolumeCurrentDensity(_) => {
            excitation.distribution == ExcitationDistribution::IntensityPerVolume
                && signal.shape() == &SignalShape::Vector3
                && excitation.direction.is_none()
                && !excitation.projection.preserve_resultant
                && !excitation.projection.preserve_moment
        }
        ExcitationAction::WallMotion(_) => {
            excitation.distribution == ExcitationDistribution::UniformKinematic
        }
        ExcitationAction::AcousticIncidentPressure(_) => {
            excitation.distribution == ExcitationDistribution::UniformKinematic
                && signal.shape() == &SignalShape::Scalar
                && excitation.direction.is_none()
                && excitation.kinematics == crate::ExcitationKinematics::FixedFrame
        }
        _ => true,
    };
    if !distribution_valid {
        return Err("distribution is incompatible with the excitation action".to_string());
    }
    if matches!(action, ExcitationAction::BodyForce(_))
        && (signal.shape().topology_component_count().is_some()
            != (excitation.distribution == ExcitationDistribution::TopologyField))
    {
        return Err("spatial body-density signals require topology-field distribution; uniform density requires intensity-per-volume".into());
    }
    validate_action_target(action, &excitation.target)?;
    let scalar_directional_action = matches!(
        action,
        ExcitationAction::Force(_)
            | ExcitationAction::Moment(_)
            | ExcitationAction::InletVelocity(_)
            | ExcitationAction::WallMotion(_)
    ) && matches!(signal.shape(), SignalShape::Scalar);
    if scalar_directional_action && excitation.direction.is_none() {
        return Err("a scalar directional action requires a direction".to_string());
    }
    if excitation.distribution == ExcitationDistribution::PreserveResultant
        && !excitation.projection.preserve_resultant
    {
        return Err("preserve-resultant distribution requires resultant conservation".to_string());
    }
    if matches!(action, ExcitationAction::Moment(_)) && !excitation.projection.preserve_moment {
        return Err("moment excitation requires moment conservation".to_string());
    }
    if matches!(
        action,
        ExcitationAction::HeatGeneration(_)
            | ExcitationAction::HeatFlux(_)
            | ExcitationAction::ElectricCurrent(_)
            | ExcitationAction::ElectricCharge(_)
            | ExcitationAction::ElectricChargeDensity(_)
            | ExcitationAction::AcousticIncidentPressure(_)
            | ExcitationAction::MassFlow(_)
    ) && excitation.projection.preserve_moment
    {
        return Err("a scalar-integral excitation cannot require moment conservation".to_string());
    }
    if excitation.kinematics == ExcitationKinematics::Follower
        && !matches!(
            action,
            ExcitationAction::Force(_)
                | ExcitationAction::Traction(_)
                | ExcitationAction::Pressure(_)
        )
    {
        return Err("follower kinematics is not valid for this action".to_string());
    }
    Ok(())
}

fn validate_action_target(
    action: &ExcitationAction,
    target: &ExcitationTarget,
) -> Result<(), String> {
    use crate::EntityDimension;

    let region_dimension = match target {
        ExcitationTarget::Region(region) => Some(region.entity_dimension),
        _ => None,
    };
    let surface_target = region_dimension == Some(EntityDimension::Face)
        || matches!(
            target,
            ExcitationTarget::Port { .. } | ExcitationTarget::Interface { .. }
        );
    let valid = match action {
        ExcitationAction::Traction(_)
        | ExcitationAction::Pressure(_)
        | ExcitationAction::HeatFlux(_)
        | ExcitationAction::AmbientTemperature(_)
        | ExcitationAction::WallMotion(_)
        | ExcitationAction::NormalVelocity(_)
        | ExcitationAction::InletVelocity(_)
        | ExcitationAction::MassFlow(_)
        | ExcitationAction::TerminalVoltage(_)
        | ExcitationAction::AcousticPressure(_)
        | ExcitationAction::AcousticIncidentPressure(_) => surface_target,
        ExcitationAction::BodyForce(_)
        | ExcitationAction::HeatGeneration(_)
        | ExcitationAction::ElectricChargeDensity(_)
        | ExcitationAction::MagneticVolumeCurrentDensity(_) => {
            matches!(target, ExcitationTarget::WholeModel)
                || region_dimension == Some(EntityDimension::Volume)
        }
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err("target dimension or kind is incompatible with the excitation action".to_string())
    }
}

fn validate_cases(
    cases: &[crate::ExcitationCase],
    actions: &[ExcitationAction],
    steps: &HashSet<&str>,
) -> Result<(), AnalysisValidationError> {
    let actions_by_id = actions
        .iter()
        .map(|action| (&action.excitation().excitation_id, action.excitation()))
        .collect::<HashMap<_, _>>();
    let mut case_ids = HashSet::new();
    let mut assigned_actions = HashSet::new();
    for case in cases {
        let id = case.case_id.0.as_str();
        let invalid = id.trim().is_empty()
            || !case_ids.insert(id)
            || case.action_ids.is_empty()
            || case.action_ids.iter().collect::<HashSet<_>>().len() != case.action_ids.len()
            || case.action_ids.iter().any(|action_id| {
                if actions_by_id.contains_key(action_id) {
                    assigned_actions.insert(action_id);
                    false
                } else {
                    true
                }
            })
            || match &case.activation {
                ExcitationCaseActivation::Steps(active) => {
                    active.is_empty() || active.iter().any(|step| !steps.contains(step.as_str()))
                }
                ExcitationCaseActivation::MutuallyExclusiveGroup { group_id } => {
                    group_id.trim().is_empty()
                }
                ExcitationCaseActivation::Always => false,
            };
        if invalid {
            return Err(AnalysisValidationError::InvalidExcitationCase {
                case_id: id.to_string(),
                detail: "case ids, action references, and activation must be valid and unique"
                    .to_string(),
            });
        }
    }
    if assigned_actions.len() != actions.len() {
        return Err(AnalysisValidationError::InvalidExcitationCase {
            case_id: String::new(),
            detail: "every excitation must belong to at least one declared case".to_string(),
        });
    }
    Ok(())
}

fn validate_initial_conditions(
    conditions: &[InitialCondition],
    signals: &HashMap<&SignalId, &PhysicalSignal>,
    steps: &HashSet<&str>,
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for condition in conditions {
        let id = condition.initial_condition_id();
        if id.trim().is_empty() || !ids.insert(id) {
            return invalid_initial(id, "initial-condition ids must be non-empty and unique");
        }
        let result = match condition {
            InitialCondition::Displacement(value) => validate_initial(
                value,
                PhysicalQuantityKind::Displacement,
                InitialValueShape::Vector3,
                signals,
                steps,
            ),
            InitialCondition::Velocity(value) | InitialCondition::FluidVelocity(value) => {
                validate_initial(
                    value,
                    PhysicalQuantityKind::Velocity,
                    InitialValueShape::Vector3,
                    signals,
                    steps,
                )
            }
            InitialCondition::AngularVelocity(value) => validate_initial(
                value,
                PhysicalQuantityKind::AngularVelocity,
                InitialValueShape::Vector3,
                signals,
                steps,
            ),
            InitialCondition::Temperature(value) => validate_initial(
                value,
                PhysicalQuantityKind::Temperature,
                InitialValueShape::Scalar,
                signals,
                steps,
            )
            .and_then(|()| match &value.value {
                InitialValue::Uniform(temperature) if temperature.canonical_value() < 0.0 => {
                    Err("initial temperature must be at or above zero kelvin".to_string())
                }
                _ => Ok(()),
            }),
            InitialCondition::ElectricPotential(value) => validate_initial(
                value,
                PhysicalQuantityKind::ElectricPotential,
                InitialValueShape::Scalar,
                signals,
                steps,
            ),
            InitialCondition::MagneticVectorPotential(value) => validate_initial(
                value,
                PhysicalQuantityKind::MagneticVectorPotential,
                InitialValueShape::Vector3,
                signals,
                steps,
            ),
            InitialCondition::AcousticPressure(value) | InitialCondition::FluidPressure(value) => {
                validate_initial(
                    value,
                    PhysicalQuantityKind::Pressure,
                    InitialValueShape::Scalar,
                    signals,
                    steps,
                )
            }
        };
        if let Err(detail) = result {
            return invalid_initial(id, detail);
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum InitialValueShape {
    Scalar,
    Vector3,
}

fn validate_initial<Q: FiniteQuantity>(
    condition: &InitialConditionValue<Q>,
    expected: PhysicalQuantityKind,
    expected_shape: InitialValueShape,
    signals: &HashMap<&SignalId, &PhysicalSignal>,
    steps: &HashSet<&str>,
) -> Result<(), String> {
    if let InitialConditionTarget::Region(target) = &condition.target {
        crate::validate_authored::validate_target(target)?;
    }
    crate::validate_authored::validate_coordinate_frame(&condition.frame)
        .map_err(|error| error.to_string())?;
    if !steps.contains(condition.applies_before_step_id.as_str()) {
        return Err("initial condition must name an existing following step".to_string());
    }
    match &condition.value {
        InitialValue::Uniform(value)
            if matches!(expected_shape, InitialValueShape::Scalar) && value.is_finite() =>
        {
            Ok(())
        }
        InitialValue::UniformVector3(values)
            if matches!(expected_shape, InitialValueShape::Vector3)
                && values.iter().all(FiniteQuantity::is_finite) =>
        {
            Ok(())
        }
        InitialValue::SignalSample {
            signal_id,
            abscissa,
            approach: _,
        } if abscissa.is_finite() && abscissa.to_bits() != (-0.0_f64).to_bits() => {
            match signals.get(signal_id) {
                Some(signal)
                    if signal.quantity() == expected
                        && signal.domain() != crate::SignalDomainKind::Frequency
                        && matches!(
                            (expected_shape, signal.shape()),
                            (InitialValueShape::Scalar, SignalShape::Scalar)
                                | (InitialValueShape::Vector3, SignalShape::Vector3)
                        ) =>
                {
                    Ok(())
                }
                Some(signal)
                    if signal.quantity() == expected
                        && signal.domain() == crate::SignalDomainKind::Frequency =>
                {
                    Err("frequency-domain signals cannot define an initial condition".to_string())
                }
                Some(signal) if signal.quantity() == expected => {
                    Err("initial-condition signal has the wrong value shape".to_string())
                }
                Some(_) => {
                    Err("initial-condition signal has the wrong physical quantity".to_string())
                }
                None => Err("initial-condition signal does not exist".to_string()),
            }
        }
        InitialValue::FieldArtifact(artifact)
            if !artifact.artifact_id.trim().is_empty()
                && !artifact.content_digest.trim().is_empty()
                && !artifact.topology_digest.trim().is_empty()
                && !artifact.field_id.trim().is_empty() =>
        {
            Ok(())
        }
        _ => Err("initial-condition value is not finite or complete".to_string()),
    }
}

fn invalid_initial<T>(
    initial_condition_id: &str,
    detail: impl Into<String>,
) -> Result<T, AnalysisValidationError> {
    Err(AnalysisValidationError::InvalidInitialCondition {
        initial_condition_id: initial_condition_id.to_string(),
        detail: detail.into(),
    })
}

fn validate_response_spectra(
    spectra: &[ResponseSpectrum],
    signals: &HashMap<&SignalId, &PhysicalSignal>,
) -> Result<(), AnalysisValidationError> {
    let mut ids = HashSet::new();
    for spectrum in spectra {
        let id = spectrum.spectrum_id.0.as_str();
        let result = (|| {
            if !ids.insert(id) {
                return Err("response-spectrum ids must be non-empty and unique".to_string());
            }
            spectrum.validate_intrinsic()?;
            if let crate::SpectrumConstruction::DerivedFromAccelerationHistory {
                source_signal_id,
                ..
            } = &spectrum.construction
            {
                match signals.get(source_signal_id) {
                    Some(signal)
                        if signal.quantity() == PhysicalQuantityKind::Acceleration
                            && signal.domain() == crate::SignalDomainKind::Time => {}
                    Some(_) => {
                        return Err(
                            "derived spectra require a time-domain acceleration signal".to_string()
                        )
                    }
                    None => {
                        return Err("derived spectrum source acceleration signal does not exist"
                            .to_string())
                    }
                }
            }
            Ok(())
        })();
        if let Err(detail) = result {
            return Err(AnalysisValidationError::InvalidResponseSpectrum {
                spectrum_id: id.to_string(),
                detail,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
