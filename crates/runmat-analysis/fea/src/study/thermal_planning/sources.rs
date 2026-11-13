use super::ThermalStudyPlanningError;
use crate::{
    excitation::{
        realize_step_excitations, sample_prescribed_heat_flux, sample_prescribed_temperature,
        sample_prescribed_temperature_difference, ExcitationSamplePoint, RealizedExcitationKind,
        RealizedExcitationTarget,
    },
    execution::{ThermalSolveSchedule, ThermalWorkload},
    ThermalProcedureControl,
};
use runmat_analysis_core::{
    AnalysisConstraint, BoundaryOperator, InitialCondition, SignalSampleApproach,
    ThermalConstraintKind,
};

/// Use the numerical pipeline's shared signal realization at exactly its bounded solve samples.
/// This catches unavailable artifacts, unsupported signal representations and complex sources
/// before a coordinator dispatches work, without assembling a mesh operator.
pub(super) fn validate(
    workload: &ThermalWorkload,
    schedule: &ThermalSolveSchedule,
    mixed_context: bool,
) -> Result<(), ThermalStudyPlanningError> {
    let model = &workload.model;
    if workload.backend != crate::ComputeBackend::Cpu
        && model
            .boundaries
            .iter()
            .any(|boundary| matches!(boundary.operator, BoundaryOperator::ThermalRadiation { .. }))
    {
        return Err(invalid(
            "nonlinear thermal radiation requires the admitted CPU backend",
        ));
    }
    let step = model
        .sequence
        .steps
        .iter()
        .find(|step| step.step_id == workload.step_id)
        .ok_or_else(|| invalid("selected thermal step is missing"))?;
    if model.thermal.is_none()
        || (!mixed_context && (model.structural.is_some() || model.thermo_mechanical.is_some()))
        || model.electro_thermal.is_some()
        || model.electromagnetic.is_some()
        || model.cfd.is_some()
        || model.interfaces.iter().any(|interface| {
            !matches!(
                interface.kind,
                runmat_analysis_core::AnalysisInterfaceKind::ConjugateHeatTransfer(_)
            )
        })
        || !model.connectors.is_empty()
        || !model.response_spectra.is_empty()
        || !model.random_excitations.is_empty()
        || model
            .initial_conditions
            .iter()
            .any(|condition| !matches!(condition, InitialCondition::Temperature(_)))
    {
        return Err(invalid(
            "study contains unsupported coupled or nonthermal physics",
        ));
    }
    for (index, &value) in schedule.points_s().iter().enumerate() {
        let sample = match workload.procedure {
            ThermalProcedureControl::Steady => ExcitationSamplePoint::Static,
            ThermalProcedureControl::BackwardEuler { .. } => ExcitationSamplePoint::TimeSeconds {
                value,
                approach: if index == 0 {
                    SignalSampleApproach::At
                } else {
                    schedule.approach(index)
                },
            },
        };
        let realized = realize_step_excitations(model, step, sample).map_err(invalid)?;
        for action in realized.iter() {
            if let RealizedExcitationKind::HeatGenerationField(ref field) = action.kind {
                if !field.is_real() {
                    return Err(invalid("thermal element scalar density must be real"));
                }
                continue;
            }
            if let RealizedExcitationKind::AmbientTemperature(value) = action.kind {
                // The ambient of a convection or radiation boundary on the same faces.
                let RealizedExcitationTarget::Region(target) = &action.target else {
                    return Err(invalid("ambient temperature needs a face region"));
                };
                if !model.boundaries.iter().any(|boundary| {
                    boundary.target.region_id == target.region_id
                        && matches!(
                            boundary.operator,
                            BoundaryOperator::ThermalConvection { .. }
                                | BoundaryOperator::ThermalRadiation { .. }
                        )
                }) {
                    return Err(invalid(
                        "ambient temperature needs a thermal.convection or thermal.radiation boundary on the same faces",
                    ));
                }
                real(value.real, value.imaginary)?;
                continue;
            }
            let value = match action.kind {
                RealizedExcitationKind::HeatGeneration(value)
                | RealizedExcitationKind::HeatFlux(value)
                | RealizedExcitationKind::TemperatureField(value) => value,
                _ => {
                    return Err(invalid(format!(
                        "excitation {:?} is not consumed by the thermal operator",
                        action.excitation_id
                    )))
                }
            };
            real(value.real, value.imaginary)?;
        }
        for constraint in &model.constraints {
            let AnalysisConstraint::Thermal(constraint) = constraint else {
                if mixed_context && matches!(constraint, AnalysisConstraint::Structural(_)) {
                    continue;
                }
                return Err(invalid("nonthermal constraint in standalone thermal study"));
            };
            let value = match &constraint.kind {
                ThermalConstraintKind::Prescribed { value, .. } => {
                    sample_prescribed_temperature(value, &model.signals, sample).map_err(invalid)?
                }
                ThermalConstraintKind::AffineJump {
                    temperature_jump, ..
                } => sample_prescribed_temperature_difference(
                    temperature_jump,
                    &model.signals,
                    sample,
                )
                .map_err(invalid)?,
                ThermalConstraintKind::Tie(_) | ThermalConstraintKind::Periodic(_) => continue,
            };
            real(value.real, value.imaginary)?;
        }
        for boundary in &model.boundaries {
            let value = match &boundary.operator {
                BoundaryOperator::ThermalHeatFlux { value } => {
                    sample_prescribed_heat_flux(value, &model.signals, sample).map_err(invalid)?
                }
                BoundaryOperator::ThermalConvection {
                    ambient_temperature,
                    ..
                } => sample_prescribed_temperature(ambient_temperature, &model.signals, sample)
                    .map_err(invalid)?,
                BoundaryOperator::ThermalRadiation {
                    ambient_temperature,
                    ..
                } => sample_prescribed_temperature(ambient_temperature, &model.signals, sample)
                    .map_err(invalid)?,
                BoundaryOperator::StructuralPressure { .. } if mixed_context => continue,
                _ => return Err(invalid("nonthermal boundary in standalone thermal study")),
            };
            real(value.real, value.imaginary)?;
        }
    }
    Ok(())
}

fn real(real: f64, imaginary: f64) -> Result<(), ThermalStudyPlanningError> {
    if !real.is_finite() || imaginary != 0.0 {
        return Err(invalid("thermal signals must realize finite real values"));
    }
    Ok(())
}

fn invalid(error: impl std::fmt::Display) -> ThermalStudyPlanningError {
    ThermalStudyPlanningError::Source(error.to_string())
}
