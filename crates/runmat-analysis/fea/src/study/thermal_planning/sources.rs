use super::ThermalStudyPlanningError;
use crate::{
    excitation::{
        realize_step_excitations, sample_prescribed_heat_flux, sample_prescribed_temperature,
        ExcitationSamplePoint, RealizedExcitationKind,
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
) -> Result<(), ThermalStudyPlanningError> {
    let model = &workload.model;
    let step = &model.sequence.steps[0];
    if model.thermal.is_none()
        || model.structural.is_some()
        || model.thermo_mechanical.is_some()
        || model.electro_thermal.is_some()
        || model.electromagnetic.is_some()
        || model.cfd.is_some()
        || !model.interfaces.is_empty()
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
                return Err(invalid("nonthermal constraint in standalone thermal study"));
            };
            let ThermalConstraintKind::Prescribed { value, .. } = &constraint.kind else {
                return Err(invalid(
                    "thermal operator does not support this constraint kind",
                ));
            };
            let value =
                sample_prescribed_temperature(value, &model.signals, sample).map_err(invalid)?;
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
