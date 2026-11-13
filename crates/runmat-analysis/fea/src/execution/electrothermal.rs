//! Genuine one-way steady electrical loss -> thermal solve on one canonical mesh.
//! This numerical composition is not scheduler admission, feedback coupling or release evidence.
use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkloadIdentity,
    ElectricalStaticSolverResult, ElectricalStaticWorkload, SolverResultIdentity, StepIdentity,
    ThermalWorkload,
};
use crate::{ComputeBackend, FeaThermalRunResult, ThermalProcedureControl};
use runmat_analysis_core::{
    AnalysisExcitation, CoordinateFrame, ExcitationAction, ExcitationCase,
    ExcitationCaseActivation, ExcitationCaseId, ExcitationDistribution, ExcitationId,
    ExcitationKinematics, ProjectionRequirement, SignalId,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;

mod coupled;
pub use coupled::{
    bind_element_heat_source, conductivity_at, electrical_part, element_temperatures,
    execute_coupled_electrothermal, thermal_part, CoupledElectrothermalCompletion,
    ElectrothermalCouplingControls, ElectrothermalElectricalSample, ElectrothermalSweep,
    ElementHeatSourceSample,
};
mod operating_point;
pub use operating_point::{
    conduction_operating_temperatures, electrothermal_conduction_study, is_electrothermal_study,
    solve_electrothermal_heat_study, ElectrothermalOperatingPoint,
};
mod transfer;
pub use transfer::RetainedJouleHeatTransfer;
mod source;
pub use source::RetainedJouleHeatSource;
#[cfg(test)]
mod tests;

const SIGNAL_ID: &str = "electrothermal.retained_joule_density";
const ACTION_ID: &str = "electrothermal.retained_joule_source";
const CASE_ID: &str = "electrothermal.retained_joule_case";

#[derive(Clone, Debug, PartialEq)]
pub struct SteadyElectrothermalCompletion {
    pub transfer: RetainedJouleHeatTransfer,
    pub thermal_workload: ThermalWorkload,
    pub thermal: FeaThermalRunResult,
}

/// Bind exact retained source values before thermal identity, preparation and numerical solve.
/// The expected electrical result identity must come from the authenticated producer/root.
pub fn bind_retained_joule_heat(
    electrical: &ElectricalStaticWorkload,
    result: &ElectricalStaticSolverResult,
    expected_result: SolverResultIdentity,
    thermal: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
) -> AnalysisExecutionResult<(RetainedJouleHeatTransfer, ThermalWorkload)> {
    electrical.validate()?;
    thermal.validate()?;
    if thermal.backend != ComputeBackend::Cpu
        || thermal.procedure != ThermalProcedureControl::Steady
        || !thermal.initial_fields.is_empty()
        || !thermal.model.excitations.is_empty()
        || thermal.model.boundaries.iter().any(|boundary| {
            matches!(
                boundary.operator,
                runmat_analysis_core::BoundaryOperator::ThermalHeatFlux { .. }
            )
        })
        || electrical
            .model
            .materials
            .iter()
            .filter_map(|m| m.electrical.as_ref())
            .any(|m| {
                m.resistive_heating_coefficient.canonical_value() != 0.0
                    || !m.conductivity_frequency_response.is_empty()
            })
    {
        return Err(invalid("one-way steady electrothermal composition requires CPU constant-coefficient conduction and a source-free steady thermal target"));
    }
    let transfer =
        RetainedJouleHeatTransfer::from_result(electrical, result, mesh, expected_result)?;
    if thermal
        .model
        .signals
        .iter()
        .any(|s| s.signal_id().0 == SIGNAL_ID)
        || thermal
            .model
            .excitation_cases
            .iter()
            .any(|c| c.case_id.0 == CASE_ID)
    {
        return Err(invalid(
            "retained Joule source identifier collides with authored target",
        ));
    }
    let mut bound = thermal.clone();
    bound
        .model
        .signals
        .push(transfer.signal(SignalId(SIGNAL_ID.into()))?);
    bound
        .model
        .excitations
        .push(ExcitationAction::HeatGeneration(AnalysisExcitation {
            excitation_id: ExcitationId(ACTION_ID.into()),
            target: runmat_analysis_core::ExcitationTarget::WholeModel,
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::TopologyField,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId(SIGNAL_ID.into()),
            active_step_ids: vec![bound.step_id.0.clone()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-10,
            },
        }));
    bound.model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId(CASE_ID.into()),
        action_ids: vec![ExcitationId(ACTION_ID.into())],
        activation: ExcitationCaseActivation::Steps(vec![bound.step_id.0.clone()]),
    });
    bound.validate()?;
    Ok((transfer, bound))
}

/// Only the thermal numerical kernel executes. Retained electrical fields are never solved again.
pub fn execute_steady_electrothermal_from_retained(
    electrical: &ElectricalStaticWorkload,
    result: &ElectricalStaticSolverResult,
    expected_result: SolverResultIdentity,
    thermal: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
) -> AnalysisExecutionResult<SteadyElectrothermalCompletion> {
    let (transfer, bound) =
        bind_retained_joule_heat(electrical, result, expected_result, thermal, mesh)?;
    let thermal = bound.execute(mesh).map_err(invalid)?;
    thermal.validate_complete().map_err(invalid)?;
    if thermal.source_workload != bound.identity()?
        || thermal.numerical_intent != bound.numerical_intent_identity()?
        || thermal.energy_balances.len() != 1
    {
        return Err(invalid(
            "thermal result differs from retained-source-bound steady intent",
        ));
    }
    let balance = thermal.energy_balances[0];
    balance.validate().map_err(invalid)?;
    let scale = transfer
        .power_w()
        .abs()
        .max(balance.applied_power_w.canonical_value().abs());
    let relative = if scale == 0.0 {
        0.0
    } else {
        (transfer.power_w() - balance.applied_power_w.canonical_value()).abs() / scale
    };
    if (scale == 0.0 && balance.applied_power_w.canonical_value() != transfer.power_w())
        || relative > bound.residual_target.max(1e-10) * 10.0
        || balance.relative_residual > bound.residual_target.max(1e-10) * 10.0
    {
        return Err(invalid(
            "retained Joule power, thermal volume source or thermal outflow fails conservation",
        ));
    }
    Ok(SteadyElectrothermalCompletion {
        transfer,
        thermal_workload: bound,
        thermal,
    })
}

fn invalid(message: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.to_string())
}
