//! Split an authored electrothermal analysis into its conduction and heat-transfer sides and
//! solve the two-way operating point that both published cases share.
use runmat_analysis_core::ObservationQuantity;
use runmat_meshing_core::SolverMeshArtifact;

use super::coupled::{
    electrical_part, element_temperatures, execute_coupled_electrothermal, thermal_part,
    CoupledElectrothermalCompletion, ElectrothermalCouplingControls, ElectrothermalSweep,
};
use crate::execution::{AnalysisExecutionError, AnalysisExecutionResult};
use crate::study::{plan_thermal_study, CompiledAnalysisStudy, ThermalStudyPlanningPolicy};
use crate::topology::AnalysisProjection;

/// The converged two-way electrothermal state.
#[derive(Clone, Debug)]
pub struct ElectrothermalOperatingPoint {
    pub sweeps: Vec<ElectrothermalSweep>,
    pub converged: bool,
    /// Element mean temperatures at the last time point, canonical element order.
    pub element_temperature_k: Vec<f64>,
    /// Last time point of a transient analysis; `None` when steady.
    pub operating_time_s: Option<f64>,
    /// Joule power and terminal power (sum of terminal potential times current) at that point.
    pub joule_power_w: f64,
    pub terminal_power_w: f64,
}

/// The heat-transfer side of an electrothermal study with the converged Joule source bound,
/// and the operating point that produced it.
pub fn solve_electrothermal_heat_study(
    compiled: &CompiledAnalysisStudy,
    mesh: &SolverMeshArtifact,
    policy: ThermalStudyPlanningPolicy,
) -> AnalysisExecutionResult<(CompiledAnalysisStudy, ElectrothermalOperatingPoint)> {
    let step = single_step(compiled)?;
    let electrical = electrical_part(&compiled.model, &step);
    let mut thermal = compiled.clone();
    thermal.model = thermal_part(&compiled.model, &step);
    keep_observations(&mut thermal.model, &step, is_thermal_quantity);
    runmat_analysis_core::retain_authored_outputs(
        &mut thermal.outputs,
        &mut thermal.assessments,
        |_, quantity| is_thermal_quantity(quantity),
    );
    let template = plan_thermal_study(&thermal, policy)
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let completion = execute_coupled_electrothermal(
        &electrical,
        &template,
        mesh,
        template.residual_target,
        ElectrothermalCouplingControls::default(),
    )?;
    let point = operating_point(&completion, mesh)?;
    thermal.model = completion.thermal_workload.model;
    Ok((thermal, point))
}

/// The conduction side's operating temperatures, taken from the heat-transfer case's published
/// thermal state (its last time point), so the conduction case reports the same operating point
/// the heat case publishes instead of solving the coupled loop again.
pub fn conduction_operating_temperatures(
    state: &crate::execution::CanonicalThermalStatePayload,
    mesh: &SolverMeshArtifact,
    interfaces: &[runmat_analysis_core::AnalysisInterface],
    transient: bool,
) -> AnalysisExecutionResult<(Vec<f64>, Option<f64>)> {
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let values = state.temperature.as_host_f64().ok_or_else(|| {
        AnalysisExecutionError::Invalid("the heat case's temperature must be host-backed".into())
    })?;
    if values.len() != state.ordered_node_ids.len() {
        return Err(AnalysisExecutionError::Invalid(
            "the heat case's temperature does not match its node order".into(),
        ));
    }
    let by_id: std::collections::BTreeMap<u64, f64> = state
        .ordered_node_ids
        .iter()
        .copied()
        .zip(values.iter().copied())
        .collect();
    let nodal = projection
        .nodes
        .iter()
        .map(|node| {
            by_id.get(&node.node_id).copied().ok_or_else(|| {
                AnalysisExecutionError::Invalid(
                    "the heat case's temperature misses a conduction mesh node".into(),
                )
            })
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let second = state.interface_temperature_by_node()?;
    let element_temperature_k = if second.is_empty() {
        element_temperatures(&projection, &nodal)?
    } else {
        side_element_temperatures(&projection, interfaces, &nodal, &second)?
    };
    Ok((
        element_temperature_k,
        transient.then_some(state.sample_time_s.canonical_value()),
    ))
}

/// Element means where second-side elements of conductance interfaces read that side's
/// temperature at the interface nodes.
pub(crate) fn side_element_temperatures(
    projection: &AnalysisProjection,
    interfaces: &[runmat_analysis_core::AnalysisInterface],
    nodal: &[f64],
    second: &[(u64, f64)],
) -> AnalysisExecutionResult<Vec<f64>> {
    let invalid = |error: String| AnalysisExecutionError::Invalid(error);
    let (plans, dofs) = crate::physics::thermal_interface::solid_conduction(projection, interfaces)
        .map_err(invalid)?;
    if dofs.second().len() != second.len()
        || dofs
            .second()
            .iter()
            .zip(second)
            .any(|(&(node, _), &(id, _))| projection.nodes[node].node_id != id)
    {
        return Err(invalid(
            "thermal interface nodes differ from the declared interfaces".into(),
        ));
    }
    let values = dofs
        .join(
            nodal,
            &dofs
                .second()
                .iter()
                .zip(second)
                .map(|(&(node, _), &(_, value))| (node, value))
                .collect::<Vec<_>>(),
        )
        .map_err(invalid)?;
    element_temperatures(&dofs.split_projection(projection.clone(), &plans), &values)
}

/// The conduction side of an electrothermal study, with only its electrical outputs.
pub fn electrothermal_conduction_study(
    compiled: &CompiledAnalysisStudy,
) -> AnalysisExecutionResult<CompiledAnalysisStudy> {
    let step = single_step(compiled)?;
    let mut conduction = compiled.clone();
    conduction.model = electrical_part(&compiled.model, &step);
    keep_observations(&mut conduction.model, &step, is_electrical_quantity);
    runmat_analysis_core::retain_authored_outputs(
        &mut conduction.outputs,
        &mut conduction.assessments,
        |_, quantity| is_electrical_quantity(quantity),
    );
    Ok(conduction)
}

/// One fresh electrothermal analysis.
pub fn is_electrothermal_study(compiled: &CompiledAnalysisStudy) -> bool {
    matches!(compiled.model.sequence.steps.as_slice(), [step]
        if step.family == runmat_analysis_core::PhysicsFamily::Coupled(
            runmat_analysis_core::CoupledPhysicsFamily::Electrothermal))
}

fn single_step(
    compiled: &CompiledAnalysisStudy,
) -> AnalysisExecutionResult<runmat_analysis_core::ProcedureStep> {
    match compiled.model.sequence.steps.as_slice() {
        [step] => Ok(step.clone()),
        _ => Err(AnalysisExecutionError::Invalid(
            "electrothermal execution requires one analysis".into(),
        )),
    }
}

fn operating_point(
    completion: &CoupledElectrothermalCompletion,
    mesh: &SolverMeshArtifact,
) -> AnalysisExecutionResult<ElectrothermalOperatingPoint> {
    let projection = AnalysisProjection::derive(mesh)
        .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
    let last = completion
        .thermal
        .temperature_snapshots
        .len()
        .checked_sub(1)
        .ok_or_else(|| {
            AnalysisExecutionError::Invalid("electrothermal thermal history is empty".into())
        })?;
    let element_temperature_k = super::coupled::element_temperatures_of_snapshot(
        &projection,
        &completion.thermal_workload.model.interfaces,
        &completion.thermal,
        last,
    )?;
    let sample = completion.electrical.last().ok_or_else(|| {
        AnalysisExecutionError::Invalid("electrothermal conduction history is empty".into())
    })?;
    Ok(ElectrothermalOperatingPoint {
        sweeps: completion.sweeps.clone(),
        converged: completion.converged,
        element_temperature_k,
        operating_time_s: (completion.electrical.len() > 1).then_some(sample.time_s),
        joule_power_w: CoupledElectrothermalCompletion::joule_power_w(sample),
        terminal_power_w: CoupledElectrothermalCompletion::terminal_power_w(sample),
    })
}

/// Each side publishes the authored observations of the quantities it solves.
fn keep_observations(
    part: &mut runmat_analysis_core::AnalysisModel,
    step: &runmat_analysis_core::ProcedureStep,
    keep: fn(ObservationQuantity) -> bool,
) {
    let observations = step
        .observations
        .iter()
        .filter(|request| request.spec().is_some_and(|spec| keep(spec.quantity)))
        .cloned()
        .collect();
    if let Some(side) = part
        .sequence
        .steps
        .iter_mut()
        .find(|side| side.step_id == step.step_id)
    {
        side.observations = observations;
    }
}

fn is_thermal_quantity(quantity: ObservationQuantity) -> bool {
    use ObservationQuantity as Q;
    matches!(
        quantity,
        Q::Temperature
            | Q::TemperatureGradient
            | Q::HeatFlux
            | Q::HeatRate
            | Q::ThermalEnergyStorageRateDensity
    )
}

fn is_electrical_quantity(quantity: ObservationQuantity) -> bool {
    use ObservationQuantity as Q;
    matches!(
        quantity,
        Q::ElectricPotential
            | Q::ElectricField
            | Q::ElectricCurrentDensity
            | Q::JoulePowerDensity
            | Q::ElectricCurrent
    )
}
