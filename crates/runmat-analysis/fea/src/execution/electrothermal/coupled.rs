//! Two-way electrothermal (Joule heating) solve on one canonical mesh.
//!
//! The authored electrothermal step is split into its two canonical single-physics parts,
//! each solved by its own kernel (no second implementation of either physics):
//!   * steady conduction, div(sigma(T) grad V) = 0, with sigma(T) = sigma_ref / (1 + alpha
//!     (T - T_ref)) evaluated at each element's mean temperature (resistivity linear in T);
//!   * heat transfer (steady or backward Euler) with the Joule density sigma(T)|grad V|^2
//!     as an element heat source.
//! The two are iterated to a fixed point. In a transient step the iteration acts on the
//! whole temperature history (waveform relaxation): every sweep solves conduction at every
//! time point of the thermal schedule, then the complete thermal transient with the
//! tabulated Joule source, until the temperature history stops changing. With alpha = 0
//! the first sweep is exact (one-way coupling) and the second confirms it.
use super::{AnalysisExecutionError, AnalysisExecutionResult, ThermalWorkload};
use crate::physics::electrical::{CanonicalElectricalResponse, ElectricalConstitutiveResponse};
use crate::topology::AnalysisProjection;
use crate::{FeaThermalRunResult, ThermalProcedureControl};
use runmat_analysis_core::*;
use runmat_meshing_core::SolverMeshArtifact;

const JOULE_SOURCE: &str = "electrothermal.joule";

/// Fixed-point controls of the coupled iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectrothermalCouplingControls {
    /// Converged when the largest change of any element temperature between two sweeps is
    /// at most this fraction of the largest temperature rise (and of 1 K).
    pub relative_temperature_tolerance: f64,
    pub maximum_sweeps: u32,
}

impl Default for ElectrothermalCouplingControls {
    fn default() -> Self {
        Self {
            relative_temperature_tolerance: 1e-6,
            maximum_sweeps: 50,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ElectrothermalSweep {
    pub maximum_temperature_change_k: f64,
    pub maximum_temperature_rise_k: f64,
    /// Joule power at the last time point of the sweep.
    pub joule_power_w: f64,
}

/// Electrical state at one time point of the converged solution.
#[derive(Debug, Clone, PartialEq)]
pub struct ElectrothermalElectricalSample {
    pub time_s: f64,
    pub element_temperature_k: Vec<f64>,
    pub element_conductivity_s_per_m: Vec<f64>,
    pub response: CanonicalElectricalResponse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoupledElectrothermalCompletion {
    pub sweeps: Vec<ElectrothermalSweep>,
    pub converged: bool,
    /// Thermal workload of the final sweep, with its Joule source bound.
    pub thermal_workload: ThermalWorkload,
    pub thermal: FeaThermalRunResult,
    /// Conduction solutions that produced the final sweep's Joule source.
    pub electrical: Vec<ElectrothermalElectricalSample>,
}

impl CoupledElectrothermalCompletion {
    /// Joule power delivered by the terminals, sum of potential x injected current
    /// (equals the integral of sigma |grad V|^2 by the conduction energy identity).
    pub fn terminal_power_w(sample: &ElectrothermalElectricalSample) -> f64 {
        sample.response.volume_source_work.canonical_value()
            + sample.response.terminal_work.canonical_value()
    }
    pub fn joule_power_w(sample: &ElectrothermalElectricalSample) -> f64 {
        match &sample.response.constitutive {
            ElectricalConstitutiveResponse::SteadyConduction {
                total_joule_power_w,
                ..
            } => total_joule_power_w.canonical_value(),
            ElectricalConstitutiveResponse::Electrostatics { .. } => 0.0,
        }
    }
}

/// Split the authored coupled model into its steady-conduction part.
/// Thermal initial conditions, boundaries and sources stay with the heat-transfer part.
pub fn electrical_part(model: &AnalysisModel, step: &ProcedureStep) -> AnalysisModel {
    let mut part = model.clone();
    // The conduction part keeps the analysis name, so authored outputs of the coupled
    // analysis (current density, potential, loss) resolve against it.
    let id = step.step_id.0.clone();
    let mut electrical = ProcedureStep::electromagnetic_static(id.clone());
    electrical.accuracy = step.accuracy.clone();
    electrical.excitation_case_selection = step.excitation_case_selection.clone();
    part.sequence =
        AnalysisSequence::from_steps(model.sequence.sequence_id.clone(), vec![electrical]);
    part.electromagnetic = Some(ElectromagneticDomain::Static {
        equation: StaticElectricalEquation::SteadyConduction,
    });
    part.thermal = None;
    part.thermo_mechanical = None;
    part.electro_thermal = None;
    part.structural = None;
    part.cfd = None;
    part.boundaries.clear();
    part.initial_conditions.clear();
    part.constraints
        .retain(|c| matches!(c, AnalysisConstraint::Electrical(_)));
    part.excitations.retain(|action| {
        matches!(
            action,
            ExcitationAction::ElectricCurrent(_) | ExcitationAction::ElectricChargeDensity(_)
        )
    });
    retarget(&mut part, step.step_id.as_str(), &id);
    part
}

/// Split the authored coupled model into its heat-transfer part (without the Joule source).
pub fn thermal_part(model: &AnalysisModel, step: &ProcedureStep) -> AnalysisModel {
    let mut part = model.clone();
    let mut thermal = match step.procedure {
        AnalysisProcedure::TransientResponse => {
            ProcedureStep::thermal_transient(step.step_id.0.clone())
        }
        _ => ProcedureStep::thermal_steady(step.step_id.0.clone()),
    };
    thermal.independent_domain = step.independent_domain.clone();
    thermal.accuracy = step.accuracy.clone();
    thermal.state_inputs = step.state_inputs.clone();
    thermal.activations = step.activations.clone();
    thermal.excitation_case_selection = step.excitation_case_selection.clone();
    part.sequence = AnalysisSequence::from_steps(model.sequence.sequence_id.clone(), vec![thermal]);
    part.electromagnetic = None;
    part.electro_thermal = None;
    if part.thermal.is_none() {
        part.thermal = model.electro_thermal.as_ref().map(|domain| ThermalDomain {
            reference_temperature: domain.reference_temperature,
        });
    }
    part.constraints
        .retain(|c| matches!(c, AnalysisConstraint::Thermal(_)));
    part.excitations.retain(|action| {
        !matches!(
            action,
            ExcitationAction::ElectricCurrent(_)
                | ExcitationAction::ElectricChargeDensity(_)
                | ExcitationAction::MagneticVolumeCurrentDensity(_)
        )
    });
    prune_cases(&mut part);
    part
}

fn retarget(model: &mut AnalysisModel, from: &str, to: &str) {
    for action in &mut model.excitations {
        let excitation = action.excitation_mut();
        for step in &mut excitation.active_step_ids {
            if step == from {
                *step = to.to_string();
            }
        }
    }
    prune_cases(model);
    for case in &mut model.excitation_cases {
        if let ExcitationCaseActivation::Steps(steps) = &mut case.activation {
            for step in steps {
                if step == from {
                    *step = to.to_string();
                }
            }
        }
    }
}

/// Drop excitation cases left empty by the split, and their selection by the step.
fn prune_cases(model: &mut AnalysisModel) {
    let present: std::collections::BTreeSet<_> = model
        .excitations
        .iter()
        .map(|action| action.excitation().excitation_id.clone())
        .collect();
    for case in &mut model.excitation_cases {
        case.action_ids.retain(|id| present.contains(id));
    }
    model
        .excitation_cases
        .retain(|case| !case.action_ids.is_empty());
    let cases: std::collections::BTreeSet<_> = model
        .excitation_cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect();
    for step in &mut model.sequence.steps {
        if let ExcitationCaseSelection::Only(selected) = &mut step.excitation_case_selection {
            selected.retain(|case| cases.contains(case));
        }
    }
}

/// sigma(T) = sigma_ref / (1 + alpha (T - T_ref)); resistivity linear in temperature.
pub fn conductivity_at(
    law: &MaterialElectricalModel,
    temperature_k: f64,
) -> AnalysisExecutionResult<f64> {
    let factor = 1.0
        + law.resistive_heating_coefficient.canonical_value()
            * (temperature_k - law.reference_temperature_k.canonical_value());
    if !factor.is_finite() || factor <= 0.0 {
        return Err(invalid(format!(
            "temperature {temperature_k} K is outside the linear resistivity law (1 + alpha (T - T_ref) <= 0)"
        )));
    }
    Ok(law.conductivity_s_per_m.canonical_value() / factor)
}

/// Element mean temperatures of one thermal snapshot. Elements on the second side of a
/// finite-conductance interface use that side's temperature at the interface nodes.
pub fn element_temperatures_of_snapshot(
    projection: &AnalysisProjection,
    interfaces: &[runmat_analysis_core::AnalysisInterface],
    thermal: &crate::FeaThermalRunResult,
    index: usize,
) -> AnalysisExecutionResult<Vec<f64>> {
    let nodal = thermal
        .temperature_snapshots
        .get(index)
        .and_then(|snapshot| snapshot.as_host_f64())
        .ok_or_else(|| invalid("thermal snapshot is not host data"))?;
    if thermal.interface_node_ids.is_empty() {
        return element_temperatures(projection, nodal);
    }
    let second = thermal
        .interface_temperature_snapshots
        .get(index)
        .and_then(|snapshot| snapshot.as_host_f64())
        .ok_or_else(|| invalid("thermal interface snapshot is not host data"))?;
    let second = thermal
        .interface_node_ids
        .iter()
        .copied()
        .zip(second.iter().copied())
        .collect::<Vec<_>>();
    super::operating_point::side_element_temperatures(projection, interfaces, nodal, &second)
}

/// Element mean temperatures from nodal values (volume average with the element's shape
/// functions, so Tet10 midside temperatures are honoured).
pub fn element_temperatures(
    projection: &AnalysisProjection,
    nodal_k: &[f64],
) -> AnalysisExecutionResult<Vec<f64>> {
    if nodal_k.len() != projection.nodes.len() {
        return Err(invalid("temperature field does not match the mesh"));
    }
    projection
        .elements
        .iter()
        .map(|element| {
            let mut integral = 0.0;
            let mut volume = 0.0;
            projection
                .visit_element_quadrature(element, |point| {
                    let value: f64 = element
                        .node_indices
                        .iter()
                        .zip(point.shape_values())
                        .map(|(&node, &shape)| nodal_k[node] * shape)
                        .sum();
                    integral += value * point.volume_weight_m3;
                    volume += point.volume_weight_m3;
                })
                .map_err(invalid)?;
            Ok(integral / volume)
        })
        .collect()
}

/// One time point of a per-element heat generation density, in canonical element order.
pub type ElementHeatSourceSample = (
    runmat_types::physical::Time,
    runmat_types::physical::Quantity<runmat_types::physical::kind::HeatGenerationDensity, Vec<f64>>,
);

fn element_heat_signal(
    signal_id: &str,
    mesh_digest: runmat_execution::Digest,
    samples: &[ElementHeatSourceSample],
    transient: bool,
) -> PhysicalSignal {
    let tensor = |values: &[f64]| SignalValue::Tensor {
        shape: vec![values.len(), 1],
        values: values
            .iter()
            .copied()
            .map(HeatGenerationDensity::new)
            .collect(),
    };
    let representation = if transient && samples.len() > 1 {
        SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: samples
                .iter()
                .map(|(time, values)| {
                    SignalKnot::continuous(
                        time.canonical_value(),
                        SignalOrdinate::Real(tensor(values.canonical_values())),
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::HoldEndpoint,
        })
    } else {
        SignalRepresentation::Constant(tensor(samples[0].1.canonical_values()))
    };
    PhysicalSignal::HeatGenerationDensity(Signal {
        signal_id: SignalId(signal_id.into()),
        shape: SignalShape::TopologyField {
            topology_id: mesh_digest.to_string(),
            location: FieldLocation::Element,
            component_count: 1,
        },
        frame: Some(CoordinateFrame::Global),
        representation,
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: samples.len(),
        },
        provenance: SignalProvenance::Authored,
    })
}

/// Bind an element heat-generation field (W/m^3 per canonical volume element, constant or
/// tabulated over time) as a whole-model heat source of `step_id`. This is how a solved
/// electromagnetic loss becomes the heat source of a thermal analysis on the same mesh.
pub fn bind_element_heat_source(
    model: &mut AnalysisModel,
    step_id: &str,
    source_id: &str,
    mesh_digest: runmat_execution::Digest,
    samples: &[ElementHeatSourceSample],
    transient: bool,
) -> AnalysisExecutionResult<()> {
    if samples.is_empty()
        || samples.iter().any(|(time, values)| {
            !time.canonical_value().is_finite()
                || values
                    .canonical_values()
                    .iter()
                    .any(|v| !v.is_finite() || *v < 0.0)
        })
    {
        return Err(invalid("heat source values must be finite and nonnegative"));
    }
    let signal_id = format!("{source_id}.density");
    let action_id = format!("{source_id}.source");
    let case_id = format!("{source_id}.case");
    if model.signals.iter().any(|s| s.signal_id().0 == signal_id)
        || model
            .excitations
            .iter()
            .any(|a| a.excitation().excitation_id.0 == action_id)
    {
        return Err(invalid(
            "the heat source identifier collides with an authored one",
        ));
    }
    model.signals.push(element_heat_signal(
        &signal_id,
        mesh_digest,
        samples,
        transient,
    ));
    model
        .excitations
        .push(ExcitationAction::HeatGeneration(AnalysisExcitation {
            excitation_id: ExcitationId(action_id.clone()),
            target: ExcitationTarget::WholeModel,
            frame: CoordinateFrame::Global,
            direction: None,
            distribution: ExcitationDistribution::TopologyField,
            kinematics: ExcitationKinematics::FixedFrame,
            signal_id: SignalId(signal_id),
            active_step_ids: vec![step_id.to_string()],
            combination_group: None,
            projection: ProjectionRequirement {
                preserve_resultant: true,
                preserve_moment: false,
                relative_tolerance: 1e-10,
            },
        }));
    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId(case_id),
        action_ids: vec![ExcitationId(action_id)],
        activation: ExcitationCaseActivation::Steps(vec![step_id.to_string()]),
    });
    Ok(())
}

fn bind_joule(
    template: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    samples: &[ElementHeatSourceSample],
) -> AnalysisExecutionResult<ThermalWorkload> {
    let transient = matches!(
        template.procedure,
        ThermalProcedureControl::BackwardEuler { .. }
    );
    let mut bound = template.clone();
    let step_id = bound.step_id.0.clone();
    bind_element_heat_source(
        &mut bound.model,
        &step_id,
        JOULE_SOURCE,
        runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
        samples,
        transient,
    )?;
    bound.validate()?;
    Ok(bound)
}

/// Iterate conduction and heat transfer to the two-way fixed point.
pub fn execute_coupled_electrothermal(
    electrical_model: &AnalysisModel,
    thermal_template: &ThermalWorkload,
    mesh: &SolverMeshArtifact,
    electrical_residual_target: f64,
    controls: ElectrothermalCouplingControls,
) -> AnalysisExecutionResult<CoupledElectrothermalCompletion> {
    if !(controls.relative_temperature_tolerance > 0.0
        && controls.relative_temperature_tolerance < 1.0)
        || controls.maximum_sweeps == 0
    {
        return Err(invalid("invalid electrothermal coupling controls"));
    }
    let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
    let properties =
        crate::properties::compile_solid_property_map(electrical_model, mesh).map_err(invalid)?;
    let laws = projection
        .elements
        .iter()
        .map(|element| {
            properties
                .solid_material(element.stable_identity)
                .and_then(|m| m.material.electrical.clone())
                .ok_or_else(|| invalid("every element needs an electrical material law"))
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let conduction_step = electrical_model.sequence.steps[0].step_id.0.clone();
    let reference_k = thermal_template
        .model
        .thermal
        .as_ref()
        .map(|domain| domain.reference_temperature.canonical_value())
        .ok_or_else(|| invalid("electrothermal analysis needs a thermal reference temperature"))?;
    let schedule = thermal_template.solve_schedule().map_err(invalid)?;
    let times: Vec<f64> = schedule.points_s().to_vec();
    let transient = matches!(
        thermal_template.procedure,
        ThermalProcedureControl::BackwardEuler { .. }
    );
    let mut temperatures: Vec<Vec<f64>> =
        vec![vec![reference_k; projection.elements.len()]; times.len()];
    let mut sweeps = Vec::new();
    loop {
        let mut electrical = Vec::with_capacity(times.len());
        let mut samples = Vec::with_capacity(times.len());
        for (index, (time, element_temperature)) in times.iter().zip(&temperatures).enumerate() {
            let started = runmat_time::Instant::now();
            let conductivity = laws
                .iter()
                .zip(element_temperature)
                .map(|(law, temperature)| conductivity_at(law, *temperature))
                .collect::<AnalysisExecutionResult<Vec<_>>>()?;
            let sample = if transient {
                crate::excitation::ExcitationSamplePoint::time(*time)
            } else {
                crate::excitation::ExcitationSamplePoint::Static
            };
            let response = crate::physics::electrical::solve_conduction_with_conductivity(
                electrical_model,
                mesh,
                &conduction_step,
                electrical_residual_target,
                sample,
                &conductivity,
            )
            .map_err(invalid)?;
            let ElectricalConstitutiveResponse::SteadyConduction {
                joule_density_w_per_m3,
                ..
            } = &response.constitutive
            else {
                return Err(invalid("conduction response expected"));
            };
            crate::progress::report_step(
                "fea.run_electrothermal_conduction",
                index + 1,
                times.len(),
                *time,
                started,
                format_args!(" (coupling sweep {})", sweeps.len() + 1),
            );
            samples.push((
                runmat_types::physical::Time::new(*time),
                joule_density_w_per_m3.clone(),
            ));
            electrical.push(ElectrothermalElectricalSample {
                time_s: *time,
                element_temperature_k: element_temperature.clone(),
                element_conductivity_s_per_m: conductivity,
                response,
            });
        }
        let bound = bind_joule(thermal_template, mesh, &samples)?;
        let thermal = bound.execute(mesh).map_err(invalid)?;
        thermal.validate_complete().map_err(invalid)?;
        if thermal.temperature_snapshots.len() != times.len() {
            return Err(invalid("thermal history differs from the coupled schedule"));
        }
        let mut next = Vec::with_capacity(times.len());
        for index in 0..thermal.temperature_snapshots.len() {
            next.push(element_temperatures_of_snapshot(
                &projection,
                &thermal_template.model.interfaces,
                &thermal,
                index,
            )?);
        }
        let mut change = 0.0_f64;
        let mut rise = 0.0_f64;
        for (old, new) in temperatures.iter().zip(&next) {
            for (a, b) in old.iter().zip(new) {
                change = change.max((a - b).abs());
                rise = rise.max((b - reference_k).abs());
            }
        }
        let last = electrical.last().expect("schedule has a time point");
        sweeps.push(ElectrothermalSweep {
            maximum_temperature_change_k: change,
            maximum_temperature_rise_k: rise,
            joule_power_w: CoupledElectrothermalCompletion::joule_power_w(last),
        });
        let converged = change <= controls.relative_temperature_tolerance * rise.max(1.0);
        if converged || sweeps.len() as u32 >= controls.maximum_sweeps {
            if !converged {
                return Err(AnalysisExecutionError::Invalid(format!(
                    "electrothermal coupling did not converge in {} sweeps (last temperature change {change:e} K)",
                    sweeps.len()
                )));
            }
            return Ok(CoupledElectrothermalCompletion {
                sweeps,
                converged,
                thermal_workload: bound,
                thermal,
                electrical,
            });
        }
        temperatures = next;
    }
}

fn invalid(message: impl std::fmt::Display) -> AnalysisExecutionError {
    AnalysisExecutionError::Invalid(message.to_string())
}
