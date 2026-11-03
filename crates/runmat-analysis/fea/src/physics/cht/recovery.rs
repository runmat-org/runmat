use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use runmat_analysis_core::{
    AnalysisField, AnalysisInterfaceKind, AnalysisModel, AnalysisProcedure, CfdDomain,
    CoupledPhysicsFamily, PhysicsFamily, ThermalInterfaceTransfer,
};

use crate::{
    diagnostics::FeaDiagnostic, fea_cht_energy_residual_field_id,
    fea_cht_fluid_temperature_field_id, fea_cht_interface_heat_flux_field_id,
    fea_cht_interface_temperature_jump_field_id, fea_cht_solid_temperature_field_id,
    FeaThermalRunResult, FEA_FIELD_CHT_FLUID_PRESSURE, FEA_FIELD_CHT_FLUID_VELOCITY,
};

use super::{
    super::cfd::{
        self, interface_face_count, recover_velocity_pressure, resample_scalar_profile,
        CfdDomainTopology,
    },
    diagnostics, interface,
};

#[derive(Clone, Copy, Debug)]
pub struct ChtCouplingOptions {
    pub sample_count: usize,
    pub procedure: AnalysisProcedure,
    pub residual_warn_threshold: f64,
}

#[derive(Clone, Debug)]
pub struct ChtCouplingResult {
    pub fields: Vec<AnalysisField>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub max_cfd_momentum_residual: f64,
    pub max_cfd_continuity_residual: f64,
    pub max_transport_residual: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ChtInterfaceClosure {
    pub interface_face_count: usize,
    pub max_temperature_jump_k:
        runmat_types::physical::Quantity<runmat_types::physical::kind::TemperatureDifference, f64>,
    pub max_energy_residual: f64,
    pub heat_flux_balance_ratio: f64,
    pub mean_interface_heat_flux_w_per_m2:
        runmat_types::physical::Quantity<runmat_types::physical::kind::HeatFlux, f64>,
    pub max_thermal_transport_residual_ratio: f64,
    pub interface_temperature_continuity_ratio: f64,
    pub max_advection_temperature_shift_k:
        runmat_types::physical::Quantity<runmat_types::physical::kind::TemperatureDifference, f64>,
    pub interface_conductance_w_per_m2k: runmat_types::physical::Quantity<
        runmat_types::physical::kind::HeatTransferCoefficient,
        f64,
    >,
    pub max_flux_temperature_law_residual_ratio: f64,
    pub max_heat_flux_realization_residual_ratio: f64,
    pub max_coupled_interface_iteration_count: usize,
    pub max_coupled_interface_residual_ratio: f64,
    pub thermal_network_edge_count: usize,
    pub thermal_network_node_count: usize,
    pub interface_connectivity_coverage_ratio: f64,
    pub mesh_backed_interface_connectivity_ratio: f64,
    pub full_topology_edge_count: usize,
    pub full_topology_element_count: usize,
    pub max_thermal_network_residual_ratio: f64,
}

pub fn couple_thermal_flow(
    model: &AnalysisModel,
    domain: &CfdDomain,
    topology: &CfdDomainTopology,
    thermal_run: &FeaThermalRunResult,
    options: ChtCouplingOptions,
) -> Result<ChtCouplingResult, String> {
    let schedule = cfd::realize_velocity_schedule(
        model,
        PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer),
        options.procedure,
        options.sample_count,
    )?;
    let nominal_time_step_s = schedule.nominal_time_step_s.canonical_value();
    let nominal_velocity_m_per_s = schedule.nominal_velocity_m_per_s.canonical_value();
    let profile_scales = schedule.profile_scales;
    let field_step = options.sample_count.saturating_sub(1);
    let (fluid_velocity, fluid_pressure) = recover_velocity_pressure(
        domain,
        topology,
        field_step,
        nominal_velocity_m_per_s,
        &profile_scales,
    );
    let (residual_momentum, residual_continuity) = cfd::residual_norms(
        &fluid_velocity,
        &fluid_pressure,
        domain,
        topology,
        options.sample_count,
    );
    let max_cfd_momentum_residual = residual_momentum.iter().copied().fold(0.0_f64, f64::max);
    let max_cfd_continuity_residual = residual_continuity.iter().copied().fold(0.0_f64, f64::max);
    let inlet_velocity_m_per_s = cfd::mean_axial_velocity(&fluid_velocity);
    let pressure_drop_pa = cfd::pressure_drop(&fluid_pressure);
    let mut fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_CHT_FLUID_VELOCITY, AnalysisFieldLayout::vector3(ObservationQuantity::FluidVelocity, FieldLocation::Node),
            vec![topology.node_count, 3],
            fluid_velocity,
        ),
        AnalysisField::host_f64(
            FEA_FIELD_CHT_FLUID_PRESSURE, AnalysisFieldLayout::scalar(ObservationQuantity::FluidPressure, FieldLocation::Node),
            vec![topology.node_count],
            fluid_pressure,
        ),
    ];
    let authored_conductance = interface_conductance(model);
    let mut closure = ChtInterfaceClosure::default();

    for (step_index, temperature) in thermal_run.temperature_snapshots.iter().enumerate() {
        let fallback_len = temperature.element_count().max(1);
        let base_temperature = temperature
            .as_host_f64()
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| {
                vec![thermal_run.reference_temperature_k.canonical_value(); fallback_len]
            });
        let mut heat_flux = thermal_run
            .heat_flux_snapshots
            .get(step_index)
            .map(|field| field_scalar_magnitudes(field, fallback_len))
            .unwrap_or_else(|| vec![0.0; fallback_len]);
        if heat_flux.is_empty() {
            heat_flux.push(0.0);
        }
        let target_interface_count = interface_face_count(topology);
        if heat_flux.len() != target_interface_count {
            heat_flux = resample_scalar_profile(&heat_flux, target_interface_count);
        }
        let max_heat_flux = heat_flux
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let conductance = authored_conductance
            .unwrap_or_else(|| (max_heat_flux / 0.01).max(25.0))
            .max(1.0e-12);
        let advection_shift_k = advection_shift(
            domain,
            inlet_velocity_m_per_s,
            &base_temperature,
            thermal_run.reference_temperature_k.canonical_value(),
        );
        let solution = interface::solve_conjugate_interface(
            &base_temperature,
            &heat_flux,
            topology,
            conductance,
            advection_shift_k,
            thermal_run.reference_temperature_k.canonical_value(),
        )?;
        let interface_count = solution.coupled_heat_flux.canonical_values().len();
        accumulate_closure(
            &mut closure,
            &solution,
            &base_temperature,
            thermal_run.reference_temperature_k.canonical_value(),
            conductance,
            advection_shift_k,
        );
        fields.push(AnalysisField::host_f64(
            fea_cht_fluid_temperature_field_id(step_index), AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node).step(step_index),
            vec![base_temperature.len().max(1)],
            solution.fluid_temperature.into_canonical_values(),
        ));
        fields.push(AnalysisField::host_f64(
            fea_cht_solid_temperature_field_id(step_index), AnalysisFieldLayout::scalar(ObservationQuantity::Temperature, FieldLocation::Node).step(step_index),
            vec![base_temperature.len().max(1)],
            solution.solid_temperature.into_canonical_values(),
        ));

        let heat_balance_ratio = heat_balance_ratio(solution.coupled_heat_flux.canonical_values());
        let energy_residual = heat_balance_ratio
            .max(solution.flux_temperature_law_residual_ratio)
            .max(solution.heat_flux_realization_residual_ratio)
            .max(solution.thermal_network_residual_ratio);
        fields.push(AnalysisField::host_f64(
            fea_cht_interface_heat_flux_field_id(step_index), AnalysisFieldLayout::scalar(ObservationQuantity::HeatFlux, FieldLocation::Interface).step(step_index),
            vec![interface_count],
            solution.coupled_heat_flux.into_canonical_values(),
        ));
        fields.push(AnalysisField::host_f64(
            fea_cht_interface_temperature_jump_field_id(step_index), AnalysisFieldLayout::scalar(ObservationQuantity::TemperatureJump, FieldLocation::Interface).step(step_index),
            vec![interface_count],
            solution.temperature_jump.into_canonical_values(),
        ));
        fields.push(AnalysisField::host_f64(
            fea_cht_energy_residual_field_id(step_index), AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global).step(step_index),
            vec![1],
            vec![energy_residual],
        ));
    }

    let max_transport_residual = closure.max_thermal_transport_residual_ratio.max(
        closure
            .max_energy_residual
            .max(closure.heat_flux_balance_ratio),
    );
    let applied_temperature_delta_k = thermal_run
        .temperature_snapshots
        .iter()
        .filter_map(AnalysisField::as_host_f64)
        .flatten()
        .map(|temperature| {
            (temperature - thermal_run.reference_temperature_k.canonical_value()).abs()
        })
        .fold(0.0_f64, f64::max);
    let mut result_diagnostics = cfd::coupling_diagnostics(
        domain,
        topology,
        cfd::CfdCouplingEvidence {
            inlet_velocity_m_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Velocity,
                _,
            >::new(inlet_velocity_m_per_s),
            executed_sample_count: profile_scales.len(),
            nominal_time_step_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Time,
                _,
            >::new(nominal_time_step_s),
            transient: options.procedure == AnalysisProcedure::TransientResponse,
            pressure_drop_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Pressure,
                _,
            >::new(pressure_drop_pa),
            mass_balance_residual: max_cfd_continuity_residual,
            max_momentum_residual: max_cfd_momentum_residual,
            max_continuity_residual: max_cfd_continuity_residual,
            residual_warn_threshold: options.residual_warn_threshold,
        },
    );
    result_diagnostics.extend(diagnostics::coupling_diagnostics(
        diagnostics::ChtDiagnosticInputs {
            authored_interface_count: model.interfaces.len(),
            reference_temperature_k: runmat_types::physical::Temperature::new(
                thermal_run.reference_temperature_k.canonical_value(),
            ),
            applied_temperature_delta_k: runmat_types::physical::TemperatureDifference::new(
                applied_temperature_delta_k,
            ),
            inlet_velocity_m_per_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Velocity,
                _,
            >::new(inlet_velocity_m_per_s),
            nominal_time_step_s: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Time,
                _,
            >::new(nominal_time_step_s),
            options,
            domain,
            closure: &closure,
        },
    ));
    Ok(ChtCouplingResult {
        fields,
        diagnostics: result_diagnostics,
        max_cfd_momentum_residual,
        max_cfd_continuity_residual,
        max_transport_residual,
    })
}

fn accumulate_closure(
    closure: &mut ChtInterfaceClosure,
    solution: &interface::ConjugateInterfaceSolution,
    base_temperature: &[f64],
    reference_temperature_k: f64,
    conductance: f64,
    advection_shift_k: f64,
) {
    let heat_balance_ratio = heat_balance_ratio(solution.coupled_heat_flux.canonical_values());
    let max_jump = solution
        .temperature_jump
        .canonical_values()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let thermal_scale_k = base_temperature
        .iter()
        .map(|value| (value - reference_temperature_k).abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let transport_residual = solution
        .flux_temperature_law_residual_ratio
        .max(heat_balance_ratio)
        .max(solution.heat_flux_realization_residual_ratio)
        .max(solution.coupled_interface_residual_ratio)
        .max(solution.thermal_network_residual_ratio);
    closure.interface_face_count = closure
        .interface_face_count
        .max(solution.coupled_heat_flux.canonical_values().len());
    closure.max_temperature_jump_k = runmat_types::physical::Quantity::new(
        closure
            .max_temperature_jump_k
            .canonical_value()
            .max(max_jump),
    );
    closure.max_energy_residual = closure.max_energy_residual.max(transport_residual);
    closure.heat_flux_balance_ratio = closure.heat_flux_balance_ratio.max(heat_balance_ratio);
    closure.mean_interface_heat_flux_w_per_m2 = runmat_types::physical::Quantity::new(
        closure
            .mean_interface_heat_flux_w_per_m2
            .canonical_value()
            .max(
                solution
                    .coupled_heat_flux
                    .canonical_values()
                    .iter()
                    .copied()
                    .sum::<f64>()
                    .abs()
                    / solution.coupled_heat_flux.canonical_values().len().max(1) as f64,
            ),
    );
    closure.max_thermal_transport_residual_ratio = closure
        .max_thermal_transport_residual_ratio
        .max(transport_residual);
    closure.interface_temperature_continuity_ratio = closure
        .interface_temperature_continuity_ratio
        .max((1.0 - max_jump / thermal_scale_k).clamp(0.0, 1.0));
    closure.max_advection_temperature_shift_k = runmat_types::physical::Quantity::new(
        closure
            .max_advection_temperature_shift_k
            .canonical_value()
            .max(advection_shift_k.abs()),
    );
    closure.interface_conductance_w_per_m2k = runmat_types::physical::Quantity::new(
        closure
            .interface_conductance_w_per_m2k
            .canonical_value()
            .max(conductance),
    );
    closure.max_flux_temperature_law_residual_ratio = closure
        .max_flux_temperature_law_residual_ratio
        .max(solution.flux_temperature_law_residual_ratio);
    closure.max_heat_flux_realization_residual_ratio = closure
        .max_heat_flux_realization_residual_ratio
        .max(solution.heat_flux_realization_residual_ratio);
    closure.max_coupled_interface_iteration_count = closure
        .max_coupled_interface_iteration_count
        .max(solution.iteration_count);
    closure.max_coupled_interface_residual_ratio = closure
        .max_coupled_interface_residual_ratio
        .max(solution.coupled_interface_residual_ratio);
    closure.thermal_network_edge_count = closure
        .thermal_network_edge_count
        .max(solution.thermal_network_edge_count);
    closure.thermal_network_node_count = closure
        .thermal_network_node_count
        .max(solution.thermal_network_node_count);
    closure.interface_connectivity_coverage_ratio = closure
        .interface_connectivity_coverage_ratio
        .max(solution.interface_connectivity_coverage_ratio);
    closure.mesh_backed_interface_connectivity_ratio = closure
        .mesh_backed_interface_connectivity_ratio
        .max(solution.mesh_backed_interface_connectivity_ratio);
    closure.full_topology_edge_count = closure
        .full_topology_edge_count
        .max(solution.full_topology_edge_count);
    closure.full_topology_element_count = closure
        .full_topology_element_count
        .max(solution.full_topology_element_count);
    closure.max_thermal_network_residual_ratio = closure
        .max_thermal_network_residual_ratio
        .max(solution.thermal_network_residual_ratio);
}

fn heat_balance_ratio(heat_flux: &[f64]) -> f64 {
    let fluid_heat = heat_flux.iter().sum::<f64>();
    let solid_heat = -fluid_heat;
    (fluid_heat + solid_heat).abs() / (fluid_heat.abs() + solid_heat.abs() + 1.0e-12)
}

fn field_scalar_magnitudes(field: &AnalysisField, fallback_len: usize) -> Vec<f64> {
    let Some(values) = field.as_host_f64() else {
        return vec![0.0; fallback_len.max(1)];
    };
    match field.shape.as_slice() {
        [count, components] if *components > 1 => (0..*count)
            .map(|index| {
                let start = index * *components;
                values
                    .get(start..start + *components)
                    .unwrap_or(&[])
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt()
            })
            .collect(),
        _ => values.to_vec(),
    }
}

fn advection_shift(
    domain: &CfdDomain,
    mean_axial_velocity_m_per_s: f64,
    temperature: &[f64],
    reference_temperature_k: f64,
) -> f64 {
    let thermal_span_k = temperature
        .iter()
        .map(|value| (value - reference_temperature_k).abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let reynolds_ratio = (cfd::reynolds_number_for_velocity(domain, mean_axial_velocity_m_per_s)
        / 1.0e5)
        .clamp(0.0, 5.0);
    thermal_span_k
        * reynolds_ratio
        * domain
            .turbulence_intensity
            .canonical_value()
            .clamp(0.0, 1.0)
        * 2.0e-3
}

fn interface_conductance(model: &AnalysisModel) -> Option<f64> {
    model
        .interfaces
        .iter()
        .find_map(|interface| match &interface.kind {
            AnalysisInterfaceKind::ConjugateHeatTransfer(interface) => Some(interface),
            AnalysisInterfaceKind::FluidStructure(_) | AnalysisInterfaceKind::Contact(_) => None,
        })
        .and_then(|interface| match interface.transfer {
            ThermalInterfaceTransfer::Perfect => None,
            ThermalInterfaceTransfer::Conductance(conductance) => {
                Some(conductance.canonical_value())
            }
            ThermalInterfaceTransfer::Resistance(resistance) => {
                Some(1.0 / resistance.canonical_value())
            }
        })
}
