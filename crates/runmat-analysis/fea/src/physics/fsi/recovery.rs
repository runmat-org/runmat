use runmat_analysis_core::{
    AnalysisField, AnalysisModel, AnalysisProcedure, CfdDomain, CoupledPhysicsFamily, PhysicsFamily,
};

use crate::{
    diagnostics::FeaDiagnostic, fea_fsi_coupling_iteration_count_field_id,
    fea_fsi_fluid_pressure_field_id, fea_fsi_fluid_velocity_field_id,
    fea_fsi_interface_displacement_field_id, fea_fsi_interface_pressure_field_id,
    fea_fsi_interface_residual_field_id, fea_fsi_interface_traction_field_id,
    fea_fsi_structural_displacement_field_id,
};

use super::{super::cfd, diagnostics, interface};

#[derive(Clone, Copy, Debug)]
pub struct FsiCouplingOptions {
    pub sample_count: usize,
    pub max_linear_iters: usize,
    pub tolerance: f64,
    pub residual_warn_threshold: f64,
}

#[derive(Clone, Debug)]
pub struct FsiCouplingResult {
    pub fields: Vec<AnalysisField>,
    pub diagnostics: Vec<FeaDiagnostic>,
    pub max_cfd_momentum_residual: f64,
    pub max_cfd_continuity_residual: f64,
    pub max_interface_residual: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct FsiInterfaceClosure {
    pub interface_node_count: usize,
    pub interface_face_count: usize,
    pub max_interface_residual: f64,
    pub force_balance_ratio: f64,
    pub max_displacement_transfer_residual_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Displacement, f64>,
    pub max_interface_displacement_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Displacement, f64>,
    pub mean_interface_pressure_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, f64>,
    pub max_traction_magnitude_pa:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Traction, f64>,
    pub max_coupling_iteration_count: usize,
    pub max_pressure_feedback_residual_ratio: f64,
    pub max_two_way_interface_residual_ratio: f64,
    pub max_structural_traction_update_residual_ratio: f64,
    pub max_pressure_displacement_law_residual_ratio: f64,
    pub max_structural_solve_residual_ratio: f64,
    pub max_interface_work_j_per_m2:
        runmat_types::physical::Quantity<runmat_types::physical::kind::SurfaceEnergyDensity, f64>,
    pub max_structural_strain_energy_j_per_m2:
        runmat_types::physical::Quantity<runmat_types::physical::kind::SurfaceEnergyDensity, f64>,
    pub max_interface_work_energy_residual_ratio: f64,
    pub structural_coupling_edge_count: usize,
    pub interface_connectivity_coverage_ratio: f64,
    pub mesh_backed_interface_connectivity_ratio: f64,
    pub full_topology_edge_count: usize,
    pub full_topology_element_count: usize,
    pub interface_stiffness_pa_per_m: runmat_types::physical::Quantity<
        runmat_types::physical::kind::TractionDisplacementStiffness,
        f64,
    >,
}

pub fn couple_fluid_structure(
    model: &AnalysisModel,
    domain: &CfdDomain,
    topology: &cfd::CfdDomainTopology,
    options: FsiCouplingOptions,
) -> Result<FsiCouplingResult, super::super::coupling::dense::DenseCouplingError> {
    let sample_count = options.sample_count;
    let schedule = cfd::realize_velocity_schedule(
        model,
        PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction),
        AnalysisProcedure::TransientResponse,
        sample_count,
    )?;
    let nominal_time_step_s = schedule.nominal_time_step_s.canonical_value();
    let nominal_velocity_m_per_s = schedule.nominal_velocity_m_per_s.canonical_value();
    let profile_scales = schedule.profile_scales;
    let final_step = sample_count.saturating_sub(1);
    let (final_velocity, final_pressure) = cfd::recover_velocity_pressure(
        domain,
        topology,
        final_step,
        nominal_velocity_m_per_s,
        &profile_scales,
    );
    let (residual_momentum, residual_continuity) = cfd::residual_norms(
        &final_velocity,
        &final_pressure,
        domain,
        topology,
        sample_count,
    );
    let max_cfd_momentum_residual = residual_momentum.iter().copied().fold(0.0_f64, f64::max);
    let max_cfd_continuity_residual = residual_continuity.iter().copied().fold(0.0_f64, f64::max);
    let structural_compliance_per_pa = structural_compliance_per_pa(model);
    let mut fields = Vec::new();
    let mut closure = FsiInterfaceClosure::default();

    for step_index in 0..sample_count {
        let (fluid_velocity, fluid_pressure) = cfd::recover_velocity_pressure(
            domain,
            topology,
            step_index,
            nominal_velocity_m_per_s,
            &profile_scales,
        );
        let interface_step = interface::solve_partitioned_interface(
            &fluid_pressure,
            topology,
            structural_compliance_per_pa,
            options.max_linear_iters,
            options.tolerance,
        )?;
        accumulate_closure(&mut closure, &interface_step, &fluid_pressure);

        fields.push(AnalysisField::host_f64(
            fea_fsi_fluid_velocity_field_id(step_index),
            vec![topology.node_count, 3],
            fluid_velocity,
        ));
        fields.push(AnalysisField::host_f64(
            fea_fsi_fluid_pressure_field_id(step_index),
            vec![topology.node_count],
            fluid_pressure,
        ));
        fields.push(AnalysisField::host_f64(
            fea_fsi_structural_displacement_field_id(step_index),
            vec![topology.node_count, 3],
            interface_step
                .structural_displacement
                .into_canonical_values(),
        ));
        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_displacement_field_id(step_index),
            vec![topology.node_count, 3],
            interface_step
                .interface_displacement
                .into_canonical_values(),
        ));
        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_pressure_field_id(step_index),
            vec![interface_step
                .interface_pressure
                .canonical_values()
                .len()
                .max(1)],
            interface_step.interface_pressure.canonical_values().clone(),
        ));
        let traction = interface_step
            .interface_pressure
            .canonical_values()
            .iter()
            .flat_map(|pressure| [-*pressure, 0.0, 0.0])
            .collect::<Vec<_>>();
        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_traction_field_id(step_index),
            vec![
                interface_step
                    .interface_pressure
                    .canonical_values()
                    .len()
                    .max(1),
                3,
            ],
            traction,
        ));
        let residual = residual_momentum
            .get(step_index)
            .copied()
            .unwrap_or(0.0)
            .max(residual_continuity.get(step_index).copied().unwrap_or(0.0))
            .max(interface_step.two_way_interface_residual_ratio);
        closure.max_interface_residual = closure.max_interface_residual.max(residual);
        fields.push(AnalysisField::host_f64(
            fea_fsi_interface_residual_field_id(step_index),
            vec![1],
            vec![residual],
        ));
        fields.push(AnalysisField::host_f64(
            fea_fsi_coupling_iteration_count_field_id(step_index),
            vec![1],
            vec![interface_step.iteration_count as f64],
        ));
    }

    let inlet_velocity_m_per_s = cfd::mean_axial_velocity(&final_velocity);
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
            transient: true,
            pressure_drop_pa: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Pressure,
                _,
            >::new(cfd::pressure_drop(&final_pressure)),
            mass_balance_residual: max_cfd_continuity_residual,
            max_momentum_residual: max_cfd_momentum_residual,
            max_continuity_residual: max_cfd_continuity_residual,
            residual_warn_threshold: options.residual_warn_threshold,
        },
    );
    result_diagnostics.extend(diagnostics::coupling_diagnostics(
        model,
        profile_scales.len(),
        nominal_time_step_s,
        structural_compliance_per_pa,
        options,
        &closure,
    ));

    Ok(FsiCouplingResult {
        fields,
        diagnostics: result_diagnostics,
        max_cfd_momentum_residual,
        max_cfd_continuity_residual,
        max_interface_residual: closure.max_interface_residual,
    })
}

fn accumulate_closure(
    closure: &mut FsiInterfaceClosure,
    step: &interface::PartitionedInterfaceStep,
    fluid_pressure: &[f64],
) {
    closure.interface_node_count = closure.interface_node_count.max(fluid_pressure.len());
    closure.interface_face_count = closure
        .interface_face_count
        .max(step.interface_pressure.canonical_values().len());
    closure.max_coupling_iteration_count = closure
        .max_coupling_iteration_count
        .max(step.iteration_count);
    closure.max_pressure_feedback_residual_ratio = closure
        .max_pressure_feedback_residual_ratio
        .max(step.pressure_feedback_residual_ratio);
    closure.max_two_way_interface_residual_ratio = closure
        .max_two_way_interface_residual_ratio
        .max(step.two_way_interface_residual_ratio);
    closure.max_structural_traction_update_residual_ratio = closure
        .max_structural_traction_update_residual_ratio
        .max(step.structural_traction_update_residual_ratio);
    closure.max_pressure_displacement_law_residual_ratio = closure
        .max_pressure_displacement_law_residual_ratio
        .max(step.pressure_displacement_law_residual_ratio);
    closure.max_structural_solve_residual_ratio = closure
        .max_structural_solve_residual_ratio
        .max(step.structural_solve_residual_ratio);
    closure.max_interface_work_energy_residual_ratio = closure
        .max_interface_work_energy_residual_ratio
        .max(step.interface_work_energy_residual_ratio);
    closure.max_interface_work_j_per_m2 = runmat_types::physical::Quantity::new(
        closure
            .max_interface_work_j_per_m2
            .canonical_value()
            .max(step.interface_work_j_per_m2.canonical_value().abs()),
    );
    closure.max_structural_strain_energy_j_per_m2 = runmat_types::physical::Quantity::new(
        closure
            .max_structural_strain_energy_j_per_m2
            .canonical_value()
            .max(
                step.structural_strain_energy_j_per_m2
                    .canonical_value()
                    .abs(),
            ),
    );
    closure.structural_coupling_edge_count = closure
        .structural_coupling_edge_count
        .max(step.structural_coupling_edge_count);
    closure.interface_connectivity_coverage_ratio = closure
        .interface_connectivity_coverage_ratio
        .max(step.interface_connectivity_coverage_ratio);
    closure.mesh_backed_interface_connectivity_ratio = closure
        .mesh_backed_interface_connectivity_ratio
        .max(step.mesh_backed_interface_connectivity_ratio);
    closure.full_topology_edge_count = closure
        .full_topology_edge_count
        .max(step.full_topology_edge_count);
    closure.full_topology_element_count = closure
        .full_topology_element_count
        .max(step.full_topology_element_count);
    closure.interface_stiffness_pa_per_m = runmat_types::physical::Quantity::new(
        closure
            .interface_stiffness_pa_per_m
            .canonical_value()
            .max(step.interface_stiffness_pa_per_m.canonical_value()),
    );
    closure.force_balance_ratio = closure.force_balance_ratio.max(0.0);
    let mean_pressure = step
        .interface_pressure
        .canonical_values()
        .iter()
        .sum::<f64>()
        / step.interface_pressure.canonical_values().len().max(1) as f64;
    closure.mean_interface_pressure_pa = runmat_types::physical::Quantity::new(
        closure
            .mean_interface_pressure_pa
            .canonical_value()
            .max(mean_pressure.abs()),
    );
    closure.max_traction_magnitude_pa = runmat_types::physical::Quantity::new(
        closure.max_traction_magnitude_pa.canonical_value().max(
            step.interface_pressure
                .canonical_values()
                .iter()
                .map(|pressure| pressure.abs())
                .fold(0.0_f64, f64::max),
        ),
    );
    closure.max_displacement_transfer_residual_m = runmat_types::physical::Quantity::new(
        closure
            .max_displacement_transfer_residual_m
            .canonical_value()
            .max(
                step.structural_displacement
                    .canonical_values()
                    .iter()
                    .zip(step.interface_displacement.canonical_values().iter())
                    .map(|(structural, interface)| (structural - interface).abs())
                    .fold(0.0_f64, f64::max),
            ),
    );
    closure.max_interface_displacement_m = runmat_types::physical::Quantity::new(
        closure.max_interface_displacement_m.canonical_value().max(
            step.interface_displacement
                .canonical_values()
                .chunks_exact(3)
                .map(|components| {
                    (components[0] * components[0]
                        + components[1] * components[1]
                        + components[2] * components[2])
                        .sqrt()
                })
                .fold(0.0_f64, f64::max),
        ),
    );
}

fn structural_compliance_per_pa(model: &AnalysisModel) -> f64 {
    let (sum, count) = model
        .materials
        .iter()
        .filter_map(|material| {
            let modulus = material
                .require_mechanical()
                .ok()?
                .youngs_modulus_pa
                .canonical_value();
            (modulus.is_finite() && modulus > 0.0).then_some(modulus)
        })
        .fold((0.0_f64, 0_usize), |(sum, count), modulus| {
            (sum + modulus, count + 1)
        });
    let youngs_modulus = if count > 0 {
        sum / count as f64
    } else {
        200.0e9
    };
    1.0 / youngs_modulus.max(1.0e6)
}
