use super::super::{cfd, coupling::dense};

#[derive(Debug, Clone)]
pub(super) struct PartitionedInterfaceStep {
    pub interface_pressure:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Vec<f64>>,
    pub structural_displacement:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Displacement, Vec<f64>>,
    pub interface_displacement:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Displacement, Vec<f64>>,
    pub iteration_count: usize,
    pub pressure_feedback_residual_ratio: f64,
    pub two_way_interface_residual_ratio: f64,
    pub structural_traction_update_residual_ratio: f64,
    pub pressure_displacement_law_residual_ratio: f64,
    pub structural_solve_residual_ratio: f64,
    pub interface_work_j_per_m2:
        runmat_types::physical::Quantity<runmat_types::physical::kind::SurfaceEnergyDensity, f64>,
    pub structural_strain_energy_j_per_m2:
        runmat_types::physical::Quantity<runmat_types::physical::kind::SurfaceEnergyDensity, f64>,
    pub interface_work_energy_residual_ratio: f64,
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

#[derive(Debug, Clone)]
struct StructuralInterfaceResponse {
    displacement_x:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Displacement, Vec<f64>>,
    reaction_pressure:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Pressure, Vec<f64>>,
    residual_ratio: f64,
    coupling_edge_count: usize,
}

pub(super) fn solve_partitioned_interface(
    fluid_pressure: &[f64],
    topology: &cfd::CfdDomainTopology,
    structural_compliance_per_pa: f64,
    max_linear_iters: usize,
    tolerance: f64,
) -> Result<PartitionedInterfaceStep, dense::DenseCouplingError> {
    let max_linear_iters = max_linear_iters.max(1);
    let tolerance = tolerance.max(1.0e-12);
    let interface_face_count = cfd::interface_face_count(topology);
    let face_pressure = cfd::resample_scalar_profile(
        &cfd::cell_centered_scalar(fluid_pressure),
        interface_face_count,
    );
    let pressure_scale = face_pressure
        .iter()
        .map(|pressure| pressure.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let compliance = structural_compliance_per_pa.max(1.0e-18);
    let pressure_relaxation = 0.65_f64;
    let displacement_relaxation = 0.65_f64;
    let traction_relaxation = 0.65_f64;
    let interface_stiffness_pa_per_m = 1.0 / compliance;
    let feedback_stiffness_pa_per_m = 0.25 * interface_stiffness_pa_per_m;
    let (structural_coupling_edges, mesh_backed_interface_connectivity_ratio) =
        cfd::interface_graph_edges(topology, interface_face_count);
    let interface_connectivity_coverage_ratio = cfd::interface_connectivity_coverage_ratio(
        topology,
        interface_face_count,
        structural_coupling_edges.len(),
    );
    let mut interface_pressure = vec![0.0; face_pressure.len()];
    let mut structural_traction = vec![0.0; face_pressure.len()];
    let mut structural_face_displacement = vec![0.0; face_pressure.len() * 3];
    let mut interface_face_displacement = vec![0.0; face_pressure.len() * 3];
    let mut iteration_count = 0_usize;
    let mut pressure_feedback_residual_ratio = if face_pressure.is_empty() { 0.0 } else { 1.0 };
    let mut displacement_transfer_residual_ratio = 0.0_f64;
    let mut structural_traction_update_residual_ratio =
        if face_pressure.is_empty() { 0.0 } else { 1.0 };
    let mut structural_solve_residual_ratio = if face_pressure.is_empty() { 0.0 } else { 1.0 };
    let mut structural_coupling_edge_count = 0_usize;

    for iteration in 1..=max_linear_iters {
        let target_pressure = face_pressure
            .iter()
            .enumerate()
            .map(|(face, pressure)| {
                let displacement = interface_face_displacement
                    .get(face * 3)
                    .copied()
                    .unwrap_or(0.0);
                *pressure - feedback_stiffness_pa_per_m * displacement
            })
            .collect::<Vec<_>>();
        for (interface, target) in interface_pressure.iter_mut().zip(target_pressure.iter()) {
            *interface += pressure_relaxation * (*target - *interface);
        }
        let structural_response = solve_structural_interface_response(
            &interface_pressure,
            interface_stiffness_pa_per_m,
            &structural_coupling_edges,
        )?;
        structural_solve_residual_ratio = structural_response.residual_ratio;
        structural_coupling_edge_count =
            structural_coupling_edge_count.max(structural_response.coupling_edge_count);
        for (face, displacement_x) in structural_response
            .displacement_x
            .canonical_values()
            .iter()
            .enumerate()
        {
            structural_face_displacement[face * 3] = *displacement_x;
            structural_face_displacement[face * 3 + 1] = 0.0;
            structural_face_displacement[face * 3 + 2] = 0.0;
        }
        let target_structural_traction = structural_response.reaction_pressure.canonical_values();
        for (traction, target) in structural_traction
            .iter_mut()
            .zip(target_structural_traction.iter())
        {
            *traction += traction_relaxation * (*target - *traction);
        }
        for (interface, structural) in interface_face_displacement
            .iter_mut()
            .zip(structural_face_displacement.iter())
        {
            *interface += displacement_relaxation * (*structural - *interface);
        }
        pressure_feedback_residual_ratio = target_pressure
            .iter()
            .zip(interface_pressure.iter())
            .map(|(target, interface)| (target - interface).abs())
            .fold(0.0_f64, f64::max)
            / pressure_scale;
        let displacement_scale = structural_face_displacement
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max)
            .max(1.0e-18);
        displacement_transfer_residual_ratio = structural_face_displacement
            .iter()
            .zip(interface_face_displacement.iter())
            .map(|(structural, interface)| (structural - interface).abs())
            .fold(0.0_f64, f64::max)
            / displacement_scale;
        structural_traction_update_residual_ratio = target_structural_traction
            .iter()
            .zip(structural_traction.iter())
            .map(|(target, traction)| (target - traction).abs())
            .fold(0.0_f64, f64::max)
            / pressure_scale;
        iteration_count = iteration;
        if pressure_feedback_residual_ratio <= tolerance
            && displacement_transfer_residual_ratio <= tolerance
            && structural_traction_update_residual_ratio <= tolerance
            && structural_solve_residual_ratio <= tolerance
        {
            break;
        }
    }

    let structural_response = solve_structural_interface_response(
        &interface_pressure,
        interface_stiffness_pa_per_m,
        &structural_coupling_edges,
    )?;
    structural_solve_residual_ratio = structural_solve_residual_ratio
        .max(structural_response.residual_ratio)
        .min(1.0);
    structural_coupling_edge_count =
        structural_coupling_edge_count.max(structural_response.coupling_edge_count);
    let pressure_displacement_law_residual_ratio = structural_response
        .reaction_pressure
        .canonical_values()
        .iter()
        .zip(interface_pressure.iter())
        .map(|(reaction, pressure)| (reaction - pressure).abs() / pressure.abs().max(1.0))
        .fold(0.0_f64, f64::max);
    let interface_work_j_per_m2 = interface_pressure
        .iter()
        .zip(structural_response.displacement_x.canonical_values().iter())
        .map(|(pressure, displacement)| pressure * displacement)
        .sum::<f64>();
    let structural_strain_energy_j_per_m2 = 0.5
        * structural_response
            .reaction_pressure
            .canonical_values()
            .iter()
            .zip(structural_response.displacement_x.canonical_values().iter())
            .map(|(reaction, displacement)| reaction * displacement)
            .sum::<f64>();
    let interface_work_energy_residual_ratio =
        (interface_work_j_per_m2 - 2.0 * structural_strain_energy_j_per_m2).abs()
            / (interface_work_j_per_m2.abs()
                + (2.0 * structural_strain_energy_j_per_m2).abs()
                + 1.0e-12);
    let interface_face_displacement_x = interface_face_displacement
        .chunks_exact(3)
        .map(|components| components[0])
        .collect::<Vec<_>>();

    Ok(PartitionedInterfaceStep {
        interface_pressure: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(interface_pressure),
        structural_displacement: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Displacement,
            _,
        >::new(vector_field_from_x_profile(
            structural_response.displacement_x.canonical_values(),
            fluid_pressure.len(),
        )),
        interface_displacement: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Displacement,
            _,
        >::new(vector_field_from_x_profile(
            &interface_face_displacement_x,
            fluid_pressure.len(),
        )),
        iteration_count,
        pressure_feedback_residual_ratio,
        two_way_interface_residual_ratio: pressure_feedback_residual_ratio
            .max(displacement_transfer_residual_ratio)
            .max(structural_traction_update_residual_ratio)
            .max(structural_solve_residual_ratio),
        structural_traction_update_residual_ratio,
        pressure_displacement_law_residual_ratio,
        structural_solve_residual_ratio,
        interface_work_j_per_m2: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::SurfaceEnergyDensity,
            _,
        >::new(interface_work_j_per_m2),
        structural_strain_energy_j_per_m2: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::SurfaceEnergyDensity,
            _,
        >::new(structural_strain_energy_j_per_m2),
        interface_work_energy_residual_ratio,
        structural_coupling_edge_count,
        interface_connectivity_coverage_ratio,
        mesh_backed_interface_connectivity_ratio,
        full_topology_edge_count: topology.element_topology_edge_nodes.len(),
        full_topology_element_count: topology.element_topology_element_edges.len(),
        interface_stiffness_pa_per_m: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::TractionDisplacementStiffness,
            _,
        >::new(interface_stiffness_pa_per_m),
    })
}

fn solve_structural_interface_response(
    pressure_load: &[f64],
    interface_stiffness_pa_per_m: f64,
    coupling_edges: &[(usize, usize)],
) -> Result<StructuralInterfaceResponse, dense::DenseCouplingError> {
    let face_count = pressure_load.len();
    if face_count == 0 {
        return Ok(StructuralInterfaceResponse {
            displacement_x: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Displacement,
                _,
            >::new(Vec::new()),
            reaction_pressure: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Pressure,
                _,
            >::new(Vec::new()),
            residual_ratio: 0.0,
            coupling_edge_count: 0,
        });
    }

    let diagonal_stiffness = interface_stiffness_pa_per_m.max(1.0e-9);
    let coupling_stiffness = 0.20 * diagonal_stiffness;
    let mut operator = vec![vec![0.0; face_count]; face_count];
    for (row, diagonal) in operator.iter_mut().enumerate() {
        diagonal[row] = diagonal_stiffness;
    }
    for (left, right) in coupling_edges {
        operator[*left][*left] += coupling_stiffness;
        operator[*right][*right] += coupling_stiffness;
        operator[*left][*right] -= coupling_stiffness;
        operator[*right][*left] -= coupling_stiffness;
    }

    let displacement_x = dense::solve(&operator, pressure_load)?;
    let reaction_pressure = dense::apply(&operator, &displacement_x)?;
    let pressure_scale = pressure_load
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let residual_ratio = reaction_pressure
        .iter()
        .zip(pressure_load.iter())
        .map(|(reaction, pressure)| (reaction - pressure).abs())
        .fold(0.0_f64, f64::max)
        / pressure_scale;

    Ok(StructuralInterfaceResponse {
        displacement_x: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Displacement,
            _,
        >::new(displacement_x),
        reaction_pressure: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(reaction_pressure),
        residual_ratio,
        coupling_edge_count: coupling_edges.len(),
    })
}

fn vector_field_from_x_profile(x: &[f64], target_count: usize) -> Vec<f64> {
    let template = cfd::resample_scalar_profile(x, target_count);
    let mut field = Vec::with_capacity(template.len() * 3);
    for displacement_x in template {
        field.extend_from_slice(&[displacement_x, 0.0, 0.0]);
    }
    field
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    use super::{cfd, solve_partitioned_interface};

    #[test]
    fn canonical_tetrahedron_preserves_complete_interface_graph_and_structural_balance() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let topology = cfd::CfdDomainTopology::from_solver_mesh(&mesh);
        let result = solve_partitioned_interface(
            &[104.0, 103.0, 102.0, 101.0],
            &topology,
            5.0e-12,
            128,
            1.0e-8,
        )
        .expect("anchored structural interface should solve");

        assert_eq!(result.structural_coupling_edge_count, 6);
        assert_eq!(result.interface_connectivity_coverage_ratio, 1.0);
        assert_eq!(result.mesh_backed_interface_connectivity_ratio, 1.0);
        assert!(result.structural_solve_residual_ratio <= 1.0e-12);
        assert!(result.pressure_displacement_law_residual_ratio <= 1.0e-12);
        assert!(result.interface_work_energy_residual_ratio <= 1.0e-12);
    }
}
