use super::super::{
    cfd::{
        interface_connectivity_coverage_ratio, interface_graph_edges, resample_scalar_profile,
        CfdDomainTopology,
    },
    coupling::dense,
};

#[derive(Debug, Clone)]
pub(super) struct ConjugateInterfaceSolution {
    pub fluid_temperature:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Temperature, Vec<f64>>,
    pub solid_temperature:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Temperature, Vec<f64>>,
    pub temperature_jump: runmat_types::physical::Quantity<
        runmat_types::physical::kind::TemperatureDifference,
        Vec<f64>,
    >,
    pub coupled_heat_flux:
        runmat_types::physical::Quantity<runmat_types::physical::kind::HeatFlux, Vec<f64>>,
    pub iteration_count: usize,
    pub coupled_interface_residual_ratio: f64,
    pub flux_temperature_law_residual_ratio: f64,
    pub heat_flux_realization_residual_ratio: f64,
    pub thermal_network_edge_count: usize,
    pub thermal_network_node_count: usize,
    pub interface_connectivity_coverage_ratio: f64,
    pub mesh_backed_interface_connectivity_ratio: f64,
    pub full_topology_edge_count: usize,
    pub full_topology_element_count: usize,
    pub thermal_network_residual_ratio: f64,
}

pub(super) fn solve_conjugate_interface(
    base_temperature: &[f64],
    heat_flux: &[f64],
    topology: &CfdDomainTopology,
    interface_conductance_w_per_m2k: f64,
    advection_shift_k: f64,
    reference_temperature_k: f64,
) -> Result<ConjugateInterfaceSolution, dense::DenseCouplingError> {
    let temperature_count = base_temperature.len().max(1);
    let interface_count = heat_flux.len().max(1);
    let interface_denom = interface_count.saturating_sub(1).max(1) as f64;
    let conductance = interface_conductance_w_per_m2k.max(1.0e-12);
    let axial_conductance = (0.05 * conductance).max(1.0e-12);
    let anchor_conductance = conductance;
    let base_interface_temperature = resample_scalar_profile(base_temperature, interface_count);
    let (thermal_network_edges, mesh_backed_interface_connectivity_ratio) =
        interface_graph_edges(topology, interface_count);
    let interface_connectivity_coverage_ratio = interface_connectivity_coverage_ratio(
        topology,
        interface_count,
        thermal_network_edges.len(),
    );
    let mut operator = vec![vec![0.0; interface_count]; interface_count];
    for (row, diagonal) in operator.iter_mut().enumerate() {
        diagonal[row] = anchor_conductance;
    }
    for (left, right) in &thermal_network_edges {
        operator[*left][*left] += axial_conductance;
        operator[*right][*right] += axial_conductance;
        operator[*left][*right] -= axial_conductance;
        operator[*right][*left] -= axial_conductance;
    }
    let center_rhs = (0..interface_count)
        .map(|index| {
            let base = base_interface_temperature
                .get(index)
                .copied()
                .unwrap_or(reference_temperature_k);
            let xi = index as f64 / interface_denom;
            anchor_conductance * (base + advection_shift_k * xi)
        })
        .collect::<Vec<_>>();

    let interface_center_temperature = dense::solve(&operator, &center_rhs)?;
    let center_reaction = dense::apply(&operator, &interface_center_temperature)?;
    let temperature_scale = interface_center_temperature
        .iter()
        .map(|value| (*value - reference_temperature_k).abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let thermal_network_residual_ratio = center_reaction
        .iter()
        .zip(center_rhs.iter())
        .map(|(reaction, rhs)| (reaction - rhs).abs())
        .fold(0.0_f64, f64::max)
        / (anchor_conductance * temperature_scale).max(1.0e-12);
    let heat_flux_scale = heat_flux
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max)
        .max(1.0e-12);
    let mut interface_fluid_temperature = Vec::with_capacity(interface_count);
    let mut interface_solid_temperature = Vec::with_capacity(interface_count);
    let mut temperature_jump = Vec::with_capacity(interface_count);
    let mut coupled_heat_flux = Vec::with_capacity(interface_count);
    let mut max_flux_temperature_law_residual = 0.0_f64;
    for (center, flux) in interface_center_temperature
        .iter()
        .zip(heat_flux.iter().chain(std::iter::repeat(&0.0)))
    {
        let jump = *flux / conductance;
        interface_fluid_temperature.push(*center + 0.5 * jump);
        interface_solid_temperature.push(*center - 0.5 * jump);
        temperature_jump.push(jump);
        coupled_heat_flux.push(*flux);
        max_flux_temperature_law_residual = max_flux_temperature_law_residual
            .max((conductance * jump - *flux).abs() / heat_flux_scale);
    }

    Ok(ConjugateInterfaceSolution {
        fluid_temperature: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Temperature,
            _,
        >::new(resample_scalar_profile(
            &interface_fluid_temperature,
            temperature_count,
        )),
        solid_temperature: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Temperature,
            _,
        >::new(resample_scalar_profile(
            &interface_solid_temperature,
            temperature_count,
        )),
        temperature_jump: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::TemperatureDifference,
            _,
        >::new(temperature_jump),
        coupled_heat_flux: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::HeatFlux,
            _,
        >::new(coupled_heat_flux),
        iteration_count: usize::from(temperature_count > 0),
        coupled_interface_residual_ratio: thermal_network_residual_ratio,
        flux_temperature_law_residual_ratio: max_flux_temperature_law_residual,
        heat_flux_realization_residual_ratio: 0.0,
        thermal_network_edge_count: thermal_network_edges.len(),
        thermal_network_node_count: interface_count,
        interface_connectivity_coverage_ratio,
        mesh_backed_interface_connectivity_ratio,
        full_topology_edge_count: topology.element_topology_edge_nodes.len(),
        full_topology_element_count: topology.element_topology_element_edges.len(),
        thermal_network_residual_ratio,
    })
}

#[cfg(test)]
mod tests {
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    use super::{solve_conjugate_interface, CfdDomainTopology};

    #[test]
    fn canonical_tetrahedron_preserves_flux_law_and_complete_interface_graph() {
        let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
        let topology = CfdDomainTopology::from_solver_mesh(&mesh);
        let heat_flux = [10.0, 20.0, 30.0, 40.0];
        let solution = solve_conjugate_interface(
            &[300.0, 301.0, 302.0, 303.0],
            &heat_flux,
            &topology,
            1_000.0,
            0.25,
            293.15,
        )
        .expect("anchored interface graph should solve");

        assert_eq!(solution.thermal_network_edge_count, 6);
        assert_eq!(solution.interface_connectivity_coverage_ratio, 1.0);
        assert_eq!(solution.mesh_backed_interface_connectivity_ratio, 1.0);
        assert!(solution.thermal_network_residual_ratio <= 1.0e-12);
        for (jump, flux) in solution
            .temperature_jump
            .canonical_values()
            .iter()
            .zip(heat_flux)
        {
            assert!((1_000.0 * jump - flux).abs() <= 1.0e-12);
        }
    }
}
