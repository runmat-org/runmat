use runmat_analysis_fea::physics::electrical::ElectricalConstitutiveResponse;

// Independent continuum solution on a unit prism; no production operator, quadrature,
// recovery routine or manufactured algebraic RHS is used to construct this reference.
pub(super) fn verify(
    completion: &runmat_runtime::analysis::ElectricalStudyCompletion,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    gauge: f64,
    voltage: f64,
    fields: &[Vec<f64>; 4],
) {
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(mesh).unwrap();
    let r = completion.result.response();
    assert!(r
        .ordered_node_ids
        .iter()
        .copied()
        .eq(projection.nodes.iter().map(|n| n.node_id)));
    assert!(r
        .ordered_element_ids
        .iter()
        .copied()
        .eq(projection.elements.iter().map(|e| e.element_id)));
    assert_eq!(&fields[0], r.potential_v.canonical_values());
    for (&actual, node) in fields[0].iter().zip(&projection.nodes) {
        near(
            actual,
            gauge + voltage * node.coordinates_m.canonical_value()[0],
            voltage.abs(),
            "potential",
        );
    }
    let (coefficient, total, density, vectors, electrostatic) = match &r.constitutive {
        ElectricalConstitutiveResponse::Electrostatics {
            displacement_c_per_m2,
            energy_density_j_per_m3,
            total_energy_j,
        } => (
            4.0 * 8.854_187_818_8e-12,
            total_energy_j.canonical_value(),
            energy_density_j_per_m3.canonical_values(),
            displacement_c_per_m2.canonical_values(),
            true,
        ),
        ElectricalConstitutiveResponse::SteadyConduction {
            current_density_a_per_m2,
            joule_density_w_per_m3,
            total_joule_power_w,
        } => (
            3.0,
            total_joule_power_w.canonical_value(),
            joule_density_w_per_m3.canonical_values(),
            current_density_a_per_m2.canonical_values(),
            false,
        ),
    };
    let expected_density = coefficient * voltage * voltage * if electrostatic { 0.5 } else { 1.0 };
    assert_eq!(
        fields[1],
        r.element_electric_field_v_per_m
            .canonical_values()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
    );
    assert_eq!(
        fields[2],
        vectors.iter().flatten().copied().collect::<Vec<_>>()
    );
    assert_eq!(fields[3], *density);
    for vector in fields[1].chunks_exact(3) {
        for (axis, &component) in vector.iter().enumerate() {
            near(
                component,
                if axis == 0 { -voltage } else { 0.0 },
                voltage.abs(),
                "E",
            );
        }
    }
    for vector in fields[2].chunks_exact(3) {
        for (axis, &component) in vector.iter().enumerate() {
            near(
                component,
                if axis == 0 {
                    -coefficient * voltage
                } else {
                    0.0
                },
                coefficient * voltage.abs(),
                "D/J",
            );
        }
    }
    for &value in &fields[3] {
        near(value, expected_density, expected_density, "density");
    }
    near(
        total,
        expected_density,
        expected_density,
        "integral energy/power",
    );
    assert_eq!(r.terminals.len(), 2);
    for terminal in &r.terminals {
        let mut side = None;
        for id in &terminal.ordered_node_ids {
            let index = r.ordered_node_ids.iter().position(|n| n == id).unwrap();
            let x = projection.nodes[index].coordinates_m.canonical_value()[0];
            let is_low = x.abs() < 1e-12;
            assert!(is_low || (x - 1.0).abs() < 1e-12);
            if let Some(previous) = side {
                assert_eq!(previous, is_low);
            }
            side = Some(is_low);
        }
        let expected = coefficient * voltage * if side.unwrap() { 1.0 } else { -1.0 };
        near(
            terminal.outward_flux.canonical_value(),
            expected,
            coefficient * voltage.abs(),
            "signed terminal outward flux",
        );
    }
    assert!(r.linear_relative_residual <= 1e-8);
    assert!(r.relative_flux_imbalance <= 1e-8);
    assert!(r.relative_energy_identity_error <= 1e-8);
    let material_map = runmat_analysis_fea::properties::compile_solid_property_map(
        &completion.workload.model,
        mesh,
    )
    .unwrap();
    material_map.validate_integrity().unwrap();
    for element in &mesh.topology.volume_elements {
        let owner = material_map
            .solid_material(element.stable_identity)
            .unwrap();
        assert_eq!(owner.element_id, element.element_id);
        let material = owner.material.electrical.as_ref().unwrap();
        assert_eq!(material.relative_permittivity.canonical_value(), 4.0);
        assert_eq!(material.conductivity_s_per_m.canonical_value(), 3.0);
        assert_eq!(
            material.resistive_heating_coefficient.canonical_value(),
            0.0
        );
    }
}

fn near(actual: f64, expected: f64, physical_scale: f64, name: &str) {
    assert!(actual.is_finite() && physical_scale > 0.0);
    let relative = (actual - expected).abs() / physical_scale;
    assert!(
        relative <= 1e-7,
        "{name}: actual={actual:e} expected={expected:e} relative={relative:e}"
    );
}
