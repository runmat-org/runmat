//! Independent unit-prism Poisson solution, checked after actual native cold import.
pub(super) fn verify(
    completion: &runmat_runtime::analysis::ElectricalStudyCompletion,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    gauge: f64,
    sign: f64,
    fields: &[Vec<f64>; 4],
) {
    let epsilon = 4.0 * 8.854_187_818_8e-12;
    let projection = runmat_analysis_fea::topology::AnalysisProjection::derive(mesh).unwrap();
    let response = completion.result.response();
    assert_eq!(&fields[0], response.potential_v.canonical_values());
    for (actual, node) in fields[0].iter().zip(&projection.nodes) {
        let x = node.coordinates_m.canonical_value()[0];
        near(*actual, gauge + sign * x * (1.0 - x), 1.0);
    }
    for (field, element) in fields[1].chunks_exact(3).zip(&projection.elements) {
        near(field[0], sign * (2.0 * element.centroid_m[0] - 1.0), 1.0);
        near(field[1], 0.0, 1.0);
        near(field[2], 0.0, 1.0);
    }
    for (electric, displacement) in fields[1].iter().zip(&fields[2]) {
        near(*displacement, epsilon * electric, epsilon);
    }
    let runmat_analysis_fea::physics::electrical::ElectricalConstitutiveResponse::Electrostatics {
        total_energy_j,
        energy_density_j_per_m3,
        ..
    } = &response.constitutive
    else {
        panic!("charged electrostatic constitutive response")
    };
    assert_eq!(&fields[3], energy_density_j_per_m3.canonical_values());
    near(total_energy_j.canonical_value(), epsilon / 6.0, epsilon);
    let source = response.charge_sources.as_ref().unwrap();
    near(source.integrated_source, sign * 2.0 * epsilon, epsilon);
    assert!(source.projection_evidence.iter().all(|e| e.accepted));
    for terminal in &response.terminals {
        near(
            terminal.outward_flux.canonical_value(),
            sign * epsilon,
            epsilon,
        );
    }
    near(
        response.volume_source_work.canonical_value(),
        epsilon / 3.0,
        epsilon,
    );
    near(response.terminal_work.canonical_value(), 0.0, epsilon);
    assert!(response.linear_relative_residual <= 1e-8);
    assert!(response.relative_flux_imbalance <= 1e-8);
    assert!(response.relative_energy_identity_error <= 1e-8);
}

fn near(actual: f64, expected: f64, physical_scale: f64) {
    assert!(
        (actual - expected).abs() / physical_scale <= 1e-7,
        "actual={actual:e} expected={expected:e} scale={physical_scale:e}"
    );
}
