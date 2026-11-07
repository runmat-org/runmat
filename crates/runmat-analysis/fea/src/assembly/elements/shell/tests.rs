use super::*;
use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};
use std::sync::Arc;

#[test]
fn admitted_shell_frame_and_homogeneous_constitutive_conserve_work() {
    let limit = OperationCount::from_operations(56 + 55);
    let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    context.run_sync(|| {
        geometry().local_frame().unwrap();
        admitted_homogeneous_constitutive(section(), material()).unwrap();
    });
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.completed.assembly, limit);
    assert_eq!(snapshot.reserved, OperationCount::ZERO);
}

#[test]
fn admitted_shell_stiffness_conserves_local_mitc_and_contraction_work() {
    let limit = OperationCount::from_operations(55 + 56 + 68 + 4 * 597 + 74_491);
    let ledger = Arc::new(UsageLedger::new(limit, limit).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    context.run_sync(|| {
        global_stiffness_matrix(section(), material(), geometry()).unwrap();
    });
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(snapshot.completed.assembly, limit);
    assert_eq!(snapshot.reserved, OperationCount::ZERO);
}

fn section() -> ShellSection {
    ShellSection {
        thickness_m: runmat_types::physical::Length::new(0.002),
        reference_offset_m: runmat_types::physical::Length::new(0.0),
        shear_correction: 5.0 / 6.0,
    }
}

fn material() -> ShellMaterial {
    ShellMaterial {
        youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(200.0e9),
        poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
        shear_modulus_pa: runmat_types::physical::ShearModulus::new(76.9e9),
    }
}

fn geometry() -> ShellElementGeometry {
    ShellElementGeometry {
        nodes_m: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        reference_axis: [1.0, 0.0, 0.0],
    }
}

#[test]
fn shell_local_frame_is_orthonormal() {
    let frame = geometry().local_frame().expect("frame should build");
    assert_close(frame.area_m2.canonical_value(), 0.5, 1.0e-12);
    assert_close(dot(frame.x, frame.y), 0.0, 1.0e-12);
    assert_close(dot(frame.x, frame.z), 0.0, 1.0e-12);
    assert_close(dot(frame.y, frame.z), 0.0, 1.0e-12);
    assert_close(norm(frame.x), 1.0, 1.0e-12);
    assert_close(norm(frame.y), 1.0, 1.0e-12);
    assert_close(norm(frame.z), 1.0, 1.0e-12);
}

#[test]
fn shell_global_stiffness_is_symmetric() {
    let stiffness = global_stiffness_matrix(section(), material(), geometry()).unwrap();
    for row in 0..SHELL_ELEMENT_DOF_COUNT {
        for column in 0..SHELL_ELEMENT_DOF_COUNT {
            assert_close(stiffness[row][column], stiffness[column][row], 1.0e-6);
        }
    }
}

#[test]
fn shell_has_exact_six_rigid_body_modes() {
    let stiffness = global_stiffness_matrix(section(), material(), geometry()).unwrap();
    for mode_index in 0..6 {
        let mut mode = [0.0; SHELL_ELEMENT_DOF_COUNT];
        for node in 0..SHELL_ELEMENT_NODE_COUNT {
            let base = node * SHELL_NODE_DOF_COUNT;
            if mode_index < 3 {
                mode[base + mode_index] = 1.0;
            } else {
                let mut rotation = [0.0; 3];
                rotation[mode_index - 3] = 1.0;
                let translation = cross(rotation, geometry().nodes_m[node]);
                mode[base..base + 3].copy_from_slice(&translation);
                mode[base + 3..base + 6].copy_from_slice(&rotation);
            }
        }
        for row in &stiffness {
            let residual = row.iter().zip(mode).map(|(value, mode)| value * mode).sum();
            assert_close(residual, 0.0, 1.0e-5);
        }
    }
}

#[test]
fn shell_constant_membrane_strain_matches_plane_stress_resultants() {
    let geometry = geometry();
    let epsilon_x = 2.0e-4;
    let mut displacement = [0.0; SHELL_ELEMENT_DOF_COUNT];
    for (node, position) in geometry.nodes_m.iter().enumerate() {
        displacement[node * SHELL_NODE_DOF_COUNT] = epsilon_x * position[0];
    }
    let result = recover_resultants(section(), material(), geometry, displacement).unwrap();
    let factor = material().youngs_modulus_pa.canonical_value()
        * section().thickness_m.canonical_value()
        / (1.0 - material().poisson_ratio.canonical_value().powi(2));
    assert_close(
        result.membrane_force_n_per_m[0],
        factor * epsilon_x,
        1.0e-8 * factor,
    );
    assert_close(
        result.membrane_force_n_per_m[1],
        factor * material().poisson_ratio.canonical_value() * epsilon_x,
        1.0e-8 * factor,
    );
    assert_close(result.membrane_force_n_per_m[2], 0.0, 1.0e-8 * factor);
    assert!(result.bending_moment_n.iter().all(|value| *value == 0.0));
    assert!(result
        .transverse_shear_n_per_m
        .iter()
        .all(|value| *value == 0.0));
}

#[test]
fn homogeneous_reference_surface_retains_membrane_bending_coupling() {
    let mut offset_section = section();
    offset_section.reference_offset_m =
        runmat_types::physical::Length::new(0.5 * offset_section.thickness_m.canonical_value());
    let constitutive = admitted_homogeneous_constitutive(offset_section, material()).unwrap();
    let extensional = constitutive.matrices.extensional[0][0];
    assert_close(
        constitutive.matrices.coupling[0][0],
        -0.5 * offset_section.thickness_m.canonical_value() * extensional,
        1.0e-12 * extensional,
    );
    assert_close(
        constitutive.matrices.bending[0][0],
        offset_section.thickness_m.canonical_value().powi(2) * extensional / 3.0,
        1.0e-12 * extensional,
    );
}

#[test]
fn shell_energy_is_invariant_under_rigid_frame_rotation() {
    let original_geometry = geometry();
    let rotated_geometry = ShellElementGeometry {
        nodes_m: original_geometry.nodes_m.map(cyclic_rotation),
        reference_axis: cyclic_rotation(original_geometry.reference_axis),
    };
    let original_displacement = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0e-4, -1.0e-4, 4.0e-5, 3.0e-4, -2.0e-4, 1.0e-4, -1.0e-4,
        3.0e-4, -2.0e-5, -4.0e-4, 1.0e-4, 2.0e-4,
    ];
    let mut rotated_displacement = [0.0; SHELL_ELEMENT_DOF_COUNT];
    for node in 0..SHELL_ELEMENT_NODE_COUNT {
        let base = node * SHELL_NODE_DOF_COUNT;
        for block in 0..2 {
            let offset = base + block * 3;
            rotated_displacement[offset..offset + 3].copy_from_slice(&cyclic_rotation([
                original_displacement[offset],
                original_displacement[offset + 1],
                original_displacement[offset + 2],
            ]));
        }
    }
    let original_stiffness =
        global_stiffness_matrix(section(), material(), original_geometry).unwrap();
    let rotated_stiffness =
        global_stiffness_matrix(section(), material(), rotated_geometry).unwrap();
    assert_close(
        quadratic_energy(&original_stiffness, &original_displacement),
        quadratic_energy(&rotated_stiffness, &rotated_displacement),
        1.0e-9,
    );
    assert_eq!(
        recover_resultants(
            section(),
            material(),
            original_geometry,
            original_displacement
        )
        .unwrap(),
        recover_resultants(
            section(),
            material(),
            rotated_geometry,
            rotated_displacement
        )
        .unwrap()
    );
}

#[test]
fn shell_energy_is_invariant_to_cyclic_and_reversed_node_order() {
    let original_geometry = ShellElementGeometry {
        nodes_m: [[0.0, 0.0, 0.0], [1.4, 0.2, 0.0], [0.1, 0.9, 0.0]],
        reference_axis: [1.0, 0.0, 0.0],
    };
    let original_displacement = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0e-4, -1.0e-4, 4.0e-5, 3.0e-4, -2.0e-4, 1.0e-4, -1.0e-4,
        3.0e-4, -2.0e-5, -4.0e-4, 1.0e-4, 2.0e-4,
    ];
    let original_energy = quadratic_energy(
        &global_stiffness_matrix(section(), material(), original_geometry).unwrap(),
        &original_displacement,
    );
    for permutation in [[1, 2, 0], [0, 2, 1]] {
        let permuted_geometry = ShellElementGeometry {
            nodes_m: permutation.map(|node| original_geometry.nodes_m[node]),
            ..original_geometry
        };
        let mut permuted_displacement = [0.0; SHELL_ELEMENT_DOF_COUNT];
        for (permuted_node, original_node) in permutation.into_iter().enumerate() {
            let target = permuted_node * SHELL_NODE_DOF_COUNT;
            let source = original_node * SHELL_NODE_DOF_COUNT;
            permuted_displacement[target..target + SHELL_NODE_DOF_COUNT]
                .copy_from_slice(&original_displacement[source..source + SHELL_NODE_DOF_COUNT]);
        }
        let permuted_energy = quadratic_energy(
            &global_stiffness_matrix(section(), material(), permuted_geometry).unwrap(),
            &permuted_displacement,
        );
        assert_close(permuted_energy / original_energy, 1.0, 1.0e-12);
    }
}

// witness: FEA-STRUCTURAL-SHELL-001/physical_shell_operator @
//   linear_structural/shell/numerical_kernel; coverage: partial;
//   boundary: Shell resultant thickness scaling separates membrane, bending and shear.
#[test]
fn shell_resultant_thickness_scaling_separates_membrane_bending_and_shear() {
    let mut displacement = [0.0; SHELL_ELEMENT_DOF_COUNT];
    for (node, position) in geometry().nodes_m.iter().enumerate() {
        let base = node * SHELL_NODE_DOF_COUNT;
        displacement[base] = 2.0e-4 * position[0];
        displacement[base + 4] = 3.0e-4 * position[0];
    }
    let thin = recover_resultants(section(), material(), geometry(), displacement).unwrap();
    let thick = recover_resultants(
        ShellSection {
            thickness_m: runmat_types::physical::Length::new(
                section().thickness_m.canonical_value() * 2.0,
            ),
            ..section()
        },
        material(),
        geometry(),
        displacement,
    )
    .unwrap();
    for component in 0..3 {
        if thin.membrane_force_n_per_m[component].abs() > 0.0 {
            assert_close(
                thick.membrane_force_n_per_m[component] / thin.membrane_force_n_per_m[component],
                2.0,
                1.0e-12,
            );
        }
        if thin.bending_moment_n[component].abs() > 0.0 {
            assert_close(
                thick.bending_moment_n[component] / thin.bending_moment_n[component],
                8.0,
                1.0e-12,
            );
        }
    }
    for component in 0..2 {
        if thin.transverse_shear_n_per_m[component].abs() > 0.0 {
            assert_close(
                thick.transverse_shear_n_per_m[component]
                    / thin.transverse_shear_n_per_m[component],
                2.0,
                1.0e-12,
            );
        }
    }
}

#[test]
fn assumed_shear_field_preserves_pure_bending_in_the_thin_limit() {
    let curvature = 3.0e-4;
    let mut displacement = [0.0; SHELL_ELEMENT_DOF_COUNT];
    for (node, position) in geometry().nodes_m.iter().enumerate() {
        let base = node * SHELL_NODE_DOF_COUNT;
        displacement[base + 2] = -0.5 * curvature * position[0].powi(2);
        displacement[base + 4] = curvature * position[0];
    }
    let thin_section = section();
    let thick_section = ShellSection {
        thickness_m: runmat_types::physical::Length::new(
            thin_section.thickness_m.canonical_value() * 2.0,
        ),
        ..thin_section
    };
    let result = recover_resultants(thin_section, material(), geometry(), displacement).unwrap();
    let bending_factor = material().youngs_modulus_pa.canonical_value()
        * thin_section.thickness_m.canonical_value().powi(3)
        / (12.0 * (1.0 - material().poisson_ratio.canonical_value().powi(2)));
    assert_close(
        result.bending_moment_n[0],
        bending_factor * curvature,
        1.0e-12,
    );
    assert!(result
        .transverse_shear_n_per_m
        .iter()
        .all(|value| value.abs() < 1.0e-12));
    let thin_stiffness = global_stiffness_matrix(thin_section, material(), geometry()).unwrap();
    let thick_stiffness = global_stiffness_matrix(thick_section, material(), geometry()).unwrap();
    assert_close(
        quadratic_energy(&thick_stiffness, &displacement)
            / quadratic_energy(&thin_stiffness, &displacement),
        8.0,
        1.0e-9,
    );
}

#[test]
fn shell_frame_rejects_degenerate_inputs() {
    assert_eq!(
        ShellElementGeometry {
            nodes_m: [[0.0, 0.0, 0.0]; 3],
            reference_axis: [1.0, 0.0, 0.0],
        }
        .local_frame()
        .expect_err("zero-area shell should fail"),
        ShellElementError::DegenerateArea
    );
}

fn cyclic_rotation(vector: [f64; 3]) -> [f64; 3] {
    [vector[2], vector[0], vector[1]]
}

fn quadratic_energy(
    stiffness: &ShellMatrix18,
    displacement: &[f64; SHELL_ELEMENT_DOF_COUNT],
) -> f64 {
    0.5 * stiffness
        .iter()
        .zip(displacement)
        .map(|(row, value)| {
            value
                * row
                    .iter()
                    .zip(displacement)
                    .map(|(coefficient, displacement)| coefficient * displacement)
                    .sum::<f64>()
        })
        .sum::<f64>()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} tolerance={tolerance}",
    );
}
