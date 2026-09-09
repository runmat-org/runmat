use super::*;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn tet4_and_tet10_mass_preserve_physical_total_and_rigid_translation() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let artifact = canonical_tetrahedron_solver_mesh(order);
        let material = material(2_700.0);
        let assembled =
            assemble_solver_solid_mass(&artifact, Some(material), &BTreeMap::new(), 3).unwrap();
        let expected_mass = material.density_kg_per_m3 / 6.0;
        assert_relative(assembled.total_mass_kg, expected_mass, 2.0e-12);
        assert!(assembled.lumped_diag.iter().all(|value| *value > 0.0));
        assert_relative(
            assembled.lumped_diag.iter().step_by(3).sum(),
            expected_mass,
            2.0e-12,
        );

        let mut rigid_x = vec![0.0; assembled.lumped_diag.len()];
        for value in rigid_x.iter_mut().step_by(3) {
            *value = 1.0;
        }
        let applied = apply_csr(&assembled.consistent, &rigid_x);
        let generalized_mass = rigid_x
            .iter()
            .zip(applied)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        assert_relative(generalized_mass, expected_mass, 2.0e-12);
    }
}

#[test]
fn region_density_controls_mass_without_changing_topology() {
    let artifact = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let region = artifact.topology.volume_elements[0]
        .region_id
        .source_topology_id
        .clone();
    let baseline =
        assemble_solver_solid_mass(&artifact, Some(material(1_000.0)), &BTreeMap::new(), 3)
            .unwrap();
    let selected = assemble_solver_solid_mass(
        &artifact,
        Some(material(1_000.0)),
        &BTreeMap::from([(region, material(2_500.0))]),
        3,
    )
    .unwrap();
    assert_relative(
        selected.total_mass_kg / baseline.total_mass_kg,
        2.5,
        1.0e-12,
    );
    assert_eq!(
        baseline.consistent.column_indices,
        selected.consistent.column_indices
    );
}

fn material(density_kg_per_m3: f64) -> SolidMaterial {
    SolidMaterial {
        youngs_modulus_pa: 70.0e9,
        poisson_ratio: 0.3,
        density_kg_per_m3,
    }
}

fn apply_csr(matrix: &CsrMatrix, input: &[f64]) -> Vec<f64> {
    (0..input.len())
        .map(|row| {
            (matrix.row_offsets[row]..matrix.row_offsets[row + 1])
                .map(|entry| matrix.values[entry] * input[matrix.column_indices[entry]])
                .sum()
        })
        .collect()
}

fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    let relative = (actual - expected).abs() / expected.abs().max(1.0);
    assert!(
        relative <= tolerance,
        "actual={actual} expected={expected} relative={relative} tolerance={tolerance}"
    );
}
