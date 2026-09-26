use super::*;
use crate::assembly::elements::solid::SolidMaterial;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn thermal_eigenstrain_recovery_is_stress_free_for_free_expansion_and_compressive_when_fixed() {
    let eigenstrain = 1.0e-4;
    let material = SolidMaterial {
        youngs_modulus_pa: 200.0e9,
        poisson_ratio: 0.3,
        density_kg_per_m3: 7850.0,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let coordinates = mesh
            .topology
            .nodes
            .iter()
            .map(|node| node.coordinates_m)
            .collect::<Vec<_>>();
        let geometry = match order {
            ElementOrder::Tet4 => SolidRecoveryGeometry::Tetrahedron4 {
                node_indices: [0, 1, 2, 3],
                coordinates_m: std::array::from_fn(|index| coordinates[index]),
            },
            ElementOrder::Tet10 => SolidRecoveryGeometry::Tetrahedron10 {
                node_indices: std::array::from_fn(|index| index),
                coordinates_m: std::array::from_fn(|index| coordinates[index]),
            },
        };
        let element = SolidRecoveryElementSummary {
            element_id: "thermal-patch".into(),
            region_id: "body".into(),
            material,
            thermal_eigenstrain: Some(eigenstrain),
            geometry,
        };
        let free_displacement = coordinates
            .iter()
            .flat_map(|point| point.map(|coordinate| eigenstrain * coordinate))
            .collect::<Vec<_>>();
        let free = recover_solid_fields_with_plan(
            std::slice::from_ref(&element),
            &free_displacement,
            LocalExecutionPlan::serial(),
            true,
        )
        .unwrap();
        for component in 0..3 {
            assert!((free.strain[component] - eigenstrain).abs() < 1.0e-12);
        }
        assert!(free.stress.iter().all(|stress| stress.abs() < 2.0e-3));
        assert!(free
            .integration_point_stress
            .iter()
            .all(|stress| stress.abs() < 2.0e-3));
        assert!(free.strain_energy_density[0].abs() < 1.0e-6);

        let fixed = recover_solid_fields_with_plan(
            &[element],
            &vec![0.0; free_displacement.len()],
            LocalExecutionPlan::serial(),
            false,
        )
        .unwrap();
        let expected_stress =
            -material.youngs_modulus_pa * eigenstrain / (1.0 - 2.0 * material.poisson_ratio);
        for normal_stress in &fixed.stress[..3] {
            assert!((normal_stress - expected_stress).abs() < 1.0e-5);
        }
        let expected_energy = -1.5 * eigenstrain * expected_stress;
        assert!((fixed.strain_energy_density[0] - expected_energy).abs() < 1.0e-6);
    }
}
