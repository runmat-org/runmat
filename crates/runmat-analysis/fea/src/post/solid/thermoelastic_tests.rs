use super::*;
use crate::assembly::elements::solid::SolidMaterial;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

#[test]
fn solved_linear_temperature_recovers_pointwise_stress_and_integrated_energy() {
    use crate::{
        assembly::solid_recovery,
        fixtures::{fixture_model, FixtureId},
        properties::compile_structural_property_map,
    };
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let alpha = 1.0e-5;
        let thermal = model.materials[0].thermal.as_mut().unwrap();
        thermal.expansion_coefficient_per_k = alpha;
        let reference = thermal.reference_temperature_k;
        let mechanical = model.materials[0].require_mechanical().unwrap();
        let coefficient = mechanical.youngs_modulus_pa / (1.0 - 2.0 * mechanical.poisson_ratio);
        let properties = compile_structural_property_map(&model, Some(&mesh)).unwrap();
        let temperatures = mesh
            .topology
            .nodes
            .iter()
            .map(|node| reference + 10.0 + 20.0 * node.coordinates_m[0])
            .collect::<Vec<_>>();
        let retained = temperatures.clone();
        let elements =
            solid_recovery::build(&mesh, &properties, None, Some(&temperatures)).unwrap();
        assert_eq!(temperatures, retained);
        let fixed = recover_solid_fields_with_plan(
            &elements,
            &vec![0.0; temperatures.len() * 3],
            LocalExecutionPlan::serial(),
            true,
        )
        .unwrap();
        // Unit tetrahedron: <x> = 1/4 and <x²> = 1/10, hence <(10+20x)²> = 240.
        let expected_mean_stress = -coefficient * alpha * 15.0;
        for stress in &fixed.stress[..3] {
            assert!((stress / expected_mean_stress - 1.0).abs() < 1.0e-12);
        }
        let expected_energy = 1.5 * coefficient * alpha * alpha * 240.0;
        assert!((fixed.strain_energy_density[0] / expected_energy - 1.0).abs() < 1.0e-12);
        let low = (5.0 - 5.0_f64.sqrt()) / 20.0;
        let high = (5.0 + 3.0 * 5.0_f64.sqrt()) / 20.0;
        for (point, x) in [low, high, low, low].into_iter().enumerate() {
            let expected = -coefficient * alpha * (10.0 + 20.0 * x);
            for stress in &fixed.integration_point_stress[point * 6..point * 6 + 3] {
                assert!((stress / expected - 1.0).abs() < 1.0e-12);
            }
        }
        if order == ElementOrder::Tet10 {
            // Compatible quadratic displacement with ε = α(10+20x)I exactly.
            let displacement = mesh
                .topology
                .nodes
                .iter()
                .flat_map(|node| {
                    let [x, y, z] = node.coordinates_m;
                    [
                        alpha * (10.0 * x + 10.0 * x * x - 10.0 * y * y - 10.0 * z * z),
                        alpha * (10.0 * y + 20.0 * x * y),
                        alpha * (10.0 * z + 20.0 * x * z),
                    ]
                })
                .collect::<Vec<_>>();
            let free = recover_solid_fields_with_plan(
                &elements,
                &displacement,
                LocalExecutionPlan::serial(),
                true,
            )
            .unwrap();
            assert!(free
                .integration_point_stress
                .iter()
                .all(|value| value.abs() < 1.0e-5));
            assert!(free.strain_energy_density[0].abs() < 1.0e-16);
        }
        assert!(solid_recovery::build(&mesh, &properties, None, Some(&temperatures[..1])).is_err());
        let mut invalid = temperatures.clone();
        invalid[0] = f64::NAN;
        assert!(solid_recovery::build(&mesh, &properties, None, Some(&invalid)).is_err());
    }
}

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
            prescribed_temperature_k: Some(303.15),
            solved_thermal_quadrature: None,
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
