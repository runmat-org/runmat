use super::*;
use runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh_with_order;

#[test]
fn independent_bodies_use_exact_yield_hardening_and_elasticity_owners_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = three_region_tetrahedron_solver_mesh_with_order(order);
        let mut model = fixture_model(FixtureId::MultiMaterialAssembly);
        replace_fixture_excitations(&mut model, Vec::new());
        model.constraints.clear();
        for (i, material) in model.materials.iter_mut().enumerate() {
            material.elasticity = Some(MaterialElasticityModel::IsotropicLinear(
                MaterialMechanicalModel {
                    youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(
                        [1200.0, 2400.0, 4800.0][i],
                    ),
                    poisson_ratio: runmat_types::physical::PoissonRatio::new(0.0),
                },
            ));
            material.plastic = Some(MaterialPlasticModel::J2IsotropicLinearHardening {
                yield_stress_pa: runmat_types::physical::YieldStress::new([5.0, 9.0, 15.0][i]),
                isotropic_hardening_modulus_pa: runmat_types::physical::HardeningModulus::new(
                    [30.0, 60.0, 90.0][i],
                ),
            });
        }
        for assignment in &mut model.material_assignments {
            let region = assignment.selected_region_id().unwrap();
            let fixture = runmat_meshing_core::fixtures::fixture_region;
            let id = if region == fixture("tip_steel") {
                "mat_steel"
            } else if region == fixture("mid_aluminum") {
                "mat_aluminum"
            } else if region == fixture("polymer_segment") {
                "mat_polymer"
            } else {
                panic!("unexpected region {region}")
            };
            assignment.assigned_material_id = id.into();
        }
        let initial = assemble_structural_system_borrowed(
            &model,
            &RealizedExcitations::default(),
            Some(&mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let gamma = 0.04;
        let u = affine_displacement(&mesh, &initial, [0.0, 0.0, 0.0, gamma, 0.0, 0.0]);
        let evaluated = basis.evaluate(&u, &basis.zero_state()).unwrap();
        for evidence in &evaluated.element_evidence {
            let element = mesh
                .topology
                .volume_elements
                .iter()
                .find(|e| e.element_id == evidence.element_id)
                .unwrap();
            let (young, yield_stress, hardening) =
                match element.region_id.source_topology_id.as_str() {
                    "tip_steel" => (1200.0, 5.0, 30.0),
                    "mid_aluminum" => (2400.0, 9.0, 60.0),
                    "polymer_segment" => (4800.0, 15.0, 90.0),
                    other => panic!("unexpected region {other}"),
                };
            // Independent pure-shear radial-return scalar equation, not kernel output.
            let mu = young / 2.0;
            let trial_shear = mu * gamma;
            let q_trial = 3.0_f64.sqrt() * trial_shear;
            let delta_alpha = (q_trial - yield_stress) / (3.0 * mu + hardening);
            let exact_shear = trial_shear * (1.0 - 3.0 * mu * delta_alpha / q_trial);
            for (i, stress) in evidence.mean_stress_pa.iter().enumerate() {
                let expected = if i == 3 { exact_shear } else { 0.0 };
                assert!(
                    (stress - expected).abs() < 1e-9,
                    "order={order:?}, region={}, stress={:?}, expected_shear={exact_shear}",
                    element.region_id.source_topology_id,
                    evidence.mean_stress_pa
                );
            }
            let histories = evaluated
                .trial_state
                .point_identities
                .iter()
                .zip(&evaluated.trial_state.points)
                .filter(|(id, _)| id.element_identity == element.stable_identity);
            for (_, history) in histories {
                assert!((history.equivalent_plastic_strain - delta_alpha).abs() < 1e-12);
            }
            assert!(
                (evidence.hardening_energy_j.canonical_value()
                    - 0.5 * hardening * delta_alpha.powi(2) * evidence.volume_m3)
                    .abs()
                    < 1e-12
            );
            assert!(
                (evidence.plastic_dissipation_increment_j
                    - yield_stress * delta_alpha * evidence.volume_m3)
                    .abs()
                    < 1e-12
            );
        }
        // Material display order cannot become a global constitutive owner.
        let mut reordered = model.clone();
        reordered.materials.reverse();
        reordered.material_assignments.reverse();
        let reordered_basis =
            CanonicalNonlinearSolidBasis::prepare(&reordered, &mesh, &initial).unwrap();
        let reordered_result = reordered_basis
            .evaluate(&u, &reordered_basis.zero_state())
            .unwrap();
        assert_eq!(reordered_result.internal_force, evaluated.internal_force);
        assert_eq!(reordered_result.trial_state, evaluated.trial_state);
        let mut missing = model;
        missing.material_assignments.pop();
        assert!(CanonicalNonlinearSolidBasis::prepare(&missing, &mesh, &initial).is_err());
    }
}
