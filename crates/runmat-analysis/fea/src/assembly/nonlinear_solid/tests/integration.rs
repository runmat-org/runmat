use super::*;

#[test]
fn explicit_integration_preserves_affine_physics_and_rejects_foreign_history() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, initial) = fixture(order, false);
        let original = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
        let original_state = original.zero_state();
        let strain = [0.01, -0.002, 0.003, 0.004, -0.006, 0.008];
        let stress = [14.88, 3.36, 8.16, 1.92, -2.88, 3.84];
        let displacement = affine_displacement(&mesh, &initial, strain);
        let expected = constant_stress_force(&mesh, &initial, stress);
        for rule in [
            CanonicalSolidIntegrationRule::Keast15,
            CanonicalSolidIntegrationRule::Duffy5,
            CanonicalSolidIntegrationRule::Duffy8,
            CanonicalSolidIntegrationRule::Duffy12,
            CanonicalSolidIntegrationRule::Duffy16,
        ] {
            let plan = CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
            let basis = CanonicalNonlinearSolidBasis::prepare_with_integration(
                &model, &mesh, &initial, &plan,
            )
            .unwrap();
            assert_eq!(basis.point_count(), plan.total_point_count());
            let state = basis.zero_state();
            assert_eq!(state.integration_identity, plan.identity());
            if rule == CanonicalSolidIntegrationRule::Keast15 {
                assert_eq!(basis.identity(), original.identity());
                assert_eq!(state, original_state);
            } else {
                assert_ne!(basis.identity(), original.identity());
                assert!(basis.validate_state(&original_state).is_err());
                assert!(original.validate_state(&state).is_err());
            }
            let evaluation = basis.evaluate(&displacement, &state).unwrap();
            assert_eq!(evaluation.trial_state, state);
            for (actual, expected) in evaluation.internal_force.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-9, "{order:?}/{rule:?}");
            }
            for point in &evaluation.point_evidence {
                for (actual, expected) in point.stress_pa.iter().zip(stress) {
                    assert!((actual - expected).abs() < 1e-9);
                }
            }
            let mut foreign = state.clone();
            foreign.integration_identity = Digest::sha256(b"foreign-rule");
            assert!(basis.validate_state(&foreign).is_err());
        }
    }
}

#[test]
fn cancelled_original_basis_preparation_preserves_typed_cancellation() {
    let (model, mesh, summary) = fixture(ElementOrder::Tet4, true);
    let context = crate::progress::FeaProgressContext::default();
    context.cancellation.cancel();
    let guard = crate::progress::replace_fea_progress_context(context);
    assert!(matches!(
        CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary),
        Err(NonlinearSolidError::Cancelled)
    ));
    assert!(matches!(
        NonlinearSolidError::from(crate::topology::AnalysisProjectionError::Cancelled),
        NonlinearSolidError::Cancelled
    ));
    drop(guard);
    assert!(CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).is_ok());
}

// witness: FEA-NONLINEAR-OPERATOR-001/references @ small_strain_j2/tet10/numerical_kernel;
//   coverage: partial;
//   boundary: a curved Tet10 integrates its per-point nonlinear state on the converged points of
//   the shared Tet10 element-validity rule (not the requested Keast15), and its elastic tangent
//   equals the stiffness formed from the same rule's moments; plastic response is not covered.
#[test]
fn curved_tet10_nonlinear_assembly_integrates_on_the_shared_rule_points() {
    let (model, _, _) = fixture(ElementOrder::Tet10, false);
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let mut midside = mesh.topology.nodes[4].coordinates_m.canonical_value();
    midside[1] -= 0.05;
    mesh.topology.nodes[4].coordinates_m =
        runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(midside);
    mesh.canonical_digest = runmat_meshing_core::StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let initial = assemble_structural_system_borrowed(
        &model,
        &RealizedExcitations::default(),
        Some(&mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let coordinates: [[f64; 3]; 10] = std::array::from_fn(|index| {
        let id = mesh.topology.volume_elements[0].node_ids[index];
        mesh.topology
            .nodes
            .iter()
            .find(|node| node.node_id == id)
            .unwrap()
            .coordinates_m
            .canonical_value()
    });
    let shared = runmat_meshing_core::tetrahedron10_stiffness_moments(&coordinates).unwrap();
    let expected_rule =
        crate::topology::solid_integration_curved_rule(shared.rule).expect("curved element");
    let plan = CanonicalSolidIntegrationPlan::derive(&mesh, CanonicalSolidIntegrationRule::Keast15)
        .unwrap();
    assert_eq!(plan.elements()[0].rule, expected_rule);
    let plan_points = expected_rule.reference_points().collect::<Vec<_>>();
    assert_eq!(plan_points, shared.quadrature_points());

    let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &initial).unwrap();
    assert_eq!(basis.point_count(), plan_points.len());
    let zero = vec![0.0; basis.dof_count];
    let tangent = basis.evaluate(&zero, &basis.zero_state()).unwrap().tangent;
    let linear = crate::assembly::elements::solid::tetrahedron10_global_stiffness_matrix(
        crate::assembly::elements::solid::SolidMaterial {
            youngs_modulus_pa: runmat_types::physical::Quantity::new(1200.0),
            poisson_ratio: runmat_types::physical::Quantity::new(0.25),
            density_kg_per_m3: runmat_types::physical::Quantity::new(1.0),
        },
        crate::assembly::elements::solid::Tetrahedron10ElementGeometry {
            nodes_m: coordinates,
        },
    )
    .unwrap();
    let dofs = basis.elements[0]
        .dofs
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    for probe in 0..dofs.len() {
        let mut vector = vec![0.0; basis.dof_count];
        vector[dofs[probe]] = 1.0;
        let column = spmv(&tangent, &vector).unwrap();
        for (row, &dof) in dofs.iter().enumerate() {
            let scale = linear.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
            assert!(
                (column[dof] - linear[row][probe]).abs() <= 1e-9 * scale,
                "entry ({row}, {probe}): {} vs {}",
                column[dof],
                linear[row][probe]
            );
        }
    }
}
