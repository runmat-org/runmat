use super::*;

// witness: FEA-THERMAL-DOMAIN-001/constraints @ thermal_steady/tet4|tet10/numerical_kernel;
//   boundary: Three components with prescribed/convection anchoring; removal of an anchor rejects
//   coercivity. Does not exercise tied, periodic or affine jumps.
#[test]
fn disconnected_bodies_have_independent_exact_steady_equilibria_and_require_each_anchor() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for mixed_anchors in [false, true] {
            let mut mesh =
                runmat_meshing_core::fixtures::three_region_tetrahedron_solver_mesh_with_order(
                    order,
                );
            let mut model = fixture_model(FixtureId::CantileverLinearStatic);
            model.constraints.clear();
            model.boundaries.clear();
            model.material_assignments[0].target =
                runmat_analysis_core::PropertyRegionTarget::all_solids();
            for (component, temperature_k) in [300.0, 340.0, 380.0].into_iter().enumerate() {
                let name = format!("independent_thermal_body_{component}");
                let face = mesh
                    .topology
                    .boundary_faces
                    .iter_mut()
                    .find(|face| face.face_id == 10 + 100 * component as u64)
                    .unwrap();
                face.provenance.push(PersistentEntityId::new(
                    PersistentEntityKind::Face,
                    name.clone(),
                    vec!["root".into()],
                ));
                face.provenance.sort();
                let target = RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(name.clone())),
                );
                if mixed_anchors && component > 0 {
                    model.boundaries.push(AnalysisBoundary {
                        boundary_id: name,
                        target,
                        operator: BoundaryOperator::ThermalConvection {
                            ambient_temperature: PrescribedValue::Constant(Temperature::new(
                                temperature_k,
                            )),
                            coefficient: HeatTransferCoefficient::new(18.0),
                        },
                    });
                } else {
                    model
                        .constraints
                        .push(AnalysisConstraint::Thermal(ThermalConstraint {
                            constraint_id: name,
                            kind: ThermalConstraintKind::Prescribed {
                                target,
                                value: PrescribedValue::Constant(Temperature::new(temperature_k)),
                            },
                        }));
                }
            }
            mesh.seal_canonical_digest().unwrap();
            let assemble_model = |model: &AnalysisModel| {
                assemble(
                    model,
                    &mesh,
                    crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
                        .unwrap()
                        .support(),
                    &RealizedExcitations::default(),
                    ExcitationSamplePoint::Static,
                )
                .unwrap()
            };
            let assembly = assemble_model(&model);
            let reference = assembly
                .steady_reference_temperatures(&model, ExcitationSamplePoint::Static, 293.15)
                .unwrap();
            for (node, &temperature) in assembly.projection.nodes.iter().zip(&reference) {
                let component = (node.coordinates_m.canonical_value()[0] / 2.0).floor();
                assert_eq!(temperature, 300.0 + 40.0 * component);
            }
            let operator = assembly
                .steady_increment_operator(&model, ExcitationSamplePoint::Static, &reference)
                .unwrap();
            assert!(operator
                .with_canonical_system(|numerical| numerical.nullspace_evidence().cloned())
                .is_some());
            assert!(operator.load().iter().all(|&power| power == 0.0));
            let solve = solve_operator_system(
                &operator,
                SpdPreconditionerKind::Jacobi,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                None,
            );
            assert!(solve.converged);
            assert!(solve.solution.iter().all(|&increment| increment == 0.0));
            let energy = assembly
                .energy_balance(
                    &model,
                    ExcitationSamplePoint::Static,
                    &reference,
                    &reference,
                    1.0,
                    0.0,
                )
                .unwrap();
            assert_eq!(energy.residual_w.canonical_value(), 0.0);
            assert_eq!(energy.relative_residual, 0.0);

            // Reference choice must never turn a freely floating body into an anchored one.
            if mixed_anchors {
                model.boundaries.pop();
            } else {
                model.constraints.pop();
            }
            let unanchored = assemble_model(&model);
            let reference = unanchored
                .steady_reference_temperatures(&model, ExcitationSamplePoint::Static, 293.15)
                .unwrap();
            let operator = unanchored
                .steady_increment_operator(&model, ExcitationSamplePoint::Static, &reference)
                .unwrap();
            assert!(operator
                .with_canonical_system(|numerical| numerical.nullspace_evidence().cloned())
                .is_none());
            match operator.prepare_linear() {
                Ok(prepared) => assert!(crate::algebra::validate_prepared_solver_compatibility(
                    &prepared,
                    crate::algebra::RealSolver::ConjugateGradient,
                    crate::algebra::RealPreconditioner::Jacobi,
                )
                .is_err()),
                Err(error) => assert!(matches!(
                    error,
                    crate::algebra::RealSystemError::Coercivity(_)
                )),
            }
        }
    }
}

#[test]
fn component_reference_preserves_dirichlet_contrasts_within_a_connected_body() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let mut assembly = assemble(
            &model,
            &mesh,
            crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[])
                .unwrap()
                .support(),
            &RealizedExcitations::default(),
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        assembly
            .prescribed_temperature_k
            .try_replace(0, runmat_types::physical::Quantity::new(Some(300.0)))
            .unwrap();
        assembly
            .prescribed_temperature_k
            .try_replace(1, runmat_types::physical::Quantity::new(Some(400.0)))
            .unwrap();
        let reference = assembly
            .steady_reference_temperatures(&model, ExcitationSamplePoint::Static, 293.15)
            .unwrap();
        assert!(reference.iter().all(|&value| value == 300.0));
        let operator = assembly
            .steady_increment_operator(&model, ExcitationSamplePoint::Static, &reference)
            .unwrap();
        assert_eq!(operator.load()[0], 0.0);
        assert_eq!(operator.load()[1], 100.0);
        assert!(operator
            .with_canonical_system(|numerical| numerical.nullspace_evidence().cloned())
            .is_some());
        let mut invalid_reference = reference;
        invalid_reference[1] = 400.0;
        assert!(assembly
            .steady_increment_operator(&model, ExcitationSamplePoint::Static, &invalid_reference,)
            .is_err());
    }
}
