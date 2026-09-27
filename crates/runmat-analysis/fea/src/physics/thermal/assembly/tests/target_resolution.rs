use super::*;

#[test]
fn thermal_boundaries_use_resolved_entities_instead_of_authored_aliases() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let selected = &projection.boundary_faces[0];
        let other = &projection.boundary_faces[1];
        // Deliberately make the alias identify another face. Bound identities must
        // win; falling back to the alias would apply data to the wrong boundary.
        let mut target =
            RegionTarget::boundary(other.provenance[0].analysis_selection_id().unwrap());
        target
            .bind_resolved_entities(vec![selected.provenance[0].clone()])
            .unwrap();
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        model.constraints = vec![AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "authored_hot".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: target.clone(),
                value: PrescribedValue::Constant(Temperature::new(310.0)),
            },
        })];
        model.boundaries = vec![AnalysisBoundary {
            boundary_id: "authored_flux".into(),
            target: target.clone(),
            operator: BoundaryOperator::ThermalHeatFlux {
                value: PrescribedValue::Constant(HeatFlux::new(240.0)),
            },
        }];
        let facts = crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh).unwrap();
        let assemble_model = |model: &AnalysisModel| {
            assemble(
                model,
                &mesh,
                facts.support(),
                &RealizedExcitations::default(),
                ExcitationSamplePoint::Static,
            )
            .unwrap()
        };
        let flux = assemble_model(&model);
        for node in 0..projection.nodes.len() {
            assert_eq!(
                flux.prescribed_temperature_k[node],
                selected.node_indices.contains(&node).then_some(310.0)
            );
        }
        let expected_power = 240.0 * selected.area_m2;
        assert!(
            (flux.source.iter().sum::<f64>() - expected_power).abs() <= expected_power * 1.0e-12
        );
        for &node in &other.node_indices {
            if !selected.node_indices.contains(&node) {
                assert_eq!(flux.source[node], 0.0);
            }
        }
        model.boundaries[0].operator = BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(305.0)),
            coefficient: HeatTransferCoefficient::new(18.0),
        };
        let convection = assemble_model(&model);
        let expected_power = 18.0 * 305.0 * selected.area_m2;
        assert!(
            (convection.source.iter().sum::<f64>() - expected_power).abs()
                <= expected_power * 1.0e-12
        );

        // With no resolved inventory, the normal direct-selector path remains exact.
        let direct =
            RegionTarget::boundary(selected.provenance[0].analysis_selection_id().unwrap());
        model.boundaries[0].target = direct.clone();
        let AnalysisConstraint::Thermal(constraint) = &mut model.constraints[0] else {
            unreachable!();
        };
        let ThermalConstraintKind::Prescribed { target, .. } = &mut constraint.kind else {
            unreachable!();
        };
        *target = direct;
        let direct = assemble_model(&model);
        assert_eq!(direct.source, convection.source);
        assert_eq!(
            direct.prescribed_temperature_k,
            convection.prescribed_temperature_k
        );
        assert_eq!(direct.conductivity, convection.conductivity);
    }
}
