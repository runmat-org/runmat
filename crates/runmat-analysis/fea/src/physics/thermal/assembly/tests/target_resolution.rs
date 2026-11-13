use super::*;

// witness: FEA-THERMAL-INPUT-001/boundaries @ thermal_steady/tet4|tet10/numerical_kernel; coverage:
//   partial;
//   boundary: Prescribed temperature and heat flux apply to the bound resolved boundary entity, not
//   to the entity named by a deliberately wrong alias, for both element orders; convection and
//   radiation targets are separate cases.
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
        let facts =
            crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[]).unwrap();
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
                flux.prescribed_temperature_k.canonical_values()[node],
                selected.node_indices.contains(&node).then_some(310.0)
            );
        }
        let expected_power = 240.0 * selected.area_m2.canonical_value();
        assert!(
            (flux.applied_source.canonical_values().iter().sum::<f64>() - expected_power).abs()
                <= expected_power * 1.0e-12
        );
        for &node in &other.node_indices {
            if !selected.node_indices.contains(&node) {
                assert_eq!(flux.applied_source.canonical_values()[node], 0.0);
            }
        }
        model.boundaries[0].operator = BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(305.0)),
            coefficient: HeatTransferCoefficient::new(18.0),
        };
        let convection = assemble_model(&model);
        let reference = vec![0.0; projection.nodes.len()];
        let convection_source = convection
            .relative_source(&model, ExcitationSamplePoint::Static, &reference)
            .unwrap();
        let expected_power = 18.0 * 305.0 * selected.area_m2.canonical_value();
        assert!(
            (convection_source.iter().sum::<f64>() - expected_power).abs()
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
        assert_eq!(
            direct
                .relative_source(&model, ExcitationSamplePoint::Static, &reference)
                .unwrap(),
            convection_source
        );
        assert_eq!(
            direct.prescribed_temperature_k.canonical_values(),
            convection.prescribed_temperature_k.canonical_values()
        );
        assert_eq!(direct.conductivity, convection.conductivity);
    }
}

/// A solid region selected by CAD solid name resolves to the exact Solid the elements were
/// meshed from, which elements carry as provenance beside their canonical mesh region. Heat
/// generation on such a target applies Q V over the selected elements.
#[test]
fn heat_generation_on_a_cad_solid_target_resolves_volume_elements_through_provenance() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(order);
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let element = &projection.elements[0];
        let solid = element
            .provenance
            .iter()
            .find(|entity| entity.kind == runmat_meshing_core::PersistentEntityKind::Solid)
            .unwrap()
            .clone();
        assert_ne!(solid, element.region_id);
        let mut target = RegionTarget::volume("cell_1");
        target.bind_resolved_entities(vec![solid]).unwrap();
        let mut excitations = heat_generation(12_383.9);
        let mut realized = excitations.iter().cloned().collect::<Vec<_>>();
        realized[0].target = RealizedExcitationTarget::Region(target);
        excitations = RealizedExcitations::new(realized);
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let facts =
            crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh, &[]).unwrap();
        let assembly = assemble(
            &model,
            &mesh,
            facts.support(),
            &excitations,
            ExcitationSamplePoint::Static,
        )
        .unwrap();
        let expected_power = 12_383.9 * element.signed_volume_m3;
        assert!(
            (assembly
                .applied_source
                .canonical_values()
                .iter()
                .sum::<f64>()
                - expected_power)
                .abs()
                <= expected_power.abs() * 1.0e-12
        );
    }
}
