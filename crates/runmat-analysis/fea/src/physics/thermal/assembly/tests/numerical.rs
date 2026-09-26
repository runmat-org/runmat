use super::*;

#[test]
fn canonical_tet10_thermal_matrices_pass_constant_patch_and_capacity_integral() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let assembly = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::new(Vec::new()),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let conductivity_sum = assembly.conductivity.values().iter().sum::<f64>();
    let conductivity_scale = assembly
        .conductivity
        .values()
        .iter()
        .map(|value| value.abs())
        .sum::<f64>();
    assert!(conductivity_sum.abs() <= conductivity_scale * 1.0e-12);

    let material = &model.materials[0];
    let thermal = material.require_thermal().unwrap();
    let expected_capacity = material.require_mass_density().unwrap()
        * thermal.specific_heat_j_per_kgk
        * assembly.projection.elements[0].signed_volume_m3;
    let integrated_capacity = assembly.capacity.values().iter().sum::<f64>();
    assert!((integrated_capacity - expected_capacity).abs() <= expected_capacity * 1.0e-12);

    let linear_temperature = assembly
        .projection
        .nodes
        .iter()
        .map(|node| {
            2.0 * node.coordinates_m[0] + 3.0 * node.coordinates_m[1] - node.coordinates_m[2]
        })
        .collect::<Vec<_>>();
    let gradient = assembly
        .recover_element_temperature_gradients(&linear_temperature)
        .unwrap()[0];
    assert!((gradient[0] - 2.0).abs() <= 1.0e-12);
    assert!((gradient[1] - 3.0).abs() <= 1.0e-12);
    assert!((gradient[2] + 1.0).abs() <= 1.0e-12);
    let flux = assembly
        .recover_element_heat_flux(&linear_temperature)
        .unwrap()[0];
    let conductivity = thermal.conductivity_w_per_mk;
    assert!((flux[0] + 2.0 * conductivity).abs() <= conductivity * 1.0e-12);
    assert!((flux[1] + 3.0 * conductivity).abs() <= conductivity * 1.0e-12);
    assert!((flux[2] - conductivity).abs() <= conductivity * 1.0e-12);
}

#[test]
fn uniform_heat_generation_is_conserved_and_uniform_temperature_is_preserved() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let density = 1200.0;
    let assembly = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &heat_generation(density),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let expected_source = density * assembly.projection.elements[0].signed_volume_m3;
    let projected_source = assembly.source.iter().sum::<f64>();
    assert!((projected_source - expected_source).abs() <= expected_source * 1.0e-12);
    assert_eq!(assembly.source_evidence.len(), 1);
    assert!(assembly.source_evidence[0].accepted);
    assert_eq!(
        assembly.source_evidence[0].sample,
        crate::excitation::ProjectionSample::Static
    );
    let heated_initial = vec![300.0; assembly.projection.nodes.len()];
    let heated_operator = assembly.transient_operator(0.1, &heated_initial).unwrap();
    let heated = solve_operator_system(
        &heated_operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        Some(&heated_initial),
    );
    assert!(heated.converged);
    assert!(
        assembly
            .energy_balance(&heated_initial, &heated.solution, 0.1)
            .relative_residual
            <= 1.0e-10
    );

    let no_source = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::new(Vec::new()),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let initial = vec![315.0; no_source.projection.nodes.len()];
    let operator = no_source.transient_operator(0.25, &initial).unwrap();
    let solved = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        Some(&initial),
    );
    assert!(solved.converged);
    for temperature in solved.solution {
        assert!((temperature - 315.0).abs() <= 1.0e-9);
    }
}

#[test]
fn exact_face_flux_and_convection_preserve_integrated_power_and_ambient_state() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let face = &projection.boundary_faces[0];
    let selector = face.provenance[0].analysis_selection_id().unwrap();
    model.boundaries = vec![AnalysisBoundary {
        boundary_id: "applied_flux".into(),
        target: RegionTarget::boundary(selector.clone()),
        operator: BoundaryOperator::ThermalHeatFlux {
            value: PrescribedValue::Constant(HeatFlux::new(240.0)),
        },
    }];
    let flux = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::new(Vec::new()),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let expected_power = 240.0 * face.area_m2;
    assert!((flux.source.iter().sum::<f64>() - expected_power).abs() <= expected_power * 1.0e-12);

    model.boundaries = vec![AnalysisBoundary {
        boundary_id: "ambient_convection".into(),
        target: RegionTarget::boundary(selector),
        operator: BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(305.0)),
            coefficient: HeatTransferCoefficient::new(18.0),
        },
    }];
    let convection = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::new(Vec::new()),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let initial = vec![305.0; convection.projection.nodes.len()];
    let operator = convection.transient_operator(0.25, &initial).unwrap();
    let solved = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        Some(&initial),
    );
    assert!(solved.converged);
    assert!(solved
        .solution
        .iter()
        .all(|temperature| (temperature - 305.0).abs() <= 1.0e-9));
    assert!(
        convection
            .energy_balance(&initial, &solved.solution, 0.25)
            .relative_residual
            <= 1.0e-10
    );
}

#[test]
fn prescribed_face_temperature_is_enforced_exactly() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mesh = runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let face = &projection.boundary_faces[0];
    let selector = face.provenance[0].analysis_selection_id().unwrap();
    model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "fixed_temperature".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary(selector),
                value: PrescribedValue::Constant(Temperature::new(290.0)),
            },
        }));
    let assembly = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::new(Vec::new()),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let initial = vec![315.0; assembly.projection.nodes.len()];
    let operator = assembly.transient_operator(0.25, &initial).unwrap();
    let solved = solve_operator_system(
        &operator,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        Some(&initial),
    );
    assert!(solved.converged);
    for &node in &face.node_indices {
        assert!((solved.solution[node] - 290.0).abs() <= 1.0e-12);
    }
    assert!(
        assembly
            .energy_balance(&initial, &solved.solution, 0.25)
            .relative_residual
            <= 1.0e-10
    );
}

#[test]
fn backward_euler_converges_at_first_order_to_the_closed_form_tetrahedron_mode() {
    let (mut model, mesh, free_node, base_region) = thermal_reference_model();
    prescribe_base(&mut model, base_region, 300.0);
    let assembly = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let final_time_s = 0.2;
    let exact = 300.0 + 100.0 * (-10.0_f64 * final_time_s).exp();
    let mut errors = Vec::new();

    for step_count in [4, 8, 16] {
        let time_step_s = final_time_s / step_count as f64;
        let mut temperature = vec![300.0; assembly.projection.nodes.len()];
        temperature[free_node] = 400.0;
        for _ in 0..step_count {
            let operator = assembly
                .transient_operator(time_step_s, &temperature)
                .unwrap();
            let solve = solve_operator_system(
                &operator,
                SpdPreconditionerKind::Jacobi,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                Some(&temperature),
            );
            assert!(solve.converged);
            temperature = solve.solution;
        }
        errors.push((temperature[free_node] - exact).abs());
    }

    assert!(errors.windows(2).all(|pair| pair[1] < 0.56 * pair[0]));
    let observed_orders = errors
        .windows(2)
        .map(|pair| (pair[0] / pair[1]).log2())
        .collect::<Vec<_>>();
    assert!(observed_orders.iter().all(|order| *order > 0.8));
}

#[test]
fn steady_convection_matches_the_closed_form_single_free_node_balance() {
    let (mut model, mut mesh, free_node, base_region) = thermal_reference_model();
    prescribe_base(&mut model, base_region, 350.0);
    let convection_region = PersistentEntityId {
        kind: PersistentEntityKind::Face,
        source_topology_id: "convective_sides".into(),
        assembly_path: vec!["root".into()],
    };
    let base_node_ids = mesh.topology.boundary_faces.iter().find_map(|face| {
        face.provenance
            .iter()
            .any(|id| id.source_topology_id == "thermal_base")
            .then(|| face.node_ids.clone())
    });
    let base_node_ids = base_node_ids.unwrap();
    for face in &mut mesh.topology.boundary_faces {
        if !face
            .node_ids
            .iter()
            .all(|node| base_node_ids.contains(node))
        {
            face.provenance.push(convection_region.clone());
            face.provenance.sort();
        }
    }
    mesh.seal_canonical_digest().unwrap();
    let coefficient = 0.2;
    let ambient_temperature_k = 290.0;
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "side_convection".into(),
        target: RegionTarget::boundary(convection_region.source_topology_id),
        operator: BoundaryOperator::ThermalConvection {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(ambient_temperature_k)),
            coefficient: HeatTransferCoefficient::new(coefficient),
        },
    });

    let assembly = assemble(
        &model,
        &mesh,
        crate::planning::resource::ThermalOperatorResourceFacts::derive(&mesh)
            .unwrap()
            .support(),
        &RealizedExcitations::default(),
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let side_area_m2 = assembly
        .projection
        .boundary_faces
        .iter()
        .filter(|face| {
            face.provenance
                .iter()
                .any(|id| id.source_topology_id == "convective_sides")
        })
        .map(|face| face.area_m2)
        .sum::<f64>();
    let conductivity_term = 1.0 / 6.0;
    let convection_diagonal = coefficient * side_area_m2 / 6.0;
    let expected = (conductivity_term * 350.0
        + convection_diagonal * (2.0 * ambient_temperature_k - 350.0))
        / (conductivity_term + convection_diagonal);
    let solve = solve_operator_system(
        &assembly.steady_operator().unwrap(),
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );

    assert!(solve.converged);
    assert!((solve.solution[free_node] - expected).abs() <= 1.0e-10);
}
