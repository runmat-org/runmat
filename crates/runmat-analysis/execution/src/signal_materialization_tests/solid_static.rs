use super::*;

#[test]
fn generated_face_force_matches_explicit_tet4_static_balance_and_analytic_shear() {
    use runmat_analysis_fea::{
        excitation::SpatialProjectionBalance,
        fixtures::{fixture_model, FixtureId},
        run_linear_static_with_options, ComputeBackend, LinearStaticSolveOptions,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    };
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
    use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

    let baseline_model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (index, name) in [(0, "root"), (1, "tip")] {
        mesh.topology.boundary_faces[index]
            .provenance
            .push(PersistentEntityId::new(
                PersistentEntityKind::Face,
                name.into(),
                vec!["root".into()],
            ));
        mesh.topology.boundary_faces[index].provenance.sort();
    }
    mesh.seal_canonical_digest().unwrap();

    let generated_signal = materialize_constant_vector_signal(
        &baseline_model.signals[0],
        [0.0, -1000.0, 0.0],
        "solid.face.force",
    );
    let PhysicalSignal::Force(signal) = &generated_signal else {
        panic!("the cantilever fixture must contain one force signal");
    };
    assert!(matches!(signal.provenance, SignalProvenance::Generated(_)));
    let mut explicit_signal = generated_signal.clone();
    let PhysicalSignal::Force(signal) = &mut explicit_signal else {
        unreachable!()
    };
    signal.provenance = SignalProvenance::Authored;

    let mut explicit_model = baseline_model.clone();
    explicit_model.signals = vec![explicit_signal];
    let mut generated_model = baseline_model.clone();
    generated_model.signals = vec![generated_signal];
    let options = LinearStaticSolveOptions {
        solver_mesh: Some(mesh.clone()),
        ..LinearStaticSolveOptions::default()
    };
    let baseline =
        run_linear_static_with_options(&baseline_model, ComputeBackend::Cpu, options.clone())
            .unwrap();
    let explicit =
        run_linear_static_with_options(&explicit_model, ComputeBackend::Cpu, options.clone())
            .unwrap();
    let generated =
        run_linear_static_with_options(&generated_model, ComputeBackend::Cpu, options).unwrap();

    for field_id in [
        FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY,
    ] {
        assert_eq!(
            explicit.field(field_id),
            baseline.field(field_id),
            "explicit {field_id}"
        );
        assert_eq!(
            generated.field(field_id),
            baseline.field(field_id),
            "generated {field_id}"
        );
    }
    assert_eq!(
        explicit.spatial_projection_evidence,
        baseline.spatial_projection_evidence
    );
    assert_eq!(
        generated.spatial_projection_evidence,
        baseline.spatial_projection_evidence
    );
    assert_eq!(
        explicit.structural_static_balance,
        baseline.structural_static_balance
    );
    assert_eq!(
        generated.structural_static_balance,
        baseline.structural_static_balance
    );
    assert!(
        generated
            .structural_static_balance
            .as_ref()
            .unwrap()
            .accepted
    );

    let evidence = &generated.spatial_projection_evidence[0];
    evidence.validate().unwrap();
    assert!(evidence.accepted);
    let SpatialProjectionBalance::VectorResultant {
        projected_resultant,
        projected_moment,
        ..
    } = evidence.balance
    else {
        panic!("face force must preserve a vector resultant");
    };
    let projected_about_origin = [
        projected_moment.real[0] + evidence.reference_point_m[1] * projected_resultant.real[2]
            - evidence.reference_point_m[2] * projected_resultant.real[1],
        projected_moment.real[1] + evidence.reference_point_m[2] * projected_resultant.real[0]
            - evidence.reference_point_m[0] * projected_resultant.real[2],
        projected_moment.real[2] + evidence.reference_point_m[0] * projected_resultant.real[1]
            - evidence.reference_point_m[1] * projected_resultant.real[0],
    ];
    let expected_force = [0.0, -1000.0, 0.0];
    let expected_moment = [1000.0 / 3.0, 0.0, -1000.0 / 3.0];
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= 1.0e-8 * expected.abs().max(1.0),
            "actual={actual} expected={expected}"
        );
    };
    for axis in 0..3 {
        close(projected_resultant.real[axis], expected_force[axis]);
        close(projected_about_origin[axis], expected_moment[axis]);
    }
    let reactions = generated
        .field(FEA_FIELD_STRUCTURAL_REACTION_FORCE)
        .unwrap()
        .as_host_f64()
        .unwrap();
    let mut reaction_force = [0.0; 3];
    let mut reaction_moment = [0.0; 3];
    for (node, force) in mesh.topology.nodes.iter().zip(reactions.chunks_exact(3)) {
        for axis in 0..3 {
            reaction_force[axis] += force[axis];
        }
        let [x, y, z] = node.coordinates_m.canonical_value();
        reaction_moment[0] += y * force[2] - z * force[1];
        reaction_moment[1] += z * force[0] - x * force[2];
        reaction_moment[2] += x * force[1] - y * force[0];
    }
    for axis in 0..3 {
        close(reaction_force[axis], -expected_force[axis]);
        close(reaction_moment[axis], -expected_moment[axis]);
    }
    let shear_modulus = 200.0e9 / (2.0 * (1.0 + 0.3));
    let expected_tip_y = -2000.0 / shear_modulus;
    let expected_energy = 0.5 * (-1000.0 / 3.0) * expected_tip_y;
    close(
        generated
            .field(FEA_FIELD_STRUCTURAL_DISPLACEMENT)
            .unwrap()
            .as_host_f64()
            .unwrap()[10],
        expected_tip_y,
    );
    close(
        generated
            .field(FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
            .unwrap()
            .as_host_f64()
            .unwrap()[0],
        expected_energy,
    );
}
