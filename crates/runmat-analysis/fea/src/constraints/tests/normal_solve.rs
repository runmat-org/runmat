use super::*;

#[test]
fn face_normal_offset_solves_with_analytic_compression_and_reaction() {
    for angle in [0.0, std::f64::consts::FRAC_PI_4] {
        solve_face_normal_offset(angle);
    }
}

fn solve_face_normal_offset(angle: f64) {
    // Face 1 is y=0 with outward normal -ey. The prescribed normal offset
    // -d on nodes 1,2,4 and fixed node 3 produce u=(0,d*(1-y),0). The only
    // free DOF is node 4 local x, which has zero force. The second case
    // rotates both the geometry and tangent supports so C has mixed y/z rows.
    let (mut model, mut mesh) = model_and_mesh();
    let (sin, cos) = angle.sin_cos();
    for node in &mut mesh.topology.nodes {
        let [x, y, z] = node.coordinates_m.canonical_value();
        node.coordinates_m = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new([x, cos * y - sin * z, sin * y + cos * z]);
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let tangent_frame = CoordinateFrame::Cartesian(runmat_analysis_core::CartesianFrame {
        frame_id: "support_axes".into(),
        origin: Position::new([0.0; 3]),
        x_axis: [1.0, 0.0, 0.0],
        y_axis: [0.0, cos, sin],
    });
    let offset_m = 0.001;
    model.excitations.clear();
    model.constraints = vec![
        framed_zero(
            "first_tangents",
            1,
            &[StructuralTranslationDof::Ux, StructuralTranslationDof::Uz],
            tangent_frame.clone(),
        ),
        framed_zero(
            "second_tangents",
            2,
            &[StructuralTranslationDof::Ux, StructuralTranslationDof::Uz],
            tangent_frame.clone(),
        ),
        framed_zero(
            "opposite_anchor",
            3,
            &[
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ],
            tangent_frame.clone(),
        ),
        framed_zero(
            "fourth_tangent",
            4,
            &[StructuralTranslationDof::Uz],
            tangent_frame,
        ),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "face_normal_offset".into(),
            frame: CoordinateFrame::BoundaryNormalTangent,
            kind: StructuralConstraintKind::NormalDisplacement {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:1",
                )),
                value: PrescribedValue::Constant(Displacement::new(-offset_m)),
            },
        }),
    ];

    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh.clone()),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert_eq!(plan.rows.len(), 11);
    if angle == 0.0 {
        assert_eq!(plan.enforcement, ConstraintEnforcement::DirectElimination);
        assert!(summary.constraint_matrix.is_none());
    } else {
        assert_eq!(plan.enforcement, ConstraintEnforcement::AffineReduction);
        assert_eq!(summary.constraint_matrix.as_ref().unwrap().row_count(), 11);
    }
    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
    assert!(solve.converged, "{:?}", solve.diagnostics);
    let global_offset = [0.0, cos * offset_m, sin * offset_m];
    let expected_displacement = [
        global_offset[0],
        global_offset[1],
        global_offset[2], // node 1
        global_offset[0],
        global_offset[1],
        global_offset[2], // node 2
        0.0,
        0.0,
        0.0, // node 3
        global_offset[0],
        global_offset[1],
        global_offset[2], // node 4
    ];
    assert_eq!(solve.solution.len(), expected_displacement.len());
    for (actual, expected) in solve.solution.iter().zip(expected_displacement) {
        assert!((actual - expected).abs() <= 1.0e-10);
    }

    let fields = recover_result_fields(
        &summary,
        &solve,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();
    let normal = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "face_normal_offset")
        .unwrap();
    let youngs_pa = 200.0e9;
    let poisson = 0.3;
    let shear_pa = youngs_pa / (2.0 * (1.0 + poisson));
    let lambda_pa = youngs_pa * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let expected_normal_reaction_n = (lambda_pa + 2.0 * shear_pa) * offset_m / 6.0;
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-7,
            "actual={actual} expected={expected}"
        );
    };
    assert_eq!(normal.active_row_count, 3);
    assert!(normal.maximum_displacement_error_meters <= 1.0e-10);
    for (actual, direction) in normal
        .resultant_force_newtons
        .into_iter()
        .zip([0.0, cos, sin])
    {
        close(actual, direction * expected_normal_reaction_n);
    }
    close(
        normal.constraint_work_joules,
        expected_normal_reaction_n * offset_m,
    );
    for component in result.resultant_force_newtons {
        close(component, 0.0);
    }
    let energy = fields
        .iter()
        .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap()[0];
    close(energy, 0.5 * expected_normal_reaction_n * offset_m);
    let stress = fields
        .iter()
        .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_STRESS)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(stress.len(), 6);
    let lateral_stress = -lambda_pa * offset_m;
    let normal_stress = -(lambda_pa + 2.0 * shear_pa) * offset_m;
    for (actual, expected) in stress.iter().zip([
        lateral_stress,
        normal_stress * cos * cos + lateral_stress * sin * sin,
        normal_stress * sin * sin + lateral_stress * cos * cos,
        (normal_stress - lateral_stress) * sin * cos,
        0.0,
        0.0,
    ]) {
        close(*actual, expected);
    }
}

fn framed_zero(
    constraint_id: &str,
    node: u64,
    dofs: &[StructuralTranslationDof],
    frame: CoordinateFrame,
) -> AnalysisConstraint {
    AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: constraint_id.into(),
        frame,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::point(format!("node:{node}")),
            translations: dofs
                .iter()
                .copied()
                .map(|dof| TranslationDofValue {
                    dof,
                    value: PrescribedValue::Constant(Displacement::new(0.0)),
                })
                .collect(),
            rotations: Vec::new(),
        }),
    })
}
