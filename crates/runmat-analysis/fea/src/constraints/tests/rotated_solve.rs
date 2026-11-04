use super::*;

#[test]
fn rotated_prescription_satisfies_tet4_shear_reaction_and_energy() {
    // Nodes 1-3 form the fixed z=0 face of the unit Tet4. Moving node 4
    // along a 45-degree in-plane frame axis produces pure shear in xz/yz.
    let (mut model, mesh) = model_and_mesh();
    let axis = 2.0_f64.sqrt().recip();
    let prescribed_m = 0.001;
    model.excitations.clear();
    model.constraints = vec![
        prescribed_constraint(
            "fixed_one",
            1,
            &[
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ],
        ),
        prescribed_constraint(
            "fixed_two",
            2,
            &[
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ],
        ),
        prescribed_constraint(
            "fixed_three",
            3,
            &[
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ],
        ),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "diagonal_motion".into(),
            frame: CoordinateFrame::Cartesian(runmat_analysis_core::CartesianFrame {
                frame_id: "diagonal".into(),
                origin: Position::new([0.0; 3]),
                x_axis: [axis, axis, 0.0],
                y_axis: [-axis, axis, 0.0],
            }),
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::point("node:4"),
                translations: vec![TranslationDofValue {
                    dof: StructuralTranslationDof::Ux,
                    value: PrescribedValue::Constant(Displacement::new(prescribed_m)),
                }],
                rotations: Vec::new(),
            }),
        }),
    ];

    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh.clone()),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
    assert!(solve.converged, "{:?}", solve.diagnostics);
    let expected_tip = axis * prescribed_m;
    for component in [9, 10] {
        assert!((solve.solution[component] - expected_tip).abs() <= 1.0e-10);
    }
    assert!(solve.solution[11].abs() <= 1.0e-10);

    let fields = recover_result_fields(
        &summary,
        &solve,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();
    let motion = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "diagonal_motion")
        .unwrap();
    let shear_modulus_pa = 200.0e9 / (2.0 * (1.0 + 0.3));
    let expected_reaction_n = shear_modulus_pa * prescribed_m / 6.0;
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-7,
            "actual={actual} expected={expected}"
        );
    };
    assert_eq!(motion.active_row_count, 1);
    assert!(motion.maximum_displacement_error_meters <= 1.0e-10);
    close(
        motion.row_reactions[0].generalized_reaction,
        expected_reaction_n,
    );
    close(
        motion.resultant_force_newtons[0],
        axis * expected_reaction_n,
    );
    close(
        motion.resultant_force_newtons[1],
        axis * expected_reaction_n,
    );
    close(motion.resultant_force_newtons[2], 0.0);
    close(result.resultant_force_newtons[0], 0.0);
    close(result.resultant_force_newtons[1], 0.0);
    close(result.resultant_force_newtons[2], 0.0);
    close(
        motion.constraint_work_joules,
        expected_reaction_n * prescribed_m,
    );
    let energy = fields
        .iter()
        .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap()[0];
    close(energy, 0.5 * expected_reaction_n * prescribed_m);
}

#[test]
fn quarter_turn_periodicity_preserves_physical_expansion_and_reactions() {
    // The rotation about x maps node 3 at (0,1,0) onto node 4 at (0,0,1).
    // The prescribed uy at node 3 therefore induces uz at node 4. With the
    // x-axis nodes fixed, u=(0,d*y,d*z) is an exact constant-strain solution.
    let (mut model, mesh) = model_and_mesh();
    let prescribed_m = 0.001;
    model.excitations.clear();
    model.constraints = vec![
        prescribed_constraint(
            "fixed_one",
            1,
            &[
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ],
        ),
        prescribed_constraint(
            "fixed_two",
            2,
            &[
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ],
        ),
        prescribed_constraint("rotation_anchor", 3, &[StructuralTranslationDof::Uz]),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "radial_expansion".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::point("node:3"),
                translations: vec![TranslationDofValue {
                    dof: StructuralTranslationDof::Uy,
                    value: PrescribedValue::Constant(Displacement::new(prescribed_m)),
                }],
                rotations: Vec::new(),
            }),
        }),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "quarter_turn".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Periodic(PeriodicConstraint {
                pair: PairedConstraint {
                    primary: RegionTarget::point("node:3"),
                    secondary: RegionTarget::point("node:4"),
                    pairing: PairingPolicy::Matching,
                },
                transform: PeriodicTransform::Rotation {
                    origin: Position::new([0.0; 3]),
                    axis: [1.0, 0.0, 0.0],
                    angle: Rotation::new(std::f64::consts::FRAC_PI_2),
                },
            }),
        }),
    ];

    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh.clone()),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    assert_eq!(summary.constraint_matrix.as_ref().unwrap().row_count(), 11);
    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    )
    .unwrap();
    assert!(solve.converged, "{:?}", solve.diagnostics);
    let expected_displacement = [
        0.0,
        0.0,
        0.0, // node 1
        0.0,
        0.0,
        0.0, // node 2
        0.0,
        prescribed_m,
        0.0, // node 3
        0.0,
        0.0,
        prescribed_m, // node 4
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
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();
    let expansion = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "radial_expansion")
        .unwrap();
    let periodic = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "quarter_turn")
        .unwrap();
    let youngs_pa = 200.0e9;
    let poisson = 0.3;
    let shear_pa = youngs_pa / (2.0 * (1.0 + poisson));
    let lambda_pa = youngs_pa * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let volume_m3 = 1.0 / 6.0;
    let face_reaction_n = (2.0 * lambda_pa + 2.0 * shear_pa) * prescribed_m * volume_m3;
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-7,
            "actual={actual} expected={expected}"
        );
    };
    assert_eq!(periodic.active_row_count, 3);
    assert!(periodic.maximum_displacement_error_meters <= 1.0e-10);
    close(periodic.resultant_force_newtons[0], 0.0);
    close(periodic.resultant_force_newtons[1], -face_reaction_n);
    close(periodic.resultant_force_newtons[2], face_reaction_n);
    close(periodic.constraint_work_joules, 0.0);
    assert_eq!(expansion.active_row_count, 1);
    close(
        expansion.row_reactions[0].generalized_reaction,
        2.0 * face_reaction_n,
    );
    close(
        expansion.constraint_work_joules,
        2.0 * face_reaction_n * prescribed_m,
    );
    close(result.resultant_force_newtons[0], 0.0);
    close(result.resultant_force_newtons[1], 0.0);
    close(result.resultant_force_newtons[2], 0.0);
    let energy = fields
        .iter()
        .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_TOTAL_STRAIN_ENERGY)
        .unwrap()
        .as_host_f64()
        .unwrap()[0];
    close(energy, face_reaction_n * prescribed_m);
    let stress = fields
        .iter()
        .find(|field| field.field_id == crate::FEA_FIELD_STRUCTURAL_STRESS)
        .unwrap()
        .as_host_f64()
        .unwrap();
    assert_eq!(stress.len(), 6);
    let lateral_stress_pa = 2.0 * lambda_pa * prescribed_m;
    let radial_stress_pa = (2.0 * lambda_pa + 2.0 * shear_pa) * prescribed_m;
    for (actual, expected) in stress.iter().zip([
        lateral_stress_pa,
        radial_stress_pa,
        radial_stress_pa,
        0.0,
        0.0,
        0.0,
    ]) {
        close(*actual, expected);
    }
}
