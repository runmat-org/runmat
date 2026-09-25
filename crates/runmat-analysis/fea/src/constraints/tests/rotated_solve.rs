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
    );
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
