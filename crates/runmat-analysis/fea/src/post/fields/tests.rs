use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation};
use super::*;
use crate::{
    assembly::{
        assemble_structural_system, dofs::StructuralDofKind, StructuralFormulationRequirement,
    },
    contracts::{
        FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE, FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
        FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE, FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
        FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
    },
    excitation::{realize_step_excitations, ExcitationSamplePoint, RealizedExcitations},
    fixtures::{fixture_model, FixtureId},
};
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, ObservationQuantity,
    PrescribedStructuralDofs, PrescribedValue, RegionTarget, StructuralConstraint,
    StructuralConstraintKind, StructuralTranslationDof, TranslationDofValue,
};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

fn explicit_recovery_summary(fixture: FixtureId) -> AssemblySummary {
    let model = fixture_model(fixture);
    assemble_structural_system(
        &model,
        &static_excitations(&model),
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("physical explicit fixture should assemble")
}

#[test]
fn scoped_beam_and_shell_resultants_match_serial_in_canonical_element_order() {
    use crate::planning::resource::LocalThreadCount;

    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(3).unwrap()).unwrap();
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        let mut summary = explicit_recovery_summary(fixture);
        // Multiple physical recovery entries force the admitted plan to use all
        // workers without manufacturing a second stiffness operator or load.
        if let Some(beam) = summary.structural_beam_recovery.first().cloned() {
            summary.structural_beam_recovery = (0..7)
                .map(|index| {
                    let mut element = beam.clone();
                    element.element_id = format!("beam_recovery_{index}");
                    element.material.youngs_modulus_pa = runmat_types::physical::Quantity::new(
                        element.material.youngs_modulus_pa.canonical_value()
                            * (1.0 + index as f64 * 0.1),
                    );
                    element
                })
                .collect();
        }
        if let Some(shell) = summary.structural_shell_recovery.first().cloned() {
            summary.structural_shell_recovery = (0..7)
                .map(|index| {
                    let mut element = shell.clone();
                    element.element_id = format!("shell_recovery_{index}");
                    element.geometry.nodes_m[2][1] += index as f64 * 0.1;
                    element
                })
                .collect();
        }
        let solution = (0..summary.dof_count)
            .map(|index| index as f64 * 1.0e-4)
            .collect::<Vec<_>>();
        let serial = recover_deformation_result_fields_with_plan(
            &summary,
            &solution,
            LocalExecutionPlan::serial(),
            DeformationRecoveryDetail::Complete,
        )
        .unwrap();
        for _ in 0..3 {
            let scoped = recover_deformation_result_fields_with_plan(
                &summary,
                &solution,
                plan,
                DeformationRecoveryDetail::Complete,
            )
            .unwrap();
            assert_eq!(scoped, serial);
        }
    }
}

// witness: FEA-STRUCTURAL-BEAM-001/physical_beam_mixed_dofs @
//   linear_structural/beam/numerical_kernel; coverage: partial;
//   boundary: Two-element beam recovery preserves the physical assembly fields.
#[test]
fn scoped_two_element_beam_recovery_preserves_physical_assembly_fields() {
    use crate::planning::resource::LocalThreadCount;
    use runmat_analysis_core::{StructuralElementKind, StructuralNode};

    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let structural = model.structural.as_mut().expect("beam fixture topology");
    structural.nodes.push(StructuralNode {
        node_id: 3,
        coordinates_m:
            runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new([
                2.0, 0.0, 0.0,
            ]),
        solid_attachment: None,
    });
    let mut second = structural.elements[0].clone();
    second.element_id = "beam_2".into();
    let StructuralElementKind::Beam(beam) = &mut second.kind else {
        panic!("fixture must contain a physical beam");
    };
    beam.node_ids = [2, 3];
    structural.elements.push(second);
    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    assert_eq!(summary.structural_beam_recovery.len(), 2);
    let solution = (0..summary.dof_count)
        .map(|index| index as f64 * 1.0e-4)
        .collect::<Vec<_>>();
    let serial = recover_deformation_result_fields_with_plan(
        &summary,
        &solution,
        LocalExecutionPlan::serial(),
        DeformationRecoveryDetail::Complete,
    )
    .unwrap();
    let scoped = recover_deformation_result_fields_with_plan(
        &summary,
        &solution,
        LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap(),
        DeformationRecoveryDetail::Complete,
    )
    .unwrap();
    assert_eq!(scoped, serial);
}

#[test]
fn scoped_explicit_recovery_rejects_nested_and_cancelled_execution() {
    use crate::planning::resource::{execute_scoped_chunks, LocalThreadCount};
    use crate::progress::{replace_fea_progress_context, FeaProgressContext};

    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap();
    for fixture in [
        FixtureId::StructuralBeamCantileverEndMomentReference,
        FixtureId::StructuralShellPlateMomentReference,
    ] {
        let mut summary = explicit_recovery_summary(fixture);
        if let Some(beam) = summary.structural_beam_recovery.first().cloned() {
            summary.structural_beam_recovery = vec![beam; 4];
        }
        if let Some(shell) = summary.structural_shell_recovery.first().cloned() {
            summary.structural_shell_recovery = vec![shell; 4];
        }
        let solution = vec![1.0e-4; summary.dof_count];
        let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
            Ok::<_, ()>(recover_deformation_result_fields_with_plan(
                &summary,
                &solution,
                plan,
                DeformationRecoveryDetail::Complete,
            ))
        })
        .unwrap();
        assert!(matches!(
            nested[0],
            Err(StructuralFieldRecoveryError::ScopedExecution { .. })
        ));

        let context = FeaProgressContext::default();
        context.cancellation.cancel();
        let _guard = replace_fea_progress_context(context);
        assert_eq!(
            recover_deformation_result_fields_with_plan(
                &summary,
                &solution,
                plan,
                DeformationRecoveryDetail::Complete
            ),
            Err(StructuralFieldRecoveryError::Cancelled),
        );
    }
}

#[test]
fn scoped_shell_recovery_preserves_element_attribution_on_invalid_geometry() {
    use crate::planning::resource::LocalThreadCount;

    let mut summary = explicit_recovery_summary(FixtureId::StructuralShellPlateMomentReference);
    let original = summary.structural_shell_recovery[0].clone();
    summary.structural_shell_recovery = vec![original.clone(); 4];
    summary.structural_shell_recovery[2].element_id = "degenerate_shell".into();
    summary.structural_shell_recovery[2].geometry.nodes_m[2] =
        summary.structural_shell_recovery[2].geometry.nodes_m[0];
    let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(2).unwrap()).unwrap();
    assert!(matches!(
        recover_deformation_result_fields_with_plan(
            &summary,
            &vec![1.0e-4; summary.dof_count],
            plan,
            DeformationRecoveryDetail::Complete,
        ),
        Err(StructuralFieldRecoveryError::Shell { element_id, .. })
            if element_id == "degenerate_shell"
    ));
}

fn static_excitations(model: &runmat_analysis_core::AnalysisModel) -> RealizedExcitations {
    realize_step_excitations(
        model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .expect("field-recovery fixture excitations should realize")
}

#[test]
fn synthetic_operator_recovery_does_not_publish_continuum_fields() {
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        None,
        StructuralFormulationRequirement::SyntheticOperator,
    )
    .expect("the synthetic fixture operator should assemble");

    let fields = recover_deformation_result_fields(&summary, &vec![0.0; summary.dof_count])
        .expect("kinematic recovery should remain available");

    assert!(fields
        .iter()
        .any(|field| field.field_id == FEA_FIELD_STRUCTURAL_DISPLACEMENT));
    for field_id in [
        FEA_FIELD_STRUCTURAL_STRAIN,
        FEA_FIELD_STRUCTURAL_STRESS,
        FEA_FIELD_STRUCTURAL_VON_MISES,
        FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
        FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
    ] {
        assert!(
            fields.iter().all(|field| field.field_id != field_id),
            "synthetic topology cannot support physical field `{field_id}`"
        );
    }
}

#[test]
fn generic_field_take_does_not_claim_constraint_group_reaction_ownership() {
    let mut fields = vec![
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_DISPLACEMENT, AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node),
            vec![1, VECTOR_COMPONENT_COUNT],
            vec![1.0, 2.0, 3.0],
        ),
        AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE, AnalysisFieldLayout::vector3(ObservationQuantity::ReactionForce, FieldLocation::Constraint),
            vec![1, VECTOR_COMPONENT_COUNT],
            vec![4.0, 5.0, 6.0],
        ),
    ];

    let recovered = take_structural_observable_fields(&mut fields).unwrap();

    assert_eq!(recovered.len(), 1);
    assert_eq!(
        recovered[0].definition().field_id(),
        FEA_FIELD_STRUCTURAL_DISPLACEMENT
    );
    assert_eq!(fields.len(), 1);
    assert_eq!(
        fields[0].field_id,
        FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE
    );
}

#[test]
fn solid_tetrahedron4_recovery_uses_solver_mesh_field_shapes() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.constraints.clear();
    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(tetrahedron4_mesh()),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("valid Tet4 system should assemble");
    let solve = LinearSolveResult {
        spectral_estimate: None,
        iterations: 1,
        residual_norm: 0.0,
        relative_residual: None,
        converged: true,
        host_sync_count: 0,
        solver_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: vec![
            0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03,
        ],
        solver_method: crate::contracts::SolverMethod::MatrixFreePcg,
        preconditioner: crate::contracts::SolverPreconditioner::None,
        diagnostics: Vec::new(),
    };

    let fields = recover_result_fields(&summary, &solve, LocalExecutionPlan::serial()).unwrap();
    assert_eq!(
        field_shape(&fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        &[4, 3]
    );
    assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_STRAIN), &[1, 6]);
    assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_STRESS), &[1, 6]);
    assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_VON_MISES), &[1]);
    assert_eq!(
        field_shape(&fields, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY),
        &[1]
    );
    assert_eq!(
        field_shape(&fields, FEA_FIELD_STRUCTURAL_NODAL_VON_MISES),
        &[4]
    );
    assert!(host_field(&fields, FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY)[0] >= 0.0);
    let element_von_mises = host_field(&fields, FEA_FIELD_STRUCTURAL_VON_MISES)[0];
    assert!(host_field(&fields, FEA_FIELD_STRUCTURAL_NODAL_VON_MISES)
        .iter()
        .all(|value| (*value - element_von_mises).abs() <= 1.0e-12));
    let metrics = structural_field_recovery_metrics(&summary, &solve.solution).unwrap();
    assert_eq!(metrics.basis, "solid_tetrahedron4_constant_strain");
    assert_eq!(metrics.solver_mesh_node_count, 4);
    assert_eq!(metrics.solver_mesh_element_count, 1);
}

#[test]
fn solid_tetrahedron10_recovery_uses_midside_displacements() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.constraints.clear();
    let mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet10);
    let displacement = mesh
        .topology
        .nodes
        .iter()
        .flat_map(|node| [node.coordinates_m.canonical_value()[0].powi(2), 0.0, 0.0])
        .collect::<Vec<_>>();
    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("valid Tet10 system should assemble");
    let solve = LinearSolveResult {
        spectral_estimate: None,
        iterations: 1,
        residual_norm: 0.0,
        relative_residual: None,
        converged: true,
        host_sync_count: 0,
        solver_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: displacement,
        solver_method: crate::contracts::SolverMethod::MatrixFreePcg,
        preconditioner: crate::contracts::SolverPreconditioner::None,
        diagnostics: Vec::new(),
    };

    let fields = recover_result_fields(&summary, &solve, LocalExecutionPlan::serial()).unwrap();
    let strain = host_field(&fields, FEA_FIELD_STRUCTURAL_STRAIN);
    let integration_point_strain =
        host_field(&fields, FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN);
    assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_STRAIN), &[1, 6]);
    assert_eq!(
        field_shape(&fields, FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN),
        &[4, 6]
    );
    assert!((strain[0] - 0.5).abs() < 1.0e-13);
    assert!(strain[1..].iter().all(|value| value.abs() < 1.0e-13));
    let recovered_xx = integration_point_strain
        .chunks_exact(6)
        .map(|point| point[0])
        .collect::<Vec<_>>();
    assert!(recovered_xx.windows(2).any(|pair| pair[0] != pair[1]));
    assert!(integration_point_strain
        .chunks_exact(6)
        .all(|point| point[1..].iter().all(|value| value.abs() < 1.0e-13)));
    assert_eq!(
        summary.structural_formulations[0].recovery,
        Some(
            runmat_analysis_core::StructuralRecoveryImplementation::Tet10IntegrationPointProjection
        )
    );
    let metrics = structural_field_recovery_metrics(&summary, &solve.solution).unwrap();
    assert_eq!(
        metrics.basis,
        "solid_tetrahedron10_integration_point_projection"
    );
    assert_eq!(metrics.solver_mesh_node_count, 10);

    let history = recover_transient_history_deformation_fields(&summary, &solve.solution).unwrap();
    for field_id in [
        FEA_FIELD_STRUCTURAL_DISPLACEMENT,
        FEA_FIELD_STRUCTURAL_STRAIN,
        FEA_FIELD_STRUCTURAL_STRESS,
        FEA_FIELD_STRUCTURAL_VON_MISES,
        FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
    ] {
        assert_eq!(
            host_field(&history, field_id),
            host_field(&fields, field_id)
        );
    }
    for field_id in [
        FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN,
        FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
        FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
    ] {
        assert!(history.iter().all(|field| field.field_id != field_id));
        assert!(fields.iter().any(|field| field.field_id == field_id));
    }
}

#[test]
fn reaction_force_retains_the_constrained_node_and_component() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "node_y".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::point("node:2"),
            translations: vec![TranslationDofValue {
                dof: StructuralTranslationDof::Uy,
                value: PrescribedValue::Constant(Displacement::new(0.0)),
            }],
            rotations: Vec::new(),
        }),
    })];
    let mesh = tetrahedron4_mesh();
    let node_index = mesh
        .topology
        .nodes
        .iter()
        .position(|node| node.node_id == 2)
        .expect("fixture contains node 2");
    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("component constraint should assemble");
    let constrained_dof = summary
        .structural_dof_layout
        .index(node_index, StructuralDofKind::Uy)
        .expect("node y DOF should exist");
    let mut internal_force = vec![0.0; summary.dof_count];
    internal_force[constrained_dof] = 42.0;

    let reactions = recover_reaction_force(
        &summary,
        &internal_force,
        summary.structural_dof_layout.node_count(),
    );

    assert_eq!(reactions.len(), 4 * VECTOR_COMPONENT_COUNT);
    assert_eq!(reactions[node_index * VECTOR_COMPONENT_COUNT + 1], 42.0);
    assert_eq!(
        reactions
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0)
            .map(|(index, _)| index)
            .collect::<Vec<_>>(),
        vec![node_index * VECTOR_COMPONENT_COUNT + 1]
    );
}

#[test]
fn finite_nonconverged_solve_still_recovers_solver_mesh_fields() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.constraints.clear();
    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(tetrahedron4_mesh()),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .expect("valid Tet4 system should assemble");
    let solve = LinearSolveResult {
        spectral_estimate: None,
        iterations: 10,
        residual_norm: 1.0e-4,
        relative_residual: None,
        converged: false,
        host_sync_count: 0,
        solver_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        device_apply_k_count: 0,
        device_apply_k_attempt_count: 0,
        accelerator_evidence: None,
        solution: vec![
            0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03,
        ],
        solver_method: crate::contracts::SolverMethod::MatrixFreePcg,
        preconditioner: crate::contracts::SolverPreconditioner::None,
        diagnostics: Vec::new(),
    };

    let fields = recover_result_fields(&summary, &solve, LocalExecutionPlan::serial()).unwrap();

    assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_VON_MISES), &[1]);
    assert_eq!(field_shape(&fields, FEA_FIELD_STRUCTURAL_STRAIN), &[1, 6]);
    assert_eq!(
        field_shape(&fields, FEA_FIELD_STRUCTURAL_DISPLACEMENT),
        &[4, 3]
    );
}

#[test]
fn observable_catalog_types_every_structural_deformation_family() {
    let definitions = structural_observable_field_definitions().unwrap();
    let expected = [
        (FEA_FIELD_STRUCTURAL_ROTATION, ObservationQuantity::Rotation),
        (
            FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE,
            ObservationQuantity::BeamAxialForce,
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
            ObservationQuantity::BeamShearForce,
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
            ObservationQuantity::ShellMembraneForce,
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
            ObservationQuantity::ShellBendingMoment,
        ),
    ];
    for (field_id, quantity) in expected {
        assert!(definitions.iter().any(|definition| {
            definition.field_id() == field_id && definition.quantity() == quantity
        }));
    }
    let mut identities = definitions
        .iter()
        .map(|definition| definition.identity())
        .collect::<Vec<_>>();
    identities.sort();
    identities.dedup();
    assert_eq!(identities.len(), definitions.len());
}

#[test]
fn statistical_linear_quantities_reuse_the_observable_catalog_semantics() {
    let definitions = structural_observable_field_definitions().unwrap();
    for quantity in StructuralLinearFieldQuantity::ALL {
        let definition = definitions
            .iter()
            .find(|definition| definition.field_id() == quantity.source_field_id())
            .expect("every statistical linear quantity must have one observable definition");
        assert_eq!(definition.quantity(), quantity.observation_quantity());
        assert_eq!(definition.location(), quantity.location());
    }
}

fn field_shape<'a>(fields: &'a [AnalysisField], field_id: &str) -> &'a [usize] {
    fields
        .iter()
        .find(|field| field.field_id == field_id)
        .map(|field| field.shape.as_slice())
        .expect("field should be present")
}

fn host_field<'a>(fields: &'a [AnalysisField], field_id: &str) -> &'a [f64] {
    fields
        .iter()
        .find(|field| field.field_id == field_id)
        .and_then(AnalysisField::as_host_f64)
        .expect("field should be present")
}

fn tetrahedron4_mesh() -> SolverMeshArtifact {
    crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4)
}
