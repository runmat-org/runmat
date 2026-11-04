use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Displacement, Length, PairedConstraint, PairingPolicy,
    Position, ReferencePoint, ReferencePointCoupling, StructuralConstraint,
    StructuralConstraintEquation, StructuralConstraintEquationTerm, StructuralConstraintKind,
    StructuralTranslationDof,
};
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest};

use super::*;
use crate::constraints::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralConstraintResult,
    CanonicalStructuralTopologyIdentity, ConstraintEnforcement,
};

#[test]
fn mixed_plan_round_trips_with_structural_rotation_rows() {
    let (model, mesh) = model_and_mesh();
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert!(matches!(
        plan.topology,
        CanonicalStructuralTopologyIdentity::ConformalMixed { .. }
    ));
    assert!(plan.rows.iter().any(|row| {
        row.constraint_id == "beam_root_rotation"
            && row
                .terms
                .iter()
                .any(|term| matches!(term.dof, CanonicalStructuralDof::Rotation(_)))
    }));
    let decoded =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&plan.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(decoded, plan);
    decoded.validate_against(&model, &mesh).unwrap();
}

#[test]
fn explicit_plan_round_trips_without_a_solver_mesh() {
    let model = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralBeamCantileverEndMomentReference,
    );
    let plan = CanonicalStructuralConstraintPlan::compile_for_model(&model, None).unwrap();
    assert!(matches!(
        plan.topology,
        CanonicalStructuralTopologyIdentity::Explicit { .. }
    ));
    assert_eq!(plan.evidence.authored_constraint_count, 1);
    assert_eq!(plan.evidence.active_row_count, 6);
    plan.validate_against_explicit(&model).unwrap();
    let decoded =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&plan.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(decoded, plan);
}

#[test]
fn general_equation_spans_solid_and_explicit_nodes() {
    let (mut model, mesh) = model_and_mesh();
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "solid_beam_ux_relation".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Equation(StructuralConstraintEquation {
                terms: vec![
                    StructuralConstraintEquationTerm {
                        target: runmat_analysis_core::RegionTarget::point("node:1"),
                        dof: StructuralTranslationDof::Ux,
                        coefficient: 1.0,
                    },
                    StructuralConstraintEquationTerm {
                        target: runmat_analysis_core::RegionTarget::point("node:102"),
                        dof: StructuralTranslationDof::Ux,
                        coefficient: -1.0,
                    },
                ],
                right_hand_side: Displacement::new(0.0),
            }),
        }));
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let row = plan
        .rows
        .iter()
        .find(|row| row.constraint_id == "solid_beam_ux_relation")
        .unwrap();
    assert_eq!(row.terms.len(), 2);
    assert_eq!(plan.enforcement, ConstraintEnforcement::AffineReduction);
    let result = crate::run_linear_static_with_options(
        &model,
        crate::ComputeBackend::Cpu,
        crate::LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            ..crate::LinearStaticSolveOptions::default()
        },
    )
    .unwrap();
    assert!(result
        .field(crate::FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE)
        .is_some());
}

#[test]
fn geometric_tie_couples_a_nonconformal_beam_node_to_a_solid_face() {
    let (mut model, mesh) = model_and_mesh();
    let face = &mesh.topology.boundary_faces[1];
    let mut face_center = [0.0; 3];
    for node_id in face.node_ids.iter().take(3) {
        let point = mesh
            .topology
            .nodes
            .iter()
            .find(|node| node.node_id == *node_id)
            .unwrap()
            .coordinates_m
            .canonical_value();
        for component in 0..3 {
            face_center[component] += point[component] / 3.0;
        }
    }
    let structural = model.structural.as_mut().unwrap();
    let root = structural
        .nodes
        .iter_mut()
        .find(|node| node.node_id == 101)
        .unwrap();
    root.coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new(face_center);
    root.solid_attachment = None;
    structural
        .nodes
        .iter_mut()
        .find(|node| node.node_id == 102)
        .unwrap()
        .coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([face_center[0] + 1.0, face_center[1], face_center[2]]);
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "nonconformal_beam_root_tie".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Tie(PairedConstraint {
                primary: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("solid_load"),
                ),
                secondary: runmat_analysis_core::RegionTarget::point("node:101"),
                pairing: PairingPolicy::GeometricTolerance(Length::new(1.0e-9)),
            }),
        }));

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let group = plan
        .groups
        .iter()
        .find(|group| group.constraint_id == "nonconformal_beam_root_tie")
        .unwrap();
    assert_eq!(group.row_count, 3);
    assert_eq!(group.pairings.len(), 1);
    assert_eq!(group.pairings[0].primary_weights.len(), 3);
    assert!(group.pairings[0].geometric_error_meters <= 1.0e-12);
    assert_eq!(plan.enforcement, ConstraintEnforcement::AffineReduction);

    let result = crate::run_linear_static_with_options(
        &model,
        crate::ComputeBackend::Cpu,
        crate::LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            ..crate::LinearStaticSolveOptions::default()
        },
    )
    .unwrap();
    assert!(result
        .field(crate::FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE)
        .is_some());
}

// witness: FEA-STRUCTURAL-CONSTRAINT-001/normal_symmetry_rigid_coupling @
//   linear_structural/shell/numerical_kernel; coverage: partial;
//   boundary: Shell symmetry uses the authored surface normal.
#[test]
fn explicit_shell_symmetry_uses_the_authored_surface_normal() {
    let (mut model, mesh) = shell_model_and_mesh();
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "shell_symmetry".into(),
            frame: CoordinateFrame::BoundaryNormalTangent,
            kind: StructuralConstraintKind::Symmetry {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("skin"),
                ),
            },
        }));

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let rows = plan
        .rows
        .iter()
        .filter(|row| row.constraint_id == "shell_symmetry")
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|row| row.terms.len() == 1));
    assert!(rows.iter().all(|row| {
        row.terms[0].dof == CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux)
            && (row.terms[0].coefficient.abs() - 1.0).abs() <= 1.0e-12
    }));
}

// witness: FEA-STRUCTURAL-CONSTRAINT-001/distributing_ties_equations @
//   linear_structural/shell/numerical_kernel; coverage: partial;
//   boundary: Shell distributing coupling retains reference-point evidence.
// witness: FEA-STRUCTURAL-EXECUTION-001/constraints_reactions_shell @
//   linear_structural/shell/numerical_kernel; coverage: partial;
//   boundary: Shell distributing coupling with reference-point evidence.
#[test]
fn explicit_shell_distributing_coupling_retains_reference_point_evidence() {
    let (mut model, mesh) = shell_model_and_mesh();
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "shell_reference_coupling".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::DistributingCoupling(ReferencePointCoupling {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("skin"),
                ),
                reference_point: ReferencePoint {
                    reference_point_id: "shell_control".into(),
                    position: Position::new([0.0, 0.0, 0.0]),
                },
            }),
        }));

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert_eq!(plan.reference_points.len(), 1);
    let group = plan
        .groups
        .iter()
        .find(|group| group.constraint_id == "shell_reference_coupling")
        .unwrap();
    let coupling = group.coupling.as_ref().unwrap();
    assert_eq!(coupling.kind, ConstraintCouplingKind::Distributing);
    assert_eq!(coupling.target_weights.len(), 3);
    assert!(
        (coupling
            .target_weights
            .iter()
            .map(|entry| entry.weight)
            .sum::<f64>()
            - 1.0)
            .abs()
            <= 1.0e-12
    );
    assert_eq!(group.row_count, 6);

    let result = crate::run_linear_static_with_options(
        &model,
        crate::ComputeBackend::Cpu,
        crate::LinearStaticSolveOptions {
            solver_mesh: Some(mesh.clone()),
            ..crate::LinearStaticSolveOptions::default()
        },
    )
    .unwrap();
    assert!(result
        .field(crate::FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE)
        .is_some());

    let excitations = crate::excitation::realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        crate::excitation::ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = crate::assembly::assemble_structural_system(
        &model,
        &excitations,
        Some(mesh.clone()),
        crate::assembly::StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let solve = crate::solve::linear::solve_linear_system(
        &summary,
        crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        &crate::solve::backend::cpu_reference::CpuReferenceBackend,
        None,
    )
    .unwrap();
    assert!(solve.converged, "{:?}", solve.diagnostics);
    let equilibrium = crate::algebra::apply_k(&summary.operator, &solve.solution)
        .into_iter()
        .zip(summary.operator.load())
        .map(|(internal, applied)| internal - applied)
        .collect::<Vec<_>>();
    let reaction = crate::solve::recover_structural_reaction(&summary, &equilibrium).unwrap();
    let constraint_result =
        CanonicalStructuralConstraintResult::recover_for_model_from_solver_state(
            &plan,
            &model,
            Some(&mesh),
            &solve.solution,
            &reaction,
        )
        .unwrap();
    assert_eq!(constraint_result.reference_points.len(), 1);
    assert!(constraint_result.reference_points[0]
        .translation_meters
        .iter()
        .all(|value| value.is_finite()));
    constraint_result
        .validate_against_model(&plan, &model, Some(&mesh))
        .unwrap();
}

// witness: FEA-STRUCTURAL-EXECUTION-001/constraints_reactions_shell @
//   linear_structural/shell/numerical_kernel; coverage: partial;
//   boundary: A mesh-free shell reference coupling uses the same generalized DOF path.
#[test]
fn mesh_free_shell_reference_coupling_uses_the_same_generalized_dof_path() {
    let mut model = crate::fixtures::fixture_model(
        crate::fixtures::FixtureId::StructuralShellPlateMomentReference,
    );
    model
        .constraints
        .push(AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "panel_reference_coupling".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::RigidCoupling(ReferencePointCoupling {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("shell_panel"),
                ),
                reference_point: ReferencePoint {
                    reference_point_id: "panel_control".into(),
                    position: Position::new([0.25, 0.25, 0.0]),
                },
            }),
        }));
    let plan = CanonicalStructuralConstraintPlan::compile_for_model(&model, None).unwrap();
    assert_eq!(plan.reference_points.len(), 1);
    let result = crate::run_linear_static_with_options(
        &model,
        crate::ComputeBackend::Cpu,
        crate::LinearStaticSolveOptions {
            ..crate::LinearStaticSolveOptions::default()
        },
    )
    .unwrap();
    assert!(result
        .field(crate::FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT)
        .is_some());
}

fn shell_model_and_mesh() -> (AnalysisModel, SolverMeshArtifact) {
    let (_, mesh) = model_and_mesh();
    let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
    (
        crate::fixtures::mixed_solid_beam_shell_model(attached_coordinates),
        mesh,
    )
}

fn model_and_mesh() -> (AnalysisModel, SolverMeshArtifact) {
    let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
    let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
    mesh.topology.nodes[0]
        .provenance
        .push(PersistentEntityId::new(
            PersistentEntityKind::Vertex,
            "beam_root".into(),
            vec!["root".into()],
        ));
    mesh.topology.nodes[0].provenance.sort();
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            "root".into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    mesh.topology.boundary_faces[1]
        .provenance
        .push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            "solid_load".into(),
            vec!["root".into()],
        ));
    mesh.topology.boundary_faces[1].provenance.sort();
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    (
        crate::fixtures::mixed_solid_beam_model(attached_coordinates),
        mesh,
    )
}

#[test]
fn mixed_dispatch_preserves_target_failure_and_cancellation_instead_of_falling_through() {
    let (model, mesh) = model_and_mesh();
    let structural = model.structural.as_ref().unwrap();
    let topology = CanonicalStructuralNodeSpace::derive(&mesh, structural).unwrap();
    let constraint = StructuralConstraint {
        constraint_id: "absent".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Fixed {
            target: runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("absent"),
            ),
        },
    };
    let direct = super::super::prepared::PreparedTargets::derive(
        &constraint,
        Some(&mesh),
        Some((structural, &topology)),
    )
    .err()
    .unwrap();
    let mut invalid = model.clone();
    invalid
        .constraints
        .push(AnalysisConstraint::Structural(constraint.clone()));
    assert_eq!(
        CanonicalStructuralConstraintPlan::compile(&invalid, &mesh).unwrap_err(),
        direct
    );
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        super::super::prepared::PreparedTargets::derive(
            &constraint,
            Some(&mesh),
            Some((structural, &topology))
        )
        .err()
        .unwrap(),
        ConstraintPlanError::Cancelled
    );
}
