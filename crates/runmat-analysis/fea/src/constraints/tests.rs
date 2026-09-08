use runmat_analysis_core::{
    AnalysisConstraint, AnalysisField, AnalysisModel, CoordinateFrame, Displacement,
    PrescribedStructuralDofs, PrescribedValue, RegionTarget, StructuralConstraint,
    StructuralConstraintEquation, StructuralConstraintEquationTerm, StructuralConstraintKind,
    StructuralTranslationDof, TranslationDofValue,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityId,
    SolverMeshArtifact, StableDigest,
};

use super::*;
use crate::fixtures::{fixture_model, FixtureId};

fn model_and_mesh() -> (AnalysisModel, SolverMeshArtifact) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId {
            kind: runmat_meshing_core::PersistentEntityKind::Face,
            source_topology_id: "root".into(),
            assembly_path: vec!["root".into()],
        });
        face.provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    (model, mesh)
}

#[test]
fn plan_is_invariant_to_authored_constraint_order() {
    let (mut model, mesh) = model_and_mesh();
    model.constraints = [
        ("prescribed_x", StructuralTranslationDof::Ux, 0.01),
        ("prescribed_y", StructuralTranslationDof::Uy, -0.02),
    ]
    .into_iter()
    .map(|(constraint_id, dof, meters)| {
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: constraint_id.into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::boundary("root"),
                translations: vec![TranslationDofValue {
                    dof,
                    value: PrescribedValue::Constant(Displacement::new(meters)),
                }],
                rotations: Vec::new(),
            }),
        })
    })
    .collect();
    let baseline = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    model.constraints.reverse();
    let reordered = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    assert_eq!(baseline, reordered);
}

#[test]
fn plan_round_trips_and_rejects_substituted_topology() {
    let (model, mesh) = model_and_mesh();
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let decoded =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&plan.canonical_bytes().unwrap())
            .unwrap();
    assert_eq!(decoded, plan);
    decoded.validate_against(&model, &mesh).unwrap();

    let mut substituted = mesh.clone();
    substituted.topology.nodes[0].coordinates_m[0] += 0.25;
    substituted.canonical_digest = StableDigest::ZERO;
    substituted.seal_canonical_digest().unwrap();
    assert_eq!(
        decoded.validate_against(&model, &substituted),
        Err(ConstraintPlanError::TopologyMismatch)
    );
}

#[test]
fn unsupported_frame_fails_and_equivalent_overlaps_are_evidenced() {
    let (mut model, mesh) = model_and_mesh();
    let structural = model.constraints[0].structural().unwrap().clone();
    let mut framed = structural.clone();
    framed.frame = CoordinateFrame::Material;
    model.constraints = vec![AnalysisConstraint::Structural(framed)];
    assert!(matches!(
        CanonicalStructuralConstraintPlan::compile(&model, &mesh),
        Err(ConstraintPlanError::UnsupportedFrame { .. })
    ));

    model.constraints = vec![
        AnalysisConstraint::Structural(structural.clone()),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "second_fixed".into(),
            ..structural
        }),
    ];
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert!(plan.evidence.redundant_row_count > 0);
    assert_eq!(
        plan.evidence.active_row_count + plan.evidence.redundant_row_count,
        plan.rows.len()
    );
    assert_eq!(
        plan.rows
            .iter()
            .filter(|row| row.reaction_recovery == ReactionRecovery::RedundantEquation)
            .count(),
        plan.evidence.redundant_row_count
    );
    assert_eq!(
        plan.rows
            .iter()
            .filter(|row| {
                row.reaction_recovery == ReactionRecovery::UnconstrainedEquilibriumResidual
            })
            .count(),
        plan.evidence.active_row_count
    );

    let mut substituted = plan;
    let active = substituted
        .rows
        .iter_mut()
        .find(|row| row.reaction_recovery == ReactionRecovery::UnconstrainedEquilibriumResidual)
        .unwrap();
    active.reaction_recovery = ReactionRecovery::RedundantEquation;
    active.identity = super::plan::row_identity(active).unwrap();
    assert!(substituted.validate().is_err());
}

#[test]
fn conflicting_prescriptions_fail_with_both_constraint_identities() {
    let (mut model, mesh) = model_and_mesh();
    model.constraints = [("first", 0.0), ("second", 0.1)]
        .into_iter()
        .map(|(constraint_id, meters)| {
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: constraint_id.into(),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                    target: RegionTarget::boundary("root"),
                    translations: vec![TranslationDofValue {
                        dof: StructuralTranslationDof::Ux,
                        value: PrescribedValue::Constant(Displacement::new(meters)),
                    }],
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();

    assert_eq!(
        CanonicalStructuralConstraintPlan::compile(&model, &mesh),
        Err(ConstraintPlanError::ConflictingRow {
            first_constraint_id: "first".into(),
            second_constraint_id: "second".into(),
        })
    );
}

#[test]
fn duplicate_constraint_ids_fail_with_the_typed_diagnostic() {
    let (mut model, mesh) = model_and_mesh();
    let structural = model.constraints[0].structural().unwrap().clone();
    model.constraints = vec![
        AnalysisConstraint::Structural(structural.clone()),
        AnalysisConstraint::Structural(structural.clone()),
    ];

    assert_eq!(
        CanonicalStructuralConstraintPlan::compile(&model, &mesh),
        Err(ConstraintPlanError::DuplicateConstraintId {
            constraint_id: structural.constraint_id,
        })
    );
}

fn equation_constraint(
    constraint_id: &str,
    left_node: u64,
    right_node: u64,
    right_hand_side_meters: f64,
) -> AnalysisConstraint {
    AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: constraint_id.into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Equation(StructuralConstraintEquation {
            terms: vec![
                StructuralConstraintEquationTerm {
                    target: RegionTarget::point(format!("node:{left_node}")),
                    dof: StructuralTranslationDof::Ux,
                    coefficient: 1.0,
                },
                StructuralConstraintEquationTerm {
                    target: RegionTarget::point(format!("node:{right_node}")),
                    dof: StructuralTranslationDof::Ux,
                    coefficient: -1.0,
                },
            ],
            right_hand_side: Displacement::new(right_hand_side_meters),
        }),
    })
}

#[test]
fn general_equation_compiles_to_one_canonical_sparse_row() {
    let (mut model, mesh) = model_and_mesh();
    model.constraints = vec![equation_constraint("relative_x", 1, 2, 0.01)];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    assert_eq!(plan.rows.len(), 1);
    assert_eq!(plan.rows[0].terms.len(), 2);
    assert_eq!(plan.rows[0].right_hand_side, 0.01);
    assert_eq!(
        plan.rows[0].reaction_recovery,
        ReactionRecovery::ConstraintMultiplier
    );
    assert_eq!(plan.groups[0].targets.len(), 2);
    assert_eq!(
        plan.groups[0].enforcement,
        ConstraintEnforcement::AffineReduction
    );
    assert_eq!(plan.evidence.rank, 1);
}

#[test]
fn equation_term_order_does_not_change_plan_identity() {
    let (mut model, mesh) = model_and_mesh();
    model.constraints = vec![equation_constraint("relative_x", 1, 2, 0.01)];
    let baseline = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    let equation = match &mut model.constraints[0] {
        AnalysisConstraint::Structural(StructuralConstraint {
            kind: StructuralConstraintKind::Equation(equation),
            ..
        }) => equation,
        _ => unreachable!(),
    };
    equation.terms.reverse();
    let reordered = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    assert_eq!(baseline, reordered);
}

#[test]
fn dependent_equations_are_evidenced_and_inconsistent_equations_fail() {
    let (mut model, mesh) = model_and_mesh();
    let first = equation_constraint("first", 1, 2, 0.01);
    let mut second = equation_constraint("second", 1, 2, 0.02);
    let AnalysisConstraint::Structural(StructuralConstraint {
        kind: StructuralConstraintKind::Equation(equation),
        ..
    }) = &mut second
    else {
        unreachable!()
    };
    for term in &mut equation.terms {
        term.coefficient *= 2.0;
    }
    model.constraints = vec![first.clone(), second.clone()];
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert_eq!(plan.evidence.active_row_count, 1);
    assert_eq!(plan.evidence.redundant_row_count, 1);

    let AnalysisConstraint::Structural(StructuralConstraint {
        kind: StructuralConstraintKind::Equation(equation),
        ..
    }) = &mut second
    else {
        unreachable!()
    };
    equation.right_hand_side = Displacement::new(0.03);
    model.constraints = vec![first, second];
    assert_eq!(
        CanonicalStructuralConstraintPlan::compile(&model, &mesh),
        Err(ConstraintPlanError::ConflictingRow {
            first_constraint_id: "first".into(),
            second_constraint_id: "second".into(),
        })
    );
}

#[test]
fn result_evidence_aggregates_only_canonical_reaction_owners() {
    let (mut model, mesh) = model_and_mesh();
    let structural = model.constraints[0].structural().unwrap().clone();
    model.constraints = vec![
        AnalysisConstraint::Structural(structural.clone()),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "redundant_fixed".into(),
            ..structural
        }),
    ];
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let node_count = mesh.topology.nodes.len();
    let mut reactions = vec![0.0; node_count * 3];
    let owner = plan
        .rows
        .iter()
        .find(|row| {
            row.reaction_recovery == ReactionRecovery::UnconstrainedEquilibriumResidual
                && row.terms[0].dof
                    == CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux)
        })
        .unwrap();
    let node_index = mesh
        .topology
        .nodes
        .iter()
        .position(|node| node.stable_identity == owner.terms[0].node_identity)
        .unwrap();
    reactions[node_index * 3] = 10.0;
    let fields = vec![
        AnalysisField::host_f64(
            crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            vec![node_count, 3],
            vec![0.0; node_count * 3],
        ),
        AnalysisField::host_f64(
            crate::FEA_FIELD_STRUCTURAL_REACTION_FORCE,
            vec![node_count, 3],
            reactions,
        ),
    ];

    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();

    assert_eq!(result.resultant_force_newtons, [10.0, 0.0, 0.0]);
    assert_eq!(result.maximum_prescribed_error_meters, 0.0);
    let redundant = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "redundant_fixed")
        .unwrap();
    assert_eq!(redundant.active_row_count, 0);
    assert!(redundant.redundant_row_count > 0);
    assert_eq!(redundant.resultant_force_newtons, [0.0; 3]);

    let mut substituted = result;
    substituted.resultant_force_newtons[0] = 20.0;
    assert_eq!(
        substituted.validate(),
        Err(ConstraintResultError::InvalidEvidence)
    );
}
