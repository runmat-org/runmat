use runmat_analysis_core::{
    AnalysisConstraint, AnalysisModel, CoordinateFrame, Displacement, PrescribedStructuralDofs,
    PrescribedValue, RegionTarget, StructuralConstraint, StructuralConstraintKind,
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
