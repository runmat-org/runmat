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
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    post::fields::recover_result_fields,
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::solve_linear_system,
        preconditioner::SpdPreconditionerKind,
    },
};

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
    framed.kind = StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
        target: RegionTarget::boundary("root"),
        translations: vec![TranslationDofValue {
            dof: StructuralTranslationDof::Ux,
            value: PrescribedValue::Constant(Displacement::new(0.0)),
        }],
        rotations: Vec::new(),
    });
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
    active.identity = super::row::row_identity(active).unwrap();
    assert!(substituted.validate().is_err());
}

#[test]
fn cartesian_prescription_compiles_frame_axis_into_global_sparse_terms() {
    let (mut model, mesh) = model_and_mesh();
    let root_two = 2.0_f64.sqrt().recip();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "diagonal_guide".into(),
        frame: CoordinateFrame::Cartesian(runmat_analysis_core::CartesianFrame {
            frame_id: "diagonal".into(),
            origin: runmat_analysis_core::Position::new([0.0; 3]),
            x_axis: [root_two, root_two, 0.0],
            y_axis: [-root_two, root_two, 0.0],
        }),
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::point("node:1"),
            translations: vec![TranslationDofValue {
                dof: StructuralTranslationDof::Ux,
                value: PrescribedValue::Constant(Displacement::new(0.01)),
            }],
            rotations: Vec::new(),
        }),
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    assert_eq!(plan.rows.len(), 1);
    assert_eq!(plan.rows[0].terms.len(), 2);
    assert!((plan.rows[0].terms[0].coefficient - root_two).abs() <= 1.0e-12);
    assert!((plan.rows[0].terms[1].coefficient - root_two).abs() <= 1.0e-12);
    assert_eq!(
        plan.groups[0].enforcement,
        ConstraintEnforcement::AffineReduction
    );
}

#[test]
fn symmetry_uses_outward_geometry_normals_on_one_persistent_face() {
    let (mut model, mesh) = model_and_mesh();
    let face_entity = mesh.topology.boundary_faces[0]
        .provenance
        .iter()
        .find(|entity| entity.source_topology_id != "root")
        .unwrap()
        .source_topology_id
        .clone();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "symmetry_plane".into(),
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::Symmetry {
            target: RegionTarget::boundary(face_entity),
        },
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    assert_eq!(
        plan.rows.len(),
        mesh.topology.boundary_faces[0].node_ids.len()
    );
    for row in &plan.rows {
        let norm = row
            .terms
            .iter()
            .map(|term| term.coefficient * term.coefficient)
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() <= 1.0e-12);
        assert_eq!(row.right_hand_side, 0.0);
    }
}

#[test]
fn normal_displacement_and_antisymmetry_compile_distinct_normal_tangent_spaces() {
    let (mut model, mesh) = model_and_mesh();
    let face_entity = mesh.topology.boundary_faces[0]
        .provenance
        .iter()
        .find(|entity| entity.source_topology_id != "root")
        .unwrap()
        .source_topology_id
        .clone();
    let target = RegionTarget::boundary(face_entity);
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "normal_offset".into(),
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::NormalDisplacement {
            target: target.clone(),
            value: PrescribedValue::Constant(Displacement::new(0.002)),
        },
    })];
    let normal = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert_eq!(
        normal.rows.len(),
        mesh.topology.boundary_faces[0].node_ids.len()
    );
    assert!(normal.rows.iter().all(|row| row.right_hand_side == 0.002));

    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "antisymmetry_plane".into(),
        frame: CoordinateFrame::BoundaryNormalTangent,
        kind: StructuralConstraintKind::Antisymmetry { target },
    })];
    let antisymmetry = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    assert_eq!(antisymmetry.rows.len(), normal.rows.len() * 2);
    for normal_row in &normal.rows {
        let normal_vector = row_vector(normal_row);
        let tangent_rows = antisymmetry
            .rows
            .iter()
            .filter(|row| row.terms[0].node_identity == normal_row.terms[0].node_identity)
            .collect::<Vec<_>>();
        assert_eq!(tangent_rows.len(), 2);
        for tangent in tangent_rows {
            let tangent_vector = row_vector(tangent);
            let dot = normal_vector
                .into_iter()
                .zip(tangent_vector)
                .map(|(normal, tangent)| normal * tangent)
                .sum::<f64>();
            assert!(dot.abs() <= 1.0e-12);
        }
    }
}

fn row_vector(row: &CanonicalStructuralConstraintRow) -> [f64; 3] {
    let mut vector = [0.0; 3];
    for term in &row.terms {
        let component = match term.dof {
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => 0,
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => 1,
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => 2,
            CanonicalStructuralDof::Rotation(_) => unreachable!(),
        };
        vector[component] = term.coefficient;
    }
    vector
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

fn prescribed_constraint(
    constraint_id: &str,
    node: u64,
    dofs: &[StructuralTranslationDof],
) -> AnalysisConstraint {
    AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: constraint_id.into(),
        frame: CoordinateFrame::Global,
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

    let node_columns = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.stable_identity, index * 3))
        .collect::<std::collections::BTreeMap<_, _>>();
    let matrix =
        ResolvedConstraintMatrix::from_plan(&plan, mesh.topology.nodes.len() * 3, |node, dof| {
            let component = match dof {
                CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => 0,
                CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => 1,
                CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => 2,
                CanonicalStructuralDof::Rotation(_) => return None,
            };
            node_columns.get(&node).map(|base| base + component)
        })
        .unwrap();
    assert_eq!(matrix.row_offsets, vec![0, 2]);
    assert_eq!(matrix.column_indices, vec![0, 3]);
    assert_eq!(matrix.coefficients, vec![1.0, -1.0]);
    assert_eq!(matrix.right_hand_side, vec![0.01]);
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
fn canonical_solid_solve_enforces_affine_equation_and_recovers_reactions() {
    let (mut model, mesh) = model_and_mesh();
    model.excitations.clear();
    model.constraints = vec![
        prescribed_constraint(
            "anchor",
            1,
            &[
                StructuralTranslationDof::Ux,
                StructuralTranslationDof::Uy,
                StructuralTranslationDof::Uz,
            ],
        ),
        prescribed_constraint(
            "guide_two",
            2,
            &[StructuralTranslationDof::Uy, StructuralTranslationDof::Uz],
        ),
        prescribed_constraint("guide_three", 3, &[StructuralTranslationDof::Uz]),
        equation_constraint("relative_x", 2, 3, 0.001),
    ];
    let summary = assemble_structural_system(
        &model,
        Some(mesh.clone()),
        None,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    assert_eq!(summary.constraint_matrix.as_ref().unwrap().row_count(), 7);
    assert_eq!(
        summary
            .affine_constraint_reduction
            .as_ref()
            .unwrap()
            .reduced_dof_count,
        5
    );

    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );
    assert!(solve.converged, "{:?}", solve.diagnostics);
    assert!((solve.solution[3] - solve.solution[6] - 0.001).abs() <= 1.0e-10);

    let fields = recover_result_fields(&summary, &solve);
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();
    let equation = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "relative_x")
        .unwrap();
    assert_eq!(equation.row_reactions.len(), 1);
    assert!(equation.maximum_prescribed_error_meters <= 1.0e-10);
    assert!(equation.row_reactions[0].generalized_reaction.is_finite());
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
