use runmat_analysis_core::{
    AnalysisConstraint, AnalysisField, AnalysisModel, CoordinateFrame, Displacement, Length,
    PairedConstraint, PairingPolicy, PeriodicConstraint, PeriodicTransform, Position,
    PrescribedStructuralDofs, PrescribedValue, ReferencePoint, ReferencePointCoupling,
    RegionTarget, Rotation, StructuralConstraint, StructuralConstraintEquation,
    StructuralConstraintEquationTerm, StructuralConstraintKind, StructuralTranslationDof,
    Translation, TranslationDofValue,
};
use runmat_execution::Digest;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityId,
    SolverMeshArtifact, StableDigest,
};

use super::*;
use crate::fixtures::{
    fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind, FixtureId,
};
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint, RealizedExcitations},
    post::fields::recover_result_fields,
    solve::{
        backend::{cpu_reference::CpuReferenceBackend, kind::LinearAlgebraBackendKind},
        linear::solve_linear_system,
        preconditioner::SpdPreconditionerKind,
    },
};

fn static_excitations(model: &AnalysisModel) -> RealizedExcitations {
    realize_step_excitations(
        model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .expect("constraint fixture excitations should realize")
}

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
    assert_eq!(plan.enforcement, ConstraintEnforcement::AffineReduction);
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
            .filter(|row| row.terms[0].owner == normal_row.terms[0].owner)
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

#[test]
fn translated_periodic_nodes_pair_bijectively_and_compile_vector_relations() {
    let (mut model, mesh) = model_and_mesh();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "periodic_x".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Periodic(PeriodicConstraint {
            pair: PairedConstraint {
                primary: RegionTarget::point("node:1"),
                secondary: RegionTarget::point("node:2"),
                pairing: PairingPolicy::Matching,
            },
            transform: PeriodicTransform::Translation(Translation::new([1.0, 0.0, 0.0])),
        }),
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    assert_eq!(plan.groups[0].pairings.len(), 1);
    assert_eq!(plan.rows.len(), 3);
    assert!(plan.rows.iter().all(|row| {
        row.terms.len() == 2 && row.terms.iter().map(|term| term.coefficient).sum::<f64>() == 0.0
    }));
}

#[test]
fn rotated_periodic_relation_rotates_displacements_with_geometry() {
    let (mut model, mesh) = model_and_mesh();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "quarter_turn".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Periodic(PeriodicConstraint {
            pair: PairedConstraint {
                primary: RegionTarget::point("node:3"),
                secondary: RegionTarget::point("node:4"),
                pairing: PairingPolicy::Matching,
            },
            transform: PeriodicTransform::Rotation {
                origin: runmat_analysis_core::Position::new([0.0; 3]),
                axis: [1.0, 0.0, 0.0],
                angle: Rotation::new(std::f64::consts::FRAC_PI_2),
            },
        }),
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    assert_eq!(plan.rows.len(), 3);
    let uz_row = plan
        .rows
        .iter()
        .find(|row| {
            row.terms.iter().any(|term| {
                term.owner
                    == CanonicalStructuralDofOwner::MeshNode(mesh.topology.nodes[3].stable_identity)
                    && term.dof == CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz)
                    && term.coefficient == 1.0
            })
        })
        .unwrap();
    assert!(uz_row.terms.iter().any(|term| {
        term.owner == CanonicalStructuralDofOwner::MeshNode(mesh.topology.nodes[2].stable_identity)
            && term.dof == CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy)
            && (term.coefficient + 1.0).abs() <= 1.0e-12
    }));
}

#[test]
fn geometric_tie_projects_secondary_node_to_primary_surface() {
    let (mut model, mut mesh) = model_and_mesh();
    mesh.topology.nodes[3].coordinates_m = [0.2, 0.3, 0.01];
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let primary = mesh.topology.boundary_faces[0]
        .provenance
        .iter()
        .find(|entity| entity.source_topology_id != "root")
        .unwrap()
        .source_topology_id
        .clone();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "surface_tie".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Tie(PairedConstraint {
            primary: RegionTarget::boundary(primary),
            secondary: RegionTarget::point("node:4"),
            pairing: PairingPolicy::GeometricTolerance(Length::new(0.02)),
        }),
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    assert_eq!(plan.groups[0].pairings.len(), 1);
    assert!((plan.groups[0].pairings[0].geometric_error_meters - 0.01).abs() <= 1.0e-12);
    let weights = plan.groups[0].pairings[0]
        .primary_weights
        .iter()
        .map(|weight| weight.weight)
        .collect::<Vec<_>>();
    assert_eq!(weights.len(), 3);
    assert!(weights.iter().all(|weight| *weight > 0.0));
    assert!((weights.into_iter().sum::<f64>() - 1.0).abs() <= 1.0e-12);
    assert_eq!(plan.rows.len(), 3);
}

#[test]
fn rigid_coupling_compiles_mesh_motion_against_six_reference_dofs() {
    let (mut model, mesh) = model_and_mesh();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "rigid_end".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::RigidCoupling(ReferencePointCoupling {
            target: RegionTarget::boundary("root"),
            reference_point: ReferencePoint {
                reference_point_id: "control".into(),
                position: Position::new([0.25, 0.25, 0.25]),
            },
        }),
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();

    assert_eq!(plan.reference_points.len(), 1);
    assert_eq!(plan.groups[0].reference_point_identities.len(), 1);
    assert_eq!(
        plan.groups[0].coupling.as_ref().unwrap().kind,
        ConstraintCouplingKind::Rigid
    );
    assert_eq!(plan.rows.len(), mesh.topology.nodes.len() * 3);
    assert!(plan.rows.iter().all(|row| {
        row.terms
            .iter()
            .any(|term| matches!(term.owner, CanonicalStructuralDofOwner::ReferencePoint(_)))
    }));
}

#[test]
fn distributing_coupling_uses_area_weights_and_six_resultant_relations() {
    let (mut model, mesh) = model_and_mesh();
    let face = mesh.topology.boundary_faces[0]
        .provenance
        .iter()
        .find(|entity| entity.source_topology_id != "root")
        .unwrap()
        .source_topology_id
        .clone();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "distributed_end".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::DistributingCoupling(ReferencePointCoupling {
            target: RegionTarget::boundary(face),
            reference_point: ReferencePoint {
                reference_point_id: "control".into(),
                position: Position::new([0.25, 0.25, 0.0]),
            },
        }),
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let coupling = plan.groups[0].coupling.as_ref().unwrap();

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
    assert_eq!(plan.rows.len(), 6);
    assert_eq!(plan.evidence.rank, 6);
}

#[test]
fn reference_coupling_rejects_a_rotationally_rank_deficient_target() {
    let (mut model, mesh) = model_and_mesh();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "point_coupling".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::RigidCoupling(ReferencePointCoupling {
            target: RegionTarget::point("node:1"),
            reference_point: ReferencePoint {
                reference_point_id: "control".into(),
                position: Position::new([1.0, 1.0, 1.0]),
            },
        }),
    })];

    let error = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap_err();

    assert!(matches!(
        error,
        ConstraintPlanError::Invalid(message)
            if message.contains("must contain non-collinear nodes")
    ));
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
    assert_eq!(plan.enforcement, ConstraintEnforcement::AffineReduction);
    assert_eq!(plan.evidence.rank, 1);

    let node_columns = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.stable_identity, index * 3))
        .collect::<std::collections::BTreeMap<_, _>>();
    let matrix =
        ResolvedConstraintMatrix::from_plan(&plan, mesh.topology.nodes.len() * 3, |owner, dof| {
            let CanonicalStructuralDofOwner::MeshNode(node) = owner else {
                return None;
            };
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
fn dense_global_equation_selects_multiplier_enforcement_before_affine_fill() {
    let constraint = equation_constraint("global", 1, 2, 0.0);
    let kind = &constraint.structural().unwrap().kind;
    let terms = (1_u8..=33)
        .map(|identity| CanonicalStructuralConstraintTerm {
            owner: CanonicalStructuralDofOwner::MeshNode(StableDigest::from_bytes([identity; 32])),
            dof: CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux),
            coefficient: 1.0,
        })
        .collect();
    let rows = vec![CanonicalStructuralConstraintRow {
        identity: Digest::from_bytes([1; 32]),
        constraint_id: "global".into(),
        terms,
        right_hand_side: 0.0,
        reaction_recovery: ReactionRecovery::ConstraintMultiplier,
    }];

    assert_eq!(
        super::enforcement::select(kind, &rows),
        ConstraintEnforcement::LagrangeMultiplier
    );
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
        &static_excitations(&model),
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
fn canonical_solid_solve_enforces_exact_multiplier_equations() {
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
    let mut summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh.clone()),
        None,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    assert_eq!(summary.constraint_matrix.as_ref().unwrap().row_count(), 7);
    summary.affine_constraint_reduction = None;

    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );

    assert!(solve.converged, "{:?}", solve.diagnostics);
    assert_eq!(
        solve.solver_method,
        crate::contracts::SolverMethod::MatrixFreeSaddlePointGmres
    );
    assert!((solve.solution[3] - solve.solution[6] - 0.001).abs() <= 1.0e-9);
    let fields = recover_result_fields(&summary, &solve);
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();
    let equation = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "relative_x")
        .unwrap();
    assert!(equation.maximum_prescribed_error_meters <= 1.0e-9);
    assert!(equation.row_reactions[0].generalized_reaction.is_finite());
}

#[test]
fn reference_point_wrench_solves_through_distributing_coupling() {
    let (mut model, mesh) = model_and_mesh();
    let boundary_ids = mesh
        .topology
        .boundary_faces
        .iter()
        .map(|face| {
            face.provenance
                .iter()
                .find(|entity| entity.source_topology_id != "root")
                .unwrap()
                .source_topology_id
                .clone()
        })
        .collect::<Vec<_>>();
    model.constraints = vec![
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "anchor".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Fixed {
                target: RegionTarget::boundary(&boundary_ids[0]),
            },
        }),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "remote_load".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::DistributingCoupling(ReferencePointCoupling {
                target: RegionTarget::boundary(&boundary_ids[3]),
                reference_point: ReferencePoint {
                    reference_point_id: "control".into(),
                    position: Position::new([1.0 / 3.0; 3]),
                },
            }),
        }),
    ];
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "remote_wrench".into(),
            region_id: "control".into(),
            kind: FixtureExcitationKind::Wrench {
                force: [10.0, 0.0, 0.0],
                moment: [0.0, 1.0, 0.0],
                point_m: [1.0 / 3.0; 3],
            },
        }],
    );

    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh.clone()),
        None,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    assert_eq!(summary.dof_count, 18);
    assert_eq!(summary.structural_dof_layout.reference_point_count(), 1);
    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );
    assert!(solve.converged, "{:?}", solve.diagnostics);
    let matrix = summary.constraint_matrix.as_ref().unwrap();
    for row in 0..matrix.row_count() {
        let start = matrix.row_offsets[row];
        let end = matrix.row_offsets[row + 1];
        let value = matrix.column_indices[start..end]
            .iter()
            .zip(&matrix.coefficients[start..end])
            .map(|(column, coefficient)| coefficient * solve.solution[*column])
            .sum::<f64>();
        assert!((value - matrix.right_hand_side[row]).abs() <= 1.0e-9);
    }
    let internal = crate::operator::apply_k_unconstrained(&summary.operator, &solve.solution);
    let reaction = internal
        .iter()
        .zip(&summary.operator.rhs)
        .map(|(internal, applied)| internal - applied)
        .collect::<Vec<_>>();
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let result = CanonicalStructuralConstraintResult::recover_from_solver_state(
        &plan,
        &mesh,
        &solve.solution,
        &reaction,
    )
    .unwrap();
    assert_eq!(result.reference_points.len(), 1);
    assert!((result.reference_points[0].constraint_force_newtons[0] + 10.0).abs() <= 1.0e-7);
    assert!((result.reference_points[0].constraint_moment_newton_meters[1] + 1.0).abs() <= 1.0e-7);
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
        .position(|node| {
            owner.terms[0].owner == CanonicalStructuralDofOwner::MeshNode(node.stable_identity)
        })
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
