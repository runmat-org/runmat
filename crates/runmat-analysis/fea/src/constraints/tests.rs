use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
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
        modal::solve_modal_system,
        nonlinear::{solve_nonlinear_system, NonlinearSolveOptions},
        preconditioner::SpdPreconditionerKind,
        transient::{solve_transient_system, StructuralTimeStepPolicy, TransientSolveOptions},
    },
    ComputeBackend,
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
        face.provenance.push(PersistentEntityId::new(
            runmat_meshing_core::PersistentEntityKind::Face,
            "root".into(),
            vec!["root".into()],
        ));
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
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
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
    {
        let quantity_change = 0.25;
        let quantity_index = 0;
        let quantity_slot = &mut substituted.topology.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
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
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
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
            target: RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(face_entity)),
            ),
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
fn fixed_constraint_projects_a_logical_region_bound_to_multiple_faces() {
    let (mut model, mesh) = model_and_mesh();
    let mut target = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
        "mounting-surfaces",
    ));
    target
        .bind_resolved_entities(vec![
            mesh.topology.boundary_faces[0].provenance[0].clone(),
            mesh.topology.boundary_faces[1].provenance[0].clone(),
        ])
        .unwrap();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "fixed_mounts".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Fixed { target },
    })];

    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let selected_nodes = mesh.topology.boundary_faces[0]
        .node_ids
        .iter()
        .chain(&mesh.topology.boundary_faces[1].node_ids)
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(plan.rows.len(), selected_nodes.len() * 3);
}

// witness: FEA-STRUCTURAL-CONSTRAINT-001/normal_symmetry_rigid_coupling @
//   linear_structural/tet4/numerical_kernel; coverage: partial;
//   boundary: Normal displacement and antisymmetry compile distinct normal and tangent spaces on a
//   fixture mesh (Tet4).
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
    let target = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_boundary_selection(
        &(face_entity),
    ));
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
                    == CanonicalStructuralDofOwner::StructuralNode(
                        mesh.topology.nodes[3].stable_identity,
                    )
                    && term.dof == CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz)
                    && term.coefficient == 1.0
            })
        })
        .unwrap();
    assert!(uz_row.terms.iter().any(|term| {
        term.owner
            == CanonicalStructuralDofOwner::StructuralNode(mesh.topology.nodes[2].stable_identity)
            && term.dof == CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy)
            && (term.coefficient + 1.0).abs() <= 1.0e-12
    }));
}

#[test]
fn geometric_tie_projects_secondary_node_to_primary_surface() {
    let (mut model, mut mesh) = model_and_mesh();
    mesh.topology.nodes[3].coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([0.2, 0.3, 0.01]);
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
            primary: RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(primary)),
            ),
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
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("root")),
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
            target: RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
            ),
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
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "root",
                    )),
                    translations: vec![TranslationDofValue {
                        dof: StructuralTranslationDof::Ux,
                        value: PrescribedValue::Constant(Displacement::new(meters)),
                    }],
                    rotations: Vec::new(),
                }),
            })
        })
        .collect();

    let error = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap_err();
    assert!(matches!(
        error,
        ConstraintPlanError::ConflictingRow {
            first_constraint_id,
            second_constraint_id,
            conflicting_dof: Some(CanonicalStructuralDofAddress {
                owner: CanonicalStructuralDofOwner::StructuralNode(identity),
                dof: CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux),
            }),
        } if first_constraint_id == "first"
            && second_constraint_id == "second"
            && identity != StableDigest::ZERO
    ));
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

// witness: FEA-SHARED-CONSTRAINTS-001/canonical_plan_and_affine_reduction @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A general equation compiles to one canonical sparse row.
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
            let CanonicalStructuralDofOwner::StructuralNode(node) = owner else {
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

// witness: FEA-SHARED-CONSTRAINTS-001/compatible_enforcement @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A dense global equation selects multiplier enforcement before any affine fill; direct
//   elimination and penalty selection are separate cases.
#[test]
fn dense_global_equation_selects_multiplier_enforcement_before_affine_fill() {
    let constraint = equation_constraint("global", 1, 2, 0.0);
    let kind = &constraint.structural().unwrap().kind;
    let terms = (1_u8..=33)
        .map(|identity| CanonicalStructuralConstraintTerm {
            owner: CanonicalStructuralDofOwner::StructuralNode(StableDigest::from_bytes(
                [identity; 32],
            )),
            dof: CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux),
            coefficient: 1.0,
        })
        .collect();
    let rows = vec![CanonicalStructuralConstraintRow {
        identity: Digest::from_bytes([1; 32]),
        constraint_id: "global".into(),
        terms,
        right_hand_side: 0.0,
        residual_quantity: ConstraintResidualQuantity::Displacement,
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
            conflicting_dof: None,
        })
    );
}

#[path = "tests/rank_plan.rs"]
mod rank_plan;

#[path = "tests/rotated_solve.rs"]
mod rotated_solve;

#[path = "tests/curved_normals.rs"]
mod curved_normals;

#[path = "tests/normal_solve.rs"]
mod normal_solve;

// witness: FEA-SHARED-CONSTRAINTS-001/canonical_plan_and_affine_reduction @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A canonical solid solve enforces an affine equation and recovers reactions.
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
    assert!(solve.relative_residual.is_some_and(|value| value <= 1e-8));
    assert!((solve.solution[3] - solve.solution[6] - 0.001).abs() <= 1.0e-10);

    let fields = recover_result_fields(
        &summary,
        &solve,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();
    let equation = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "relative_x")
        .unwrap();
    assert_eq!(equation.row_reactions.len(), 1);
    assert!(equation.maximum_displacement_error_meters <= 1.0e-10);
    assert!(equation.row_reactions[0].generalized_reaction.is_finite());
}

#[test]
fn affine_static_solve_preserves_constraint_displacement_under_stiffness_scaling() {
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
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let reference = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );
    assert!(reference.converged, "{:?}", reference.diagnostics);

    let scale = 1.0e-20;
    let original = &summary.operator;
    let stiffness = original.operators().stiffness();
    let scaled_stiffness = crate::algebra::RealSparseMatrix::try_new(
        stiffness.pattern().clone(),
        stiffness
            .values()
            .iter()
            .map(|value| value * scale)
            .collect(),
        stiffness.role(),
        stiffness.properties(),
        stiffness.contributions().to_vec(),
    )
    .unwrap();
    summary.operator = crate::algebra::AssembledRealSystem::try_new(
        crate::algebra::CanonicalRealOperators::try_new(
            scaled_stiffness,
            original.operators().mass().clone(),
            original.operators().damping().clone(),
        )
        .unwrap(),
        original.constrained_dofs().to_vec(),
        original.load().iter().map(|value| value * scale).collect(),
    )
    .unwrap();
    let scaled = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );
    assert!(scaled.converged, "{:?}", scaled.diagnostics);
    assert!(scaled.iterations > 0);
    for (actual, expected) in scaled.solution.iter().zip(&reference.solution) {
        assert!((actual - expected).abs() <= 1.0e-9);
    }
    assert!((scaled.solution[3] - scaled.solution[6] - 0.001).abs() <= 1.0e-9);
    let plan = crate::planning::resource::LocalExecutionPlan::fixed(
        crate::planning::resource::LocalThreadCount::new(3).unwrap(),
    )
    .unwrap();
    let scoped = crate::solve::linear::solve_linear_system_with_execution(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        plan,
    )
    .unwrap();
    assert!(scoped.converged, "{:?}", scoped.diagnostics);
    assert_eq!(
        scoped,
        crate::solve::linear::solve_linear_system_with_execution(
            &summary,
            SpdPreconditionerKind::Jacobi,
            LinearAlgebraBackendKind::CpuReference,
            &CpuReferenceBackend,
            None,
            plan,
        )
        .unwrap()
    );
    for (actual, expected) in scoped.solution.iter().zip(&scaled.solution) {
        assert!((actual - expected).abs() <= 1.0e-9);
    }
}

#[test]
fn affine_solver_rejects_a_substituted_reduction_before_iteration() {
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
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    summary
        .affine_constraint_reduction
        .as_mut()
        .unwrap()
        .recovery_coefficients[0] += 0.25;

    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );

    assert_eq!(solve.iterations, 0);
    assert!(!solve.converged);
    assert!(solve
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FEA_INVALID_AFFINE_SOLVER_ADMISSION"));
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
    assert!(solve.relative_residual.is_some_and(|value| value <= 1e-8));
    assert_eq!(
        solve.solver_method,
        crate::contracts::SolverMethod::SparseDirectCholesky
    );
    assert!((solve.solution[3] - solve.solution[6] - 0.001).abs() <= 1.0e-9);
    let execution = crate::planning::resource::LocalExecutionPlan::fixed(
        crate::planning::resource::LocalThreadCount::new(3).unwrap(),
    )
    .unwrap();
    let parallel = crate::solve::linear::solve_linear_system_with_execution(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        execution,
    )
    .unwrap();
    assert!(parallel.converged, "{:?}", parallel.diagnostics);
    assert_eq!(parallel.iterations, solve.iterations);
    assert_eq!(parallel.solver_method, solve.solver_method);
    assert_eq!(parallel.diagnostics, solve.diagnostics);
    assert!(parallel.residual_norm <= 1.0e-7);
    assert!((parallel.solution[3] - parallel.solution[6] - 0.001).abs() <= 1.0e-9);
    for (scoped, serial) in parallel.solution.iter().zip(&solve.solution) {
        assert!((scoped - serial).abs() <= 1.0e-12);
    }
    let repeated = crate::solve::linear::solve_linear_system_with_execution(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
        execution,
    )
    .unwrap();
    assert_eq!(repeated, parallel);
    let nested = crate::planning::resource::execute_scoped_chunks(
        crate::planning::resource::LocalExecutionPlan::serial(),
        1,
        |_| {
            Ok::<_, ()>(crate::solve::linear::solve_linear_system_with_execution(
                &summary,
                SpdPreconditionerKind::Jacobi,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                None,
                execution,
            ))
        },
    )
    .unwrap();
    assert!(matches!(
        &nested[0],
        Err(crate::algebra::ScopedSpmvError::Execution(
            crate::planning::resource::ScopedLocalExecutionError::NestedExecution
        ))
    ));
    {
        let cancellation = crate::progress::FeaCancellationToken::new();
        cancellation.cancel();
        let _context =
            crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
                cancellation,
                ..crate::progress::FeaProgressContext::default()
            });
        assert!(matches!(
            crate::solve::linear::solve_linear_system_with_execution(
                &summary,
                SpdPreconditionerKind::Jacobi,
                LinearAlgebraBackendKind::CpuReference,
                &CpuReferenceBackend,
                None,
                execution,
            ),
            Err(crate::algebra::ScopedSpmvError::Execution(
                crate::planning::resource::ScopedLocalExecutionError::Work {
                    source: crate::algebra::SpmvWorkError::Cancelled,
                    ..
                }
            ))
        ));
    }
    let fields = recover_result_fields(
        &summary,
        &solve,
        crate::planning::resource::LocalExecutionPlan::serial(),
    )
    .unwrap();
    let plan = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();
    let equation = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "relative_x")
        .unwrap();
    assert!(equation.maximum_displacement_error_meters <= 1.0e-9);
    assert!(equation.row_reactions[0].generalized_reaction.is_finite());
}

#[test]
fn multiplier_solver_rejects_an_unremoved_mechanism_before_iteration() {
    let (mut model, mesh) = model_and_mesh();
    model.excitations.clear();
    model.constraints = vec![
        prescribed_constraint("insufficient_support", 1, &[StructuralTranslationDof::Ux]),
        equation_constraint("insufficient_relation", 2, 3, 0.0),
    ];
    let mut summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    summary.affine_constraint_reduction = None;

    let solve = solve_linear_system(
        &summary,
        SpdPreconditionerKind::Jacobi,
        LinearAlgebraBackendKind::CpuReference,
        &CpuReferenceBackend,
        None,
    );

    assert_eq!(solve.iterations, 0);
    assert!(!solve.converged);
    assert!(
        solve
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "FEA_INVALID_AFFINE_SOLVER_ADMISSION"),
        "{:?}",
        solve.diagnostics
    );
}

#[test]
fn modal_reduction_enforces_the_complete_homogeneous_constraint_matrix() {
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
        equation_constraint("relative_x", 2, 3, 0.0),
    ];
    let mut summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    // Modal semantics use a homogeneous null-space reduction even when static enforcement was
    // authored as a multiplier system.
    summary.affine_constraint_reduction = None;
    let constraints = summary.constraint_matrix.clone().unwrap();
    let result = solve_modal_system(
        &summary,
        1,
        1.0e-6,
        crate::ModalConvergencePolicy::default(),
        ComputeBackend::Cpu,
    );

    assert!(result.converged, "{:?}", result.diagnostics);
    let mode = &result.mode_shapes[0];
    for row in 0..constraints.row_count() {
        let start = constraints.row_offsets[row];
        let end = constraints.row_offsets[row + 1];
        let residual = constraints.column_indices[start..end]
            .iter()
            .zip(&constraints.coefficients[start..end])
            .map(|(column, coefficient)| coefficient * mode[*column])
            .sum::<f64>();
        assert!(residual.abs() <= 1.0e-9, "row {row} residual={residual}");
    }
}

#[test]
fn modal_reduction_rejects_nonhomogeneous_constraints_before_iteration() {
    let (mut model, mesh) = model_and_mesh();
    model.excitations.clear();
    model.constraints = vec![equation_constraint("offset", 2, 3, 0.001)];
    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();

    let result = solve_modal_system(
        &summary,
        1,
        1.0e-6,
        crate::ModalConvergencePolicy::default(),
        ComputeBackend::Cpu,
    );

    assert!(!result.converged);
    assert!(result.frequencies_hz.is_empty());
    assert_eq!(result.device_apply_k_attempt_count, 0);
    assert!(result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "FEA_MODAL_PREPARATION_FAILED"));
}

#[test]
fn transient_steps_enforce_affine_constraint_kinematics() {
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
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let constraints = summary.constraint_matrix.clone().unwrap();
    for multiplier in [false, true] {
        let mut enforced = summary.clone();
        if multiplier {
            enforced.affine_constraint_reduction = None;
        }
        let result = solve_transient_system(
            &enforced,
            TransientSolveOptions {
                time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
                ..TransientSolveOptions::default()
            },
            ComputeBackend::Cpu,
        )
        .expect("constrained structural dynamics should solve");

        assert_eq!(result.converged_steps, 2, "{:?}", result.diagnostics);
        let constraint_value = |row: usize, state: &[f64]| {
            let start = constraints.row_offsets[row];
            let end = constraints.row_offsets[row + 1];
            constraints.column_indices[start..end]
                .iter()
                .zip(&constraints.coefficients[start..end])
                .map(|(column, coefficient)| coefficient * state[*column])
                .sum::<f64>()
        };
        for state in result.displacement_snapshots.iter().skip(1) {
            for row in 0..constraints.row_count() {
                let value = constraint_value(row, state);
                assert!(
                    (value - constraints.right_hand_side[row]).abs() <= 1.0e-8,
                    "multiplier={multiplier} row {row} expected={} actual={value}",
                    constraints.right_hand_side[row]
                );
            }
        }
        for (quantity, states) in [
            ("velocity", &result.velocity_snapshots),
            ("acceleration", &result.acceleration_snapshots),
        ] {
            for state in states {
                for row in 0..constraints.row_count() {
                    let value = constraint_value(row, state);
                    assert!(
                        value.abs() <= 1.0e-10,
                        "multiplier={multiplier} {quantity} row {row} expected=0 actual={value}"
                    );
                }
            }
        }
    }
}

#[test]
fn transient_steps_enforce_nonzero_direct_prescriptions() {
    let (mut model, mesh) = model_and_mesh();
    model.excitations.clear();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "prescribed_x".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
            target: RegionTarget::point("node:1"),
            translations: vec![TranslationDofValue {
                dof: StructuralTranslationDof::Ux,
                value: PrescribedValue::Constant(Displacement::new(0.001)),
            }],
            rotations: Vec::new(),
        }),
    })];
    let summary = assemble_structural_system(
        &model,
        &static_excitations(&model),
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    assert!(summary.constraint_matrix.is_none());
    assert!(summary.operator.constrained_dofs()[0]);
    assert_eq!(summary.operator.load()[0], 0.001);

    let result = solve_transient_system(
        &summary,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            ..TransientSolveOptions::default()
        },
        ComputeBackend::Cpu,
    )
    .expect("prescribed-displacement structural dynamics should solve");

    assert_eq!(result.converged_steps, 2, "{:?}", result.diagnostics);
    for state in result.displacement_snapshots.iter().skip(1) {
        assert!(
            (state[0] - 0.001).abs() <= 1.0e-10,
            "prescribed={} diagnostics={:?}",
            state[0],
            result.diagnostics
        );
    }
    assert!(result
        .velocity_snapshots
        .iter()
        .all(|state| state[0].abs() <= 1.0e-10));
    assert!(result
        .acceleration_snapshots
        .iter()
        .all(|state| state[0].abs() <= 1.0e-10));
    assert!(result
        .residual_norms
        .iter()
        .all(|residual| *residual <= 1.0e-6));
}

#[test]
fn transient_direct_prescription_refinement_is_scale_consistent() {
    for prescribed_m in [1.0e-4, 1.0e-3, 1.0e-2] {
        let (mut model, mesh) = model_and_mesh();
        model.excitations.clear();
        model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "prescribed_x".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::Prescribed(PrescribedStructuralDofs {
                target: RegionTarget::point("node:1"),
                translations: vec![TranslationDofValue {
                    dof: StructuralTranslationDof::Ux,
                    value: PrescribedValue::Constant(Displacement::new(prescribed_m)),
                }],
                rotations: Vec::new(),
            }),
        })];
        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        let solved = solve_transient_system(
            &summary,
            TransientSolveOptions {
                time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
                ..TransientSolveOptions::default()
            },
            ComputeBackend::Cpu,
        )
        .expect("constant prescribed displacement must converge at every tested scale");
        assert_eq!(solved.converged_steps, 2);
        for state in solved.displacement_snapshots.iter().skip(1) {
            assert!((state[0] - prescribed_m).abs() <= 1.0e-10);
        }
        assert!(solved
            .residual_norms
            .iter()
            .all(|value| value.is_finite() && *value <= 1.0e-6));
    }
}

// witness: FEA-SHARED-CONSTRAINTS-001/nonlinear_inequality_state @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Nonlinear accepted states preserve complete constraint kinematics.
#[test]
fn nonlinear_accepted_states_preserve_complete_constraint_kinematics() {
    for lifting in [0.0, 0.001] {
        let (mut model, mesh) = model_and_mesh();
        model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
            "affine_nonlinear",
            vec![runmat_analysis_core::ProcedureStep::structural_nonlinear(
                "nonlinear",
            )],
        );
        replace_fixture_excitations(
            &mut model,
            vec![FixtureExcitation {
                id: "physical_force".into(),
                region_id: "root".into(),
                kind: FixtureExcitationKind::Force([1.0, 0.0, 0.0]),
            }],
        );
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
            equation_constraint("relative_x", 2, 3, lifting),
        ];
        let summary = assemble_structural_system(
            &model,
            &static_excitations(&model),
            Some(mesh.clone()),
            StructuralFormulationRequirement::PhysicalDiscretization,
        )
        .unwrap();
        let constraints = summary.constraint_matrix.clone().unwrap();
        let result = solve_nonlinear_system(
            &model,
            &summary,
            &NonlinearSolveOptions {
                increment_count: 2,
                max_newton_iters: 8,
                solver_mesh: Some(mesh),
                ..NonlinearSolveOptions::default()
            },
            ComputeBackend::Cpu,
        );
        // Prescribed (affine) supports scale with the load factor like the applied loads, so a
        // nonzero lifting is reached at load factor one through admissible increments.
        let result = result.expect("affine constraints admit actual solid equilibrium");
        assert_eq!(result.displacement_snapshots.len(), 2);
        assert!(result
            .displacement_snapshots
            .last()
            .unwrap()
            .iter()
            .any(|value| value.abs() > 0.0));
        for (state, load_factor) in result
            .displacement_snapshots
            .iter()
            .zip(&result.load_factors)
        {
            for row in 0..constraints.row_count() {
                let start = constraints.row_offsets[row];
                let end = constraints.row_offsets[row + 1];
                let value = constraints.column_indices[start..end]
                    .iter()
                    .zip(&constraints.coefficients[start..end])
                    .map(|(column, coefficient)| coefficient * state[*column])
                    .sum::<f64>();
                let expected = load_factor * constraints.right_hand_side[row];
                assert!(
                    (value - expected).abs() <= 1.0e-8,
                    "row {row} expected={expected} actual={value}"
                );
            }
        }
    }
}

// witness: FEA-STRUCTURAL-CONSTRAINT-001/distributing_ties_equations @
//   linear_structural/tet4/public_study; coverage: partial;
//   boundary: The linear static pipeline admits and solves a reference-point wrench through a
//   distributing coupling under its preflight solver admission. Resultant accuracy is covered by
//   the kernel test below; rigid couplings and other procedures are not exercised.
#[test]
fn preflight_admits_a_reference_point_wrench_through_distributing_coupling() {
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
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(&boundary_ids[0])),
                ),
            },
        }),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "remote_load".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::DistributingCoupling(ReferencePointCoupling {
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(&boundary_ids[3])),
                ),
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

    let result = crate::run_linear_static_with_options(
        &model,
        ComputeBackend::Cpu,
        crate::LinearStaticSolveOptions {
            solver_mesh: Some(mesh),
            ..crate::LinearStaticSolveOptions::default()
        },
    );
    assert!(result.is_ok(), "{:?}", result.err());
}

// witness: FEA-STRUCTURAL-CONSTRAINT-001/distributing_ties_equations @
//   linear_structural/tet4/numerical_kernel; coverage: partial;
//   boundary: A reference-point wrench solves through distributing coupling on a fixture solid.
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
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(&boundary_ids[0])),
                ),
            },
        }),
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: "remote_load".into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::DistributingCoupling(ReferencePointCoupling {
                target: RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_boundary_selection(&(&boundary_ids[3])),
                ),
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
    assert!(solve.relative_residual.is_some_and(|value| value <= 1e-8));
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
    let internal = crate::algebra::apply_k(&summary.operator, &solve.solution);
    let reaction = internal
        .iter()
        .zip(summary.operator.load())
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
            owner.terms[0].owner
                == CanonicalStructuralDofOwner::StructuralNode(node.stable_identity)
        })
        .unwrap();
    reactions[node_index * 3] = 10.0;
    let fields = vec![
        AnalysisField::host_f64(
            crate::FEA_FIELD_STRUCTURAL_DISPLACEMENT, AnalysisFieldLayout::vector3(ObservationQuantity::Displacement, FieldLocation::Node),
            vec![node_count, 3],
            vec![0.0; node_count * 3],
        ),
        AnalysisField::host_f64(
            crate::FEA_FIELD_STRUCTURAL_REACTION_FORCE, AnalysisFieldLayout::vector3(ObservationQuantity::ReactionForce, FieldLocation::Node),
            vec![node_count, 3],
            reactions,
        ),
    ];

    let result = CanonicalStructuralConstraintResult::recover(&plan, &mesh, &fields).unwrap();

    assert_eq!(result.resultant_force_newtons, [10.0, 0.0, 0.0]);
    assert_eq!(result.maximum_displacement_error_meters, 0.0);
    assert_eq!(result.maximum_rotation_error_radians, 0.0);
    let redundant = result
        .groups
        .iter()
        .find(|group| group.constraint_id == "redundant_fixed")
        .unwrap();
    assert_eq!(redundant.active_row_count, 0);
    assert!(redundant.redundant_row_count > 0);
    assert_eq!(redundant.resultant_force_newtons, [0.0; 3]);

    let mut force_substituted = result.clone();
    force_substituted.groups[0].resultant_force_newtons[0] += 1.0;
    force_substituted.resultant_force_newtons[0] += 1.0;
    force_substituted.validate().unwrap();
    assert_eq!(
        force_substituted.validate_against(&plan),
        Err(ConstraintResultError::InvalidEvidence)
    );

    let mut moment_substituted = result.clone();
    moment_substituted.groups[0].resultant_moment_about_global_origin_newton_meters[0] += 1.0;
    moment_substituted.resultant_moment_about_global_origin_newton_meters[0] += 1.0;
    moment_substituted.validate().unwrap();
    assert_eq!(
        moment_substituted.validate_against_model(&plan, &model, Some(&mesh)),
        Err(ConstraintResultError::InvalidEvidence)
    );

    let mut substituted = result;
    substituted.resultant_force_newtons[0] = 20.0;
    assert_eq!(
        substituted.validate(),
        Err(ConstraintResultError::InvalidEvidence)
    );
}
