//! Independent references for counted construction, not substitute continuum qualification.
use super::*;
use crate::fixtures::{fixture_model, FixtureId};
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Position, ReferencePoint, ReferencePointCoupling,
    RegionTarget, StructuralConstraint, StructuralConstraintKind, StructuralRotationDof,
    StructuralTranslationDof,
};
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn component(dof: CanonicalStructuralDof) -> usize {
    match dof {
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux)
        | CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx) => 0,
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy)
        | CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry) => 1,
        CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz)
        | CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz) => 2,
    }
}
// witness: FEA-STRUCTURAL-CONSTRAINT-001/normal_symmetry_rigid_coupling @
//   linear_structural/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Both-order surface weights, normals and coupling rows match independent geometry.
#[test]
fn both_order_surface_weights_normals_and_coupling_rows_match_independent_geometry() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let target = resolve_nodal_target(
            &mesh,
            &RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0")),
            "surface",
        )
        .unwrap();
        let normals = normal::boundary_node_normals(&mesh, &target, "surface").unwrap();
        for id in &target.matched_node_identities {
            let normal = normals.get(id).unwrap();
            assert!(
                normal[0].abs() < 1e-14
                    && normal[1].abs() < 1e-14
                    && (normal[2] + 1.).abs() < 1e-14
            );
        }
        let reference = ReferencePoint {
            reference_point_id: "reference".into(),
            position: Position::new([0.2, 0.3, -1.]),
        };
        let identity = runmat_execution::Digest::sha256(b"independent-reference");
        for kind in [
            ConstraintCouplingKind::Rigid,
            ConstraintCouplingKind::Distributing,
        ] {
            let (rows, evidence) = coupling::compile_reference_coupling(
                &mesh, &target, &reference, identity, kind, "surface",
            )
            .unwrap();
            // Independent equal subtriangle area measures on the unit right triangle.
            for weight in &evidence.target_weights {
                let node = mesh
                    .topology
                    .nodes
                    .iter()
                    .find(|n| n.stable_identity == weight.node_identity)
                    .unwrap();
                let expected = if order == ElementOrder::Tet4 {
                    1. / 3.
                } else if node.node_id <= 3 {
                    1. / 12.
                } else {
                    1. / 4.
                };
                assert!((weight.weight - expected).abs() < 1e-14);
            }
            let translation = [0.7, -0.2, 0.5];
            let rotation = [0.03, -0.05, 0.07];
            for row in &rows {
                let balance: f64 = row
                    .terms
                    .iter()
                    .map(|term| {
                        let value = match term.owner {
                            CanonicalStructuralDofOwner::ReferencePoint(id) => {
                                assert_eq!(id, identity);
                                match term.dof {
                                    CanonicalStructuralDof::Translation(_) => {
                                        translation[component(term.dof)]
                                    }
                                    CanonicalStructuralDof::Rotation(_) => {
                                        rotation[component(term.dof)]
                                    }
                                }
                            }
                            CanonicalStructuralDofOwner::StructuralNode(id) => {
                                let p = mesh
                                    .topology
                                    .nodes
                                    .iter()
                                    .find(|n| n.stable_identity == id)
                                    .unwrap()
                                    .coordinates_m
                                    .canonical_value();
                                let arm = std::array::from_fn::<_, 3, _>(|i| {
                                    p[i] - reference.position.canonical_value()[i]
                                });
                                let cross = [
                                    rotation[1] * arm[2] - rotation[2] * arm[1],
                                    rotation[2] * arm[0] - rotation[0] * arm[2],
                                    rotation[0] * arm[1] - rotation[1] * arm[0],
                                ];
                                translation[component(term.dof)] + cross[component(term.dof)]
                            }
                        };
                        term.coefficient * value
                    })
                    .sum();
                assert!((balance - row.right_hand_side).abs() < 1e-13);
            }
        }
    }
}

#[test]
fn borrowed_reference_point_preparation_preserves_aliases_and_conflict_rejection() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let constraint = |id: &str, point: [f64; 3]| {
        AnalysisConstraint::Structural(StructuralConstraint {
            constraint_id: id.into(),
            frame: CoordinateFrame::Global,
            kind: StructuralConstraintKind::RigidCoupling(ReferencePointCoupling {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "face:0",
                )),
                reference_point: ReferencePoint {
                    reference_point_id: " reference ".into(),
                    position: Position::new(point),
                },
            }),
        })
    };
    model.constraints = vec![
        constraint("a", [0.2, 0.3, -1.]),
        constraint("b", [0.2, 0.3, -1.]),
    ];
    let p = constraint_reference_point_workspace_profile(&model).unwrap();
    assert_eq!(p.candidate_slots, 2);
    assert_eq!(p.id_payload_bytes, 18);
    let authored = plan::structural_constraints(&model).unwrap();
    let points = compile::canonical_reference_points(&authored).unwrap();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].reference_point_id, "reference");
    model.constraints.push(constraint("c", [0.2, 0.3, -2.]));
    assert!(
        compile::canonical_reference_points(&plan::structural_constraints(&model).unwrap())
            .is_err()
    );
}

#[test]
fn cancelled_construction_rejects_all_profiles_and_compilation_at_entry() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let region = RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("face:0"));
    let target = resolve_nodal_target(&mesh, &region, "surface").unwrap();
    let model = fixture_model(FixtureId::CantileverLinearStatic);
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        constraint_target_workspace_profile(&mesh, &region).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        constraint_normal_workspace_profile(&mesh, &target).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        constraint_coupling_workspace_profile(&mesh, ConstraintCouplingKind::Rigid, "surface")
            .unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        constraint_reference_point_workspace_profile(&model).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}

#[test]
fn duplicate_reference_points_retain_last_authored_signed_zero_identity() {
    for signs in [[0.0, -0.0], [-0.0, 0.0]] {
        let constraints: Vec<_> = signs
            .into_iter()
            .enumerate()
            .map(|(ordinal, x)| StructuralConstraint {
                constraint_id: format!("coupling-{ordinal}"),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::RigidCoupling(ReferencePointCoupling {
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "face:0",
                    )),
                    reference_point: ReferencePoint {
                        reference_point_id: " reference ".into(),
                        position: Position::new([x, 0.3, -1.0]),
                    },
                }),
            })
            .collect();
        let authored: Vec<_> = constraints.iter().collect();
        let points = compile::canonical_reference_points(&authored).unwrap();
        assert_eq!(points.len(), 1);
        let expected = [signs[1], 0.3, -1.0];
        assert_eq!(
            points[0].position_meters[0].to_bits(),
            expected[0].to_bits()
        );
        assert_eq!(
            points[0].identity,
            plan::reference_point_identity("reference", expected).unwrap()
        );
    }
}
