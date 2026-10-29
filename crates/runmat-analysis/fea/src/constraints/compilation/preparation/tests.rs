use super::*;
use crate::{
    constraints::{compile::canonical_reference_points, CanonicalStructuralConstraintPlan},
    fixtures::{fixture_model, mixed_solid_beam_model, FixtureId},
};
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, Position, ReferencePoint, ReferencePointCoupling,
    RegionTarget, StructuralConstraintKind,
};
use runmat_meshing_core::ElementOrder;

// witness: FEA-SHARED-AUTHORITIES-001/shared_topology_property_and_constraint_preparation @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Preparation composes shared inventories and preserves original dispatch.
#[test]
fn preparation_composes_shared_inventories_and_preserves_original_dispatch() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = crate::constraints::compilation_tests::mesh(order);
        let solid = fixture_model(FixtureId::CantileverLinearStatic);
        let mixed = mixed_solid_beam_model(mesh.topology.nodes[0].coordinates_m.canonical_value());
        let explicit = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
        for (model, mesh) in [
            (&solid, Some(&mesh)),
            (&mixed, Some(&mesh)),
            (&explicit, None),
        ] {
            let profile = constraint_preparation_workspace_profile(model, mesh).unwrap();
            let inputs = CompilationInputs::derive(model, mesh).unwrap();
            assert_eq!(
                profile.authored_bytes,
                plan::constraint_authored_workspace_profile(model)
                    .unwrap()
                    .total_requested_bytes
            );
            assert_eq!(profile.topology_bytes, inputs.workspace_bytes().unwrap());
            assert_eq!(
                profile.reference_point_bytes,
                compile::constraint_reference_point_workspace_profile(model)
                    .unwrap()
                    .total_requested_bytes
            );
            let mut largest = 0;
            for constraint in model.constraints.iter().filter_map(|c| c.structural()) {
                largest = largest.max(
                    prepared::profile::derive(constraint, inputs.mesh(), inputs.structural())
                        .unwrap()
                        .total_requested_bytes,
                );
            }
            assert_eq!(profile.largest_target_preparation_bytes, largest);
            assert_eq!(
                profile.total_requested_bytes,
                profile.authored_bytes
                    + profile.topology_bytes
                    + profile.reference_point_bytes
                    + largest
            );
            let before = serde_json::to_vec(model).unwrap();
            let plan = CanonicalStructuralConstraintPlan::compile_for_model(model, mesh).unwrap();
            let bytes = plan.canonical_bytes().unwrap();
            let cold = CanonicalStructuralConstraintPlan::from_canonical_bytes(&bytes).unwrap();
            assert_eq!(cold, plan);
            assert_eq!(serde_json::to_vec(model).unwrap(), before);
            assert_eq!(inputs.construct().unwrap().identity(), plan.topology);
        }
    }
}

#[test]
fn reference_inventory_contains_original_identity_workspace_and_retained_copies() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let id = " référence ".repeat(128);
    let position = [0.2, 0.3, -1.0];
    model.constraints = (0..4)
        .map(|ordinal| {
            AnalysisConstraint::Structural(StructuralConstraint {
                constraint_id: format!("original/{ordinal}"),
                frame: CoordinateFrame::Global,
                kind: StructuralConstraintKind::RigidCoupling(ReferencePointCoupling {
                    target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                        "face:0",
                    )),
                    reference_point: ReferencePoint {
                        reference_point_id: id.clone(),
                        position: Position::new(position),
                    },
                }),
            })
        })
        .collect();
    let profile = compile::constraint_reference_point_workspace_profile(&model).unwrap();
    let original_identity = plan::canonical_value_workspace_bytes(&(id.trim(), position)).unwrap();
    assert_eq!(profile.identity_materialization_bytes, original_identity);
    assert_eq!(profile.candidate_slots, 4);
    let borrowed = plan::structural_constraints(&model).unwrap();
    let points = canonical_reference_points(&borrowed).unwrap();
    assert_eq!(points.len(), 1);
    let retained = points.capacity() * std::mem::size_of::<CanonicalStructuralReferencePoint>()
        + points
            .iter()
            .map(|p| p.reference_point_id.capacity())
            .sum::<usize>();
    assert!(profile.total_requested_bytes >= retained as u64 + original_identity);
    assert_eq!(points[0].position_meters, position);
    assert_eq!(points[0].reference_point_id, id.trim());
}

#[test]
fn empty_inventory_and_cancelled_preparation_reject_without_topology_output() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    let mesh = crate::constraints::compilation_tests::mesh(ElementOrder::Tet4);
    let profile = constraint_preparation_workspace_profile(&model, Some(&mesh)).unwrap();
    assert_eq!(profile.largest_target_preparation_bytes, 0);
    assert_eq!(
        compile::constraint_reference_point_workspace_profile(&model)
            .unwrap()
            .identity_materialization_bytes,
        0
    );
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        constraint_preparation_workspace_profile(&model, Some(&mesh)).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
    assert_eq!(
        CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap_err(),
        ConstraintPlanError::Cancelled
    );
}
