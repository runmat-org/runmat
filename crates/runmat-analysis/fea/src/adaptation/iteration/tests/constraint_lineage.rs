//! Persistent targets are rebuilt across a real h/p topology change.
use super::governed::model;
use crate::adaptation::iteration::StructuralAdaptationConstraintLineage;
use crate::constraints::CanonicalStructuralConstraintPlan;
use runmat_analysis_core::{
    AnalysisConstraint, CoordinateFrame, RegionTarget, StructuralConstraint,
    StructuralConstraintKind,
};
use runmat_meshing_core::{
    fixtures::{canonical_tetrahedron_solver_mesh, connected_quadratic_tetrahedron_solver_mesh},
    ElementOrder, PersistentEntityKind,
};

// witness: FEA-ADAPTATION-GOVERNANCE-001/family_transfer_constraints @
//   adaptive_analysis/tet4|tet10/numerical_kernel;
//   boundary: Real Tet4 to Tet10 persistent face clamp rebuilds9 to18 rows, retains entity identity
//   and rejects foreign/missing lineage. No reaction/state transfer or adaptive solve convergence
//   claim.
// witness: FEA-SHARED-CONSTRAINTS-001/recovery_after_adaptation @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A persistent clamp rebuilds new midside constraints without numeric node aliases
//   after refinement.
#[test]
fn persistent_clamp_rebuilds_new_midside_constraints_without_numeric_node_aliases() {
    let mut source = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let mut target = connected_quadratic_tetrahedron_solver_mesh();
    for mesh in [&mut source, &mut target] {
        mesh.topology.boundary_faces[0].provenance.push(
            runmat_meshing_core::fixtures::fixture_entity(
                PersistentEntityKind::Face,
                "adaptive_clamp",
            ),
        );
        mesh.topology.boundary_faces[0].provenance.sort();
        mesh.seal_canonical_digest().unwrap();
    }
    let mut model = model();
    model.constraints = vec![AnalysisConstraint::Structural(StructuralConstraint {
        constraint_id: "persistent_clamp".into(),
        frame: CoordinateFrame::Global,
        kind: StructuralConstraintKind::Fixed {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "adaptive_clamp",
            )),
        },
    })];
    let source_plan = CanonicalStructuralConstraintPlan::compile(&model, &source).unwrap();
    let target_plan = CanonicalStructuralConstraintPlan::compile(&model, &target).unwrap();
    assert_eq!(source_plan.rows.len(), 9);
    assert_eq!(target_plan.rows.len(), 18);
    assert_eq!(
        source_plan.groups[0].targets[0].matched_entities,
        target_plan.groups[0].targets[0].matched_entities
    );
    assert_eq!(
        source_plan.groups[0].targets[0]
            .matched_node_identities
            .len(),
        3
    );
    assert_eq!(
        target_plan.groups[0].targets[0]
            .matched_node_identities
            .len(),
        6
    );
    let lineage = StructuralAdaptationConstraintLineage::derive(&model, &source, &target).unwrap();
    assert_eq!(
        lineage.source_plan_digest,
        source_plan.content_digest().unwrap()
    );
    assert_eq!(
        lineage.target_plan_digest,
        target_plan.content_digest().unwrap()
    );
    assert!(source_plan.validate_against(&model, &target).is_err());
    let mut missing = target.clone();
    missing.topology.boundary_faces[0]
        .provenance
        .retain(|e| e.source_topology_id != "adaptive_clamp");
    missing.seal_canonical_digest().unwrap();
    assert!(StructuralAdaptationConstraintLineage::derive(&model, &source, &missing).is_err());
}
