//! Compilation must borrow unrelated model data; this is not a whole-compiler memory bound.
#[path = "../support/peak_allocator/counting.rs"]
mod heap;

use runmat_analysis_fea::{
    constraints::CanonicalStructuralConstraintPlan, fixtures::mixed_solid_beam_model,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityId,
    PersistentEntityKind, StableDigest,
};

fn entity(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
    PersistentEntityId::new(kind, name.into(), vec!["root".into()])
}

#[test]
fn mixed_compilation_does_not_clone_unrelated_model_payload() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        mesh.topology.nodes[0]
            .provenance
            .push(entity(PersistentEntityKind::Vertex, "beam_root"));
        mesh.topology.nodes[0].provenance.sort();
        for face in &mut mesh.topology.boundary_faces {
            face.provenance
                .push(entity(PersistentEntityKind::Face, "root"));
            face.provenance.sort();
        }
        mesh.canonical_digest = StableDigest::ZERO;
        mesh.seal_canonical_digest().unwrap();
        let mut model =
            mixed_solid_beam_model(mesh.topology.nodes[0].coordinates_m.canonical_value());
        let mut baseline = None;
        for bytes in [8, 1_048_576, 4_194_304] {
            model.geometry_id = "x".repeat(bytes);
            let before = heap::begin_measurement();
            let result = CanonicalStructuralConstraintPlan::compile(&model, &mesh);
            let peak = heap::end_measurement().saturating_sub(before);
            let plan = result.unwrap();
            if let Some((expected_peak, expected_plan)) = &baseline {
                assert_eq!(
                    peak, *expected_peak,
                    "borrowed geometry label must not increase compilation allocation"
                );
                assert_eq!(&plan, expected_plan);
            } else {
                baseline = Some((peak, plan));
            }
            println!("order={order:?} unrelated_label_bytes={bytes} compilation_peak={peak}");
        }
    }
}
