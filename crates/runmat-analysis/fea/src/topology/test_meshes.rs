//! Shared meshing-owned canonical fixtures, not a second analysis mesh generator.
pub(crate) use runmat_meshing_core::fixtures::{quadratic_slab_mesh, slab_mesh};
use runmat_meshing_core::{PersistentEntityId, PersistentEntityKind};

pub(crate) fn entity(kind: PersistentEntityKind, source: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source.into(),
        assembly_path: vec!["root".into()],
    }
}
