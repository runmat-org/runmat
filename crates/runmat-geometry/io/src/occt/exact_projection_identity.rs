use runmat_geometry_core::{
    MassPropertiesEvaluatorId, PersistentEntityId, PersistentEntityKind, TopologicalOrientation,
};
use sha2::{Digest, Sha256};

const ROOT_SCOPE: &str = "root";

pub(super) fn scoped_id(
    kind: PersistentEntityKind,
    source_topology_id: &str,
    assembly_path: &[String],
) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source_topology_id.into(),
        assembly_path: assembly_path.to_vec(),
    }
}

pub(super) fn assembly_id(assembly_path: &[String]) -> PersistentEntityId {
    let source = if assembly_path == [ROOT_SCOPE] {
        "assembly:root"
    } else {
        "assembly"
    };
    scoped_id(PersistentEntityKind::Assembly, source, assembly_path)
}

pub(super) fn instance_id(assembly_path: &[String]) -> PersistentEntityId {
    scoped_id(PersistentEntityKind::Instance, "instance", assembly_path)
}

pub(super) fn body_id(assembly_path: &[String], is_sheet_body: bool) -> PersistentEntityId {
    let source = if is_sheet_body {
        "body:sheet"
    } else {
        "body:solid"
    };
    scoped_id(PersistentEntityKind::Body, source, assembly_path)
}

pub(super) fn mass_id(assembly_path: &[String], is_sheet_body: bool) -> MassPropertiesEvaluatorId {
    let mut hasher = Sha256::new();
    hasher.update(b"runmat.exact-geometry.mass-properties-scope\0");
    hasher.update([u8::from(is_sheet_body)]);
    for segment in assembly_path {
        hasher.update((segment.len() as u64).to_be_bytes());
        hasher.update(segment.as_bytes());
    }
    MassPropertiesEvaluatorId(format!("mass:scope:{:x}", hasher.finalize()))
}

pub(super) const fn orientation(reversed: bool) -> TopologicalOrientation {
    if reversed {
        TopologicalOrientation::Reversed
    } else {
        TopologicalOrientation::Forward
    }
}
