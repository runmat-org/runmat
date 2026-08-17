use runmat_geometry_core::{
    CurveEvaluatorId, MassPropertiesEvaluatorId, PcurveEvaluatorId, PersistentEntityId,
    PersistentEntityKind, SurfaceEvaluatorId, TopologicalOrientation, TrimClassifierId,
};
use sha2::{Digest, Sha256};

use super::ffi::bridge;

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

pub(super) fn shape_id(
    kind: PersistentEntityKind,
    key: u64,
    assembly_path: &[String],
) -> PersistentEntityId {
    scoped_id(kind, &format!("brep-shape:{key:020}"), assembly_path)
}

pub(super) fn optional_shape_id(
    kind: PersistentEntityKind,
    key: u64,
    assembly_path: &[String],
) -> Option<PersistentEntityId> {
    (key != 0).then(|| shape_id(kind, key, assembly_path))
}

pub(super) fn coedge_id(
    wire_key: u64,
    position: u64,
    assembly_path: &[String],
) -> PersistentEntityId {
    scoped_id(
        PersistentEntityKind::Coedge,
        &format!("brep-wire:{wire_key:020}:coedge:{position:020}"),
        assembly_path,
    )
}

pub(super) fn exact_lump_id(
    lump: &bridge::OcctExactLumpPayload,
    assembly_path: &[String],
) -> PersistentEntityId {
    let role = if lump.from_compsolid {
        "compsolid"
    } else {
        "solid"
    };
    scoped_id(
        PersistentEntityKind::Lump,
        &format!("brep-{role}-lump:{:020}", lump.shape_key),
        assembly_path,
    )
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

pub(super) fn body_id(assembly_path: &[String]) -> PersistentEntityId {
    let source = if assembly_path == [ROOT_SCOPE] {
        "body:root"
    } else {
        "body"
    };
    scoped_id(PersistentEntityKind::Body, source, assembly_path)
}

pub(super) fn mass_id(assembly_path: &[String]) -> MassPropertiesEvaluatorId {
    let mut hasher = Sha256::new();
    hasher.update(b"runmat.exact-geometry.mass-properties-scope\0");
    for segment in assembly_path {
        hasher.update((segment.len() as u64).to_be_bytes());
        hasher.update(segment.as_bytes());
    }
    MassPropertiesEvaluatorId(format!("mass:scope:{:x}", hasher.finalize()))
}

pub(super) fn curve_id(key: u64) -> CurveEvaluatorId {
    CurveEvaluatorId(format!("curve:brep-shape:{key:020}"))
}

pub(super) fn pcurve_id(wire_key: u64, position: u64) -> PcurveEvaluatorId {
    PcurveEvaluatorId(format!(
        "pcurve:brep-wire:{wire_key:020}:coedge:{position:020}"
    ))
}

pub(super) fn surface_id(key: u64) -> SurfaceEvaluatorId {
    SurfaceEvaluatorId(format!("surface:brep-shape:{key:020}"))
}

pub(super) fn trim_id(key: u64) -> TrimClassifierId {
    TrimClassifierId(format!("trim:brep-shape:{key:020}"))
}

pub(super) const fn orientation(reversed: bool) -> TopologicalOrientation {
    if reversed {
        TopologicalOrientation::Reversed
    } else {
        TopologicalOrientation::Forward
    }
}
