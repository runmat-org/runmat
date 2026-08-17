use runmat_geometry_core::{
    CurveEvaluatorId, PcurveEvaluatorId, PersistentEntityId, PersistentEntityKind,
    SurfaceEvaluatorId, TopologicalOrientation, TrimClassifierId,
};

use super::ffi::bridge;

const ROOT_SCOPE: &str = "root";

pub(super) fn fixed_id(kind: PersistentEntityKind, source_topology_id: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source_topology_id.into(),
        assembly_path: vec![ROOT_SCOPE.into()],
    }
}

pub(super) fn shape_id(kind: PersistentEntityKind, key: u64) -> PersistentEntityId {
    fixed_id(kind, &format!("brep-shape:{key:020}"))
}

pub(super) fn optional_shape_id(
    kind: PersistentEntityKind,
    key: u64,
) -> Option<PersistentEntityId> {
    (key != 0).then(|| shape_id(kind, key))
}

pub(super) fn coedge_id(wire_key: u64, position: u64) -> PersistentEntityId {
    fixed_id(
        PersistentEntityKind::Coedge,
        &format!("brep-wire:{wire_key:020}:coedge:{position:020}"),
    )
}

pub(super) fn exact_lump_id(lump: &bridge::OcctExactLumpPayload) -> PersistentEntityId {
    let role = if lump.from_compsolid {
        "compsolid"
    } else {
        "solid"
    };
    fixed_id(
        PersistentEntityKind::Lump,
        &format!("brep-{role}-lump:{:020}", lump.shape_key),
    )
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
