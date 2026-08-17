//! Persistent semantic naming for imported OCCT topology.
//!
//! Kernel shape-table positions remain private evaluator handles. Public entity/evaluator names
//! use standalone canonical subshape digests plus occurrence paths, so import traversal and
//! compound child order cannot change semantic identity. Coincident indistinguishable topology is
//! rejected as ambiguous until deterministic healing provides an explicit consolidation map.

use std::collections::{btree_map::Entry, BTreeMap, BTreeSet};
use std::fmt::Write as _;

use runmat_geometry_core::{
    CurveEvaluatorId, PcurveEvaluatorId, PersistentEntityId, PersistentEntityKind,
    SurfaceEvaluatorId, TrimClassifierId,
};
use sha2::{Digest, Sha256};

use super::ffi::bridge;
use crate::import::GeometryImportError;

pub(super) struct PersistentNameIndex {
    shapes: BTreeMap<(u64, PersistentEntityKind, u64), String>,
    curves: BTreeMap<u64, CurveEvaluatorId>,
    surfaces: BTreeMap<u64, SurfaceEvaluatorId>,
    trims: BTreeMap<u64, TrimClassifierId>,
    coedges: BTreeMap<(u64, u64), String>,
    pcurves: BTreeMap<(u64, u64), PcurveEvaluatorId>,
}

impl PersistentNameIndex {
    pub fn new(payload: &bridge::OcctExactShapePayload) -> Result<Self, GeometryImportError> {
        let mut index = Self {
            shapes: BTreeMap::new(),
            curves: BTreeMap::new(),
            surfaces: BTreeMap::new(),
            trims: BTreeMap::new(),
            coedges: BTreeMap::new(),
            pcurves: BTreeMap::new(),
        };
        for vertex in &payload.vertices {
            index.insert_shape(
                vertex.occurrence_index,
                PersistentEntityKind::Vertex,
                vertex.shape_key,
                &vertex.identity_digest,
            )?;
        }
        for edge in &payload.edges {
            let name = index.insert_shape(
                edge.occurrence_index,
                PersistentEntityKind::Edge,
                edge.shape_key,
                &edge.identity_digest,
            )?;
            insert_consistent(
                &mut index.curves,
                edge.shape_key,
                CurveEvaluatorId(format!("curve:{name}")),
                "curve persistent name",
            )?;
        }
        for face in &payload.faces {
            let name = index.insert_shape(
                face.occurrence_index,
                PersistentEntityKind::Face,
                face.shape_key,
                &face.identity_digest,
            )?;
            insert_consistent(
                &mut index.surfaces,
                face.shape_key,
                SurfaceEvaluatorId(format!("surface:{name}")),
                "surface persistent name",
            )?;
            insert_consistent(
                &mut index.trims,
                face.shape_key,
                TrimClassifierId(format!("trim:{name}")),
                "trim persistent name",
            )?;
        }
        for wire in &payload.wires {
            index.insert_shape(
                wire.occurrence_index,
                PersistentEntityKind::Wire,
                wire.shape_key,
                &wire.identity_digest,
            )?;
        }
        for shell in &payload.shells {
            index.insert_shape(
                shell.occurrence_index,
                PersistentEntityKind::Shell,
                shell.shape_key,
                &shell.identity_digest,
            )?;
        }
        for solid in &payload.solids {
            index.insert_shape(
                solid.occurrence_index,
                PersistentEntityKind::Solid,
                solid.shape_key,
                &solid.identity_digest,
            )?;
        }
        for lump in &payload.lumps {
            index.insert_shape(
                lump.occurrence_index,
                PersistentEntityKind::Lump,
                lump.shape_key,
                &lump.identity_digest,
            )?;
        }
        index.insert_coedges(payload)?;
        index.require_unique_names()?;
        Ok(index)
    }

    pub fn shape_id(
        &self,
        kind: PersistentEntityKind,
        key: u64,
        occurrence_index: u64,
        assembly_path: &[String],
    ) -> Result<PersistentEntityId, GeometryImportError> {
        let name = self
            .shapes
            .get(&(occurrence_index, kind, key))
            .ok_or_else(|| invalid("exact topology references an unnamed OCCT shape"))?;
        Ok(super::exact_projection_identity::scoped_id(
            kind,
            name,
            assembly_path,
        ))
    }

    pub fn optional_shape_id(
        &self,
        kind: PersistentEntityKind,
        key: u64,
        occurrence_index: u64,
        assembly_path: &[String],
    ) -> Result<Option<PersistentEntityId>, GeometryImportError> {
        (key != 0)
            .then(|| self.shape_id(kind, key, occurrence_index, assembly_path))
            .transpose()
    }

    pub fn lump_id(
        &self,
        lump: &bridge::OcctExactLumpPayload,
        assembly_path: &[String],
    ) -> Result<PersistentEntityId, GeometryImportError> {
        self.shape_id(
            PersistentEntityKind::Lump,
            lump.shape_key,
            lump.occurrence_index,
            assembly_path,
        )
    }

    pub fn coedge_id(
        &self,
        wire_key: u64,
        position: u64,
        assembly_path: &[String],
    ) -> Result<PersistentEntityId, GeometryImportError> {
        let name = self
            .coedges
            .get(&(wire_key, position))
            .ok_or_else(|| invalid("exact topology references an unnamed OCCT coedge"))?;
        Ok(super::exact_projection_identity::scoped_id(
            PersistentEntityKind::Coedge,
            name,
            assembly_path,
        ))
    }

    pub fn curve_id(&self, key: u64) -> Result<CurveEvaluatorId, GeometryImportError> {
        self.curves
            .get(&key)
            .cloned()
            .ok_or_else(|| invalid("exact edge has no persistent evaluator name"))
    }

    pub fn surface_id(&self, key: u64) -> Result<SurfaceEvaluatorId, GeometryImportError> {
        self.surfaces
            .get(&key)
            .cloned()
            .ok_or_else(|| invalid("exact face has no persistent surface name"))
    }

    pub fn trim_id(&self, key: u64) -> Result<TrimClassifierId, GeometryImportError> {
        self.trims
            .get(&key)
            .cloned()
            .ok_or_else(|| invalid("exact face has no persistent trim name"))
    }

    pub fn pcurve_id(
        &self,
        wire_key: u64,
        position: u64,
    ) -> Result<PcurveEvaluatorId, GeometryImportError> {
        self.pcurves
            .get(&(wire_key, position))
            .cloned()
            .ok_or_else(|| invalid("exact coedge has no persistent pcurve name"))
    }

    fn insert_shape(
        &mut self,
        occurrence_index: u64,
        kind: PersistentEntityKind,
        key: u64,
        digest: &[u8],
    ) -> Result<String, GeometryImportError> {
        let name = digest_name(digest)?;
        let map_key = (occurrence_index, kind, key);
        match self.shapes.insert(map_key, name.clone()) {
            Some(existing) if existing != name => {
                Err(invalid("OCCT shape has conflicting persistent names"))
            }
            Some(_) => Err(invalid("OCCT shape persistent name is duplicated")),
            None => Ok(name),
        }
    }

    fn insert_coedges(
        &mut self,
        payload: &bridge::OcctExactShapePayload,
    ) -> Result<(), GeometryImportError> {
        for coedge in &payload.coedges {
            let face = self.shape_name(
                coedge.occurrence_index,
                PersistentEntityKind::Face,
                coedge.face_key,
            )?;
            let wire = self.shape_name(
                coedge.occurrence_index,
                PersistentEntityKind::Wire,
                coedge.wire_key,
            )?;
            let edge = self.shape_name(
                coedge.occurrence_index,
                PersistentEntityKind::Edge,
                coedge.edge_key,
            )?;
            let mut hasher = Sha256::new();
            hasher.update(b"runmat.occt-persistent-coedge-name\0");
            for component in [face, wire, edge] {
                hasher.update((component.len() as u64).to_be_bytes());
                hasher.update(component.as_bytes());
            }
            hasher.update([u8::from(coedge.reversed), coedge.seam_image as u8]);
            let name = format!("occt:{:x}", hasher.finalize());
            insert_consistent(
                &mut self.coedges,
                (coedge.wire_key, coedge.coedge_key),
                name.clone(),
                "coedge persistent name",
            )?;
            insert_consistent(
                &mut self.pcurves,
                (coedge.wire_key, coedge.coedge_key),
                PcurveEvaluatorId(format!("pcurve:{name}")),
                "pcurve persistent name",
            )?;
        }
        Ok(())
    }

    fn shape_name(
        &self,
        occurrence_index: u64,
        kind: PersistentEntityKind,
        key: u64,
    ) -> Result<&str, GeometryImportError> {
        self.shapes
            .get(&(occurrence_index, kind, key))
            .map(String::as_str)
            .ok_or_else(|| invalid("OCCT persistent name incidence is incomplete"))
    }

    fn require_unique_names(&self) -> Result<(), GeometryImportError> {
        let mut names = BTreeSet::new();
        for ((occurrence, kind, _), name) in &self.shapes {
            if !names.insert((*occurrence, *kind, name)) {
                return Err(invalid(
                    "OCCT topology contains ambiguous coincident persistent names",
                ));
            }
        }
        let mut coedges = BTreeSet::new();
        for name in self.coedges.values() {
            if !coedges.insert(name) {
                return Err(invalid("OCCT coedge persistent names are ambiguous"));
            }
        }
        Ok(())
    }
}

fn digest_name(digest: &[u8]) -> Result<String, GeometryImportError> {
    if digest.len() != 32 || digest.iter().all(|byte| *byte == 0) {
        return Err(invalid("OCCT persistent shape digest is malformed"));
    }
    let mut name = String::with_capacity(5 + digest.len() * 2);
    name.push_str("occt:");
    for byte in digest {
        write!(&mut name, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(name)
}

fn insert_consistent<K: Ord, V: PartialEq>(
    map: &mut BTreeMap<K, V>,
    key: K,
    value: V,
    role: &str,
) -> Result<(), GeometryImportError> {
    match map.entry(key) {
        Entry::Vacant(entry) => {
            entry.insert(value);
            Ok(())
        }
        Entry::Occupied(entry) if entry.get() == &value => Ok(()),
        Entry::Occupied(_) => Err(invalid(format!("{role} conflicts across occurrences"))),
    }
}

fn invalid(reason: impl Into<String>) -> GeometryImportError {
    GeometryImportError::InvalidGeometry(reason.into())
}
