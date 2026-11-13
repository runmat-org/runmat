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
    /// Coedge names per occurrence. Public coedge ids are scoped by the occurrence path, so
    /// coincident faces of different parts may share a standalone name; within one occurrence a
    /// repeated name is ambiguous.
    coedge_occurrences: BTreeSet<(u64, String)>,
    ambiguous_coedges: bool,
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
            coedge_occurrences: BTreeSet::new(),
            ambiguous_coedges: false,
            pcurves: BTreeMap::new(),
        };
        let coincident = CoincidentNames::new(payload);
        for (vertex, digest) in payload.vertices.iter().zip(&coincident.vertices) {
            index.insert_shape(
                vertex.occurrence_index,
                PersistentEntityKind::Vertex,
                vertex.shape_key,
                digest,
            )?;
        }
        for (edge, digest) in payload.edges.iter().zip(&coincident.edges) {
            let name = index.insert_shape(
                edge.occurrence_index,
                PersistentEntityKind::Edge,
                edge.shape_key,
                digest,
            )?;
            insert_consistent(
                &mut index.curves,
                edge.shape_key,
                CurveEvaluatorId(format!("curve:{name}")),
                "curve persistent name",
            )?;
        }
        for (face, digest) in payload.faces.iter().zip(&coincident.faces) {
            let name = index.insert_shape(
                face.occurrence_index,
                PersistentEntityKind::Face,
                face.shape_key,
                digest,
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
        for (wire, digest) in payload.wires.iter().zip(&coincident.wires) {
            index.insert_shape(
                wire.occurrence_index,
                PersistentEntityKind::Wire,
                wire.shape_key,
                digest,
            )?;
        }
        for (shell, digest) in payload.shells.iter().zip(&coincident.shells) {
            index.insert_shape(
                shell.occurrence_index,
                PersistentEntityKind::Shell,
                shell.shape_key,
                digest,
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
            let name = format!("occt:{}", hex::encode(hasher.finalize()));
            if !self
                .coedge_occurrences
                .insert((coedge.occurrence_index, name.clone()))
            {
                self.ambiguous_coedges = true;
            }
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
                return Err(invalid(format!(
                    "OCCT topology contains ambiguous coincident persistent names: two distinct {kind:?} entities serialize identically"
                )));
            }
        }
        if self.ambiguous_coedges {
            return Err(invalid("OCCT coedge persistent names are ambiguous"));
        }
        Ok(())
    }
}

/// Content digests made unique within each occurrence.
///
/// A shape's standalone digest is its serialized content, so two distinct shapes with the same
/// content collide: two vertices at one point (edge loops touching at a point), or the coincident
/// faces of two solids that a STEP writer stores as separate copies (a part and the cavity of the
/// air around it). Such shapes differ in where they are used, so a colliding shape is named by
/// its digest together with the sorted names of the shapes that use it, resolved top-down:
/// shells by their solids, faces by their shells, wires by their face, edges by the faces whose
/// boundaries use them, and vertices by their edges. A shape with a unique digest keeps it.
struct CoincidentNames {
    vertices: Vec<Vec<u8>>,
    edges: Vec<Vec<u8>>,
    faces: Vec<Vec<u8>>,
    wires: Vec<Vec<u8>>,
    shells: Vec<Vec<u8>>,
}

impl CoincidentNames {
    fn new(payload: &bridge::OcctExactShapePayload) -> Self {
        let mut solid_parents = BTreeMap::<(u64, u64), Vec<Vec<u8>>>::new();
        for solid in &payload.solids {
            for shell in std::iter::once(&solid.outer_shell_key).chain(&solid.void_shell_keys) {
                solid_parents
                    .entry((solid.occurrence_index, *shell))
                    .or_default()
                    .push(solid.identity_digest.clone());
            }
        }
        let shells = refine(
            b"runmat.occt-coincident-shell-name\0",
            payload.shells.iter().map(|shell| {
                (
                    shell.occurrence_index,
                    shell.shape_key,
                    &shell.identity_digest,
                )
            }),
            &solid_parents,
        );
        let mut shell_parents = BTreeMap::<(u64, u64), Vec<Vec<u8>>>::new();
        for (shell, digest) in payload.shells.iter().zip(&shells) {
            for face in &shell.face_keys {
                shell_parents
                    .entry((shell.occurrence_index, *face))
                    .or_default()
                    .push(digest.clone());
            }
        }
        let faces = refine(
            b"runmat.occt-coincident-face-name\0",
            payload
                .faces
                .iter()
                .map(|face| (face.occurrence_index, face.shape_key, &face.identity_digest)),
            &shell_parents,
        );
        let face_digest = payload
            .faces
            .iter()
            .zip(&faces)
            .map(|(face, digest)| ((face.occurrence_index, face.shape_key), digest.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut wire_parents = BTreeMap::<(u64, u64), Vec<Vec<u8>>>::new();
        for wire in &payload.wires {
            if let Some(digest) = face_digest.get(&(wire.occurrence_index, wire.face_key)) {
                wire_parents
                    .entry((wire.occurrence_index, wire.shape_key))
                    .or_default()
                    .push(digest.clone());
            }
        }
        let wires = refine(
            b"runmat.occt-coincident-wire-name\0",
            payload
                .wires
                .iter()
                .map(|wire| (wire.occurrence_index, wire.shape_key, &wire.identity_digest)),
            &wire_parents,
        );
        let mut edge_parents = BTreeMap::<(u64, u64), Vec<Vec<u8>>>::new();
        for coedge in &payload.coedges {
            if let Some(digest) = face_digest.get(&(coedge.occurrence_index, coedge.face_key)) {
                edge_parents
                    .entry((coedge.occurrence_index, coedge.edge_key))
                    .or_default()
                    .push(digest.clone());
            }
        }
        let edges = refine(
            b"runmat.occt-coincident-edge-name\0",
            payload
                .edges
                .iter()
                .map(|edge| (edge.occurrence_index, edge.shape_key, &edge.identity_digest)),
            &edge_parents,
        );
        let mut vertex_parents = BTreeMap::<(u64, u64), Vec<Vec<u8>>>::new();
        for (edge, digest) in payload.edges.iter().zip(&edges) {
            for key in [edge.start_vertex_key, edge.end_vertex_key] {
                if key != 0 {
                    vertex_parents
                        .entry((edge.occurrence_index, key))
                        .or_default()
                        .push(digest.clone());
                }
            }
        }
        let vertices = refine(
            b"runmat.occt-coincident-vertex-name\0",
            payload.vertices.iter().map(|vertex| {
                (
                    vertex.occurrence_index,
                    vertex.shape_key,
                    &vertex.identity_digest,
                )
            }),
            &vertex_parents,
        );
        Self {
            vertices,
            edges,
            faces,
            wires,
            shells,
        }
    }
}

fn refine<'a>(
    domain: &[u8],
    shapes: impl Iterator<Item = (u64, u64, &'a Vec<u8>)> + Clone,
    parents: &BTreeMap<(u64, u64), Vec<Vec<u8>>>,
) -> Vec<Vec<u8>> {
    let mut counts = BTreeMap::<(u64, &[u8]), usize>::new();
    for (occurrence, _, digest) in shapes.clone() {
        *counts.entry((occurrence, digest.as_slice())).or_default() += 1;
    }
    shapes
        .map(|(occurrence, key, digest)| {
            let users = parents.get(&(occurrence, key));
            if counts[&(occurrence, digest.as_slice())] < 2 || users.is_none() {
                // Unique, or nothing distinguishes it: the uniqueness check reports the latter.
                return digest.clone();
            }
            let mut users = users.cloned().unwrap_or_default();
            users.sort_unstable();
            users.dedup();
            let mut hasher = Sha256::new();
            hasher.update(domain);
            hasher.update(digest);
            for user in users {
                hasher.update((user.len() as u64).to_be_bytes());
                hasher.update(user);
            }
            hasher.finalize().to_vec()
        })
        .collect()
}

pub(super) fn digest_name(digest: &[u8]) -> Result<String, GeometryImportError> {
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
