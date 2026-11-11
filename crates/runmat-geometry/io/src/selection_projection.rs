//! Disposable display projection of exact persistent topology. Display ordinals are only a
//! private join key; every public selectable region uses PersistentEntityId's canonical key.
use std::collections::{BTreeMap, BTreeSet};

use crate::{import::GeometryImportError, ImportedExactCad};
use runmat_geometry_core::{
    EntityIdRange, EntityKind, ExactBRepTopology, GeometryAsset, PersistentEntityId, Region,
    RegionEntityMapping,
};

#[derive(Debug, Clone)]
pub struct ExactSelectionProjection {
    faces: BTreeMap<u64, PersistentEntityId>,
    solids: Vec<(PersistentEntityId, BTreeSet<u64>)>,
}

impl ExactSelectionProjection {
    pub(crate) fn import(
        path: &str,
        bytes: &[u8],
        format: crate::GeometryFormat,
        source_units: Option<runmat_geometry_core::LengthUnit>,
        context: &crate::GeometryImportContext,
    ) -> Result<Self, GeometryImportError> {
        let exact = crate::import_exact_cad(
            path,
            bytes,
            format,
            &crate::ExactCadImportOptions {
                source_units,
                ..Default::default()
            },
            context,
        )?;
        Ok(Self::from_import(&exact))
    }

    pub fn from_import(imported: &ImportedExactCad) -> Self {
        Self::new(&imported.topology, &imported.source_face_ids)
    }

    pub fn new(topology: &ExactBRepTopology, faces: &BTreeMap<u64, PersistentEntityId>) -> Self {
        let ordinals: BTreeMap<_, _> = faces.iter().map(|(i, id)| (id, *i)).collect();
        let shell_faces: BTreeMap<_, BTreeSet<_>> = topology
            .shells
            .iter()
            .map(|shell| {
                (
                    &shell.id,
                    shell
                        .face_uses
                        .iter()
                        .filter_map(|face| ordinals.get(&face.entity_id).copied())
                        .collect(),
                )
            })
            .collect();
        let solids = topology
            .solids
            .iter()
            .map(|solid| {
                let faces = std::iter::once(&solid.outer_shell_id)
                    .chain(&solid.void_shell_ids)
                    .filter_map(|shell| shell_faces.get(shell))
                    .flatten()
                    .copied()
                    .collect();
                (solid.id.clone(), faces)
            })
            .collect();
        Self {
            faces: faces.clone(),
            solids,
        }
    }

    /// Replaces provisional tessellator regions; no source-name or ordinal fallback survives.
    /// A progressive chunk projects only the triangles present in that chunk.
    pub fn bind(&self, asset: &mut GeometryAsset) -> Result<(), GeometryImportError> {
        let mut regions = Vec::new();
        let mut mappings = Vec::new();
        let mut source_regions = BTreeMap::new();
        for region in asset
            .regions
            .iter()
            .filter(|r| r.tag.as_deref() == Some("occt_face"))
        {
            let ordinal = region
                .cad_ownership
                .as_ref()
                .and_then(|o| o.face_id)
                .ok_or_else(|| invalid("display face has no kernel join evidence"))?;
            if !self.faces.contains_key(&ordinal)
                || source_regions.insert(ordinal, region).is_some()
            {
                return Err(invalid(
                    "display face has missing or duplicate exact identity evidence",
                ));
            }
        }
        let mut by_region: BTreeMap<&str, Vec<&RegionEntityMapping>> = BTreeMap::new();
        for mapping in &asset.region_entity_mappings {
            by_region
                .entry(&mapping.region_id)
                .or_default()
                .push(mapping);
        }
        let mut source_ranges: BTreeMap<u64, Vec<RegionEntityMapping>> = BTreeMap::new();
        for (ordinal, region) in &source_regions {
            let entity = &self.faces[ordinal];
            let key = entity.analysis_selection_id().map_err(invalid)?;
            let mut projected = (*region).clone();
            projected.region_id = key.clone();
            projected
                .cad_ownership
                .get_or_insert_with(Default::default)
                .face_id = Some(*ordinal);
            regions.push(projected);
            for mapping in by_region
                .get(region.region_id.as_str())
                .into_iter()
                .flatten()
            {
                source_ranges
                    .entry(*ordinal)
                    .or_default()
                    .push((*mapping).clone());
                let mut projected = (*mapping).clone();
                projected.region_id = key.clone();
                mappings.push(projected);
            }
        }
        for (entity, ordinals) in &self.solids {
            let mut by_mesh: BTreeMap<String, Vec<EntityIdRange>> = BTreeMap::new();
            for ordinal in ordinals {
                for mapping in source_ranges.get(ordinal).into_iter().flatten() {
                    by_mesh
                        .entry(mapping.mesh_id.clone())
                        .or_default()
                        .extend_from_slice(&mapping.ranges);
                }
            }
            if by_mesh.is_empty() {
                continue;
            }
            let key = entity.analysis_selection_id().map_err(invalid)?;
            let metadata = ordinals
                .iter()
                .find_map(|ordinal| source_regions.get(ordinal).copied());
            // Face-local color/material and source-face ordinals never become body evidence.
            let owner_path = metadata
                .and_then(|region| region.cad_ownership.as_ref())
                .map(|ownership| ownership.owner_path.as_slice())
                .unwrap_or_default();
            let label = owner_path
                .iter()
                .rev()
                .find(|label| entity.assembly_path.contains(&label.label_entry))
                .cloned();
            // A body takes its product's name, or its occurrence's when the product is unnamed,
            // as the exact import names solids (`select name` resolves the same either way).
            let product = owner_path
                .iter()
                .rev()
                .find(|label| label.kind == runmat_geometry_core::CadSemanticKind::Body)
                .filter(|label| !label.name.is_empty());
            let name = product
                .or(label.as_ref())
                .filter(|label| !label.name.is_empty())
                .map(|label| label.name.clone())
                .unwrap_or_else(|| "Solid".into());
            let ownership = label.map(|label| runmat_geometry_core::CadRegionOwnership {
                owner_path: vec![label.clone()],
                label: Some(label),
                ..Default::default()
            });
            regions.push(Region {
                region_id: key.clone(),
                name,
                tag: Some("cad_body".into()),
                cad_ownership: ownership,
            });
            for (mesh_id, ranges) in by_mesh {
                mappings.push(RegionEntityMapping::new(
                    key.clone(),
                    mesh_id,
                    EntityKind::Face,
                    merge_ranges(ranges),
                ));
            }
        }
        asset.regions = regions;
        asset.region_entity_mappings = mappings;
        Ok(())
    }
}

fn merge_ranges(mut ranges: Vec<EntityIdRange>) -> Vec<EntityIdRange> {
    ranges.sort_by_key(|r| r.start);
    let mut merged: Vec<EntityIdRange> = Vec::new();
    for range in ranges {
        if let Some(last) = merged.last_mut() {
            if range.start <= last.end_exclusive().unwrap_or(u64::MAX) {
                last.count = last
                    .end_exclusive()
                    .unwrap_or(u64::MAX)
                    .max(range.end_exclusive().unwrap_or(u64::MAX))
                    .saturating_sub(last.start);
                continue;
            }
        }
        merged.push(range);
    }
    merged
}

fn invalid(message: impl std::fmt::Display) -> GeometryImportError {
    GeometryImportError::InvalidGeometry(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::PersistentEntityKind;
    fn entity(kind: PersistentEntityKind, occurrence: &str) -> PersistentEntityId {
        PersistentEntityId::new(
            kind,
            "same-shape".into(),
            vec!["root".into(), occurrence.into()],
        )
    }
    fn asset() -> GeometryAsset {
        let mut asset = crate::import_geometry("triangle.stl", b"solid s\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid s", Default::default()).unwrap().asset;
        asset.regions = vec![Region {
            region_id: "face_000001".into(),
            name: "Mount".into(),
            tag: Some("occt_face".into()),
            cad_ownership: Some(runmat_geometry_core::CadRegionOwnership {
                face_id: Some(0),
                ..Default::default()
            }),
        }];
        asset.region_entity_mappings =
            vec![RegionEntityMapping::all_faces("face_000001", "mesh_1", 1)];
        asset
    }
    #[test]
    fn display_selection_and_exact_selection_have_one_occurrence_qualified_identity() {
        let face = entity(PersistentEntityKind::Face, "part-a");
        let solid = entity(PersistentEntityKind::Solid, "part-a");
        let projection = ExactSelectionProjection {
            faces: BTreeMap::from([(0, face.clone())]),
            solids: vec![(solid.clone(), BTreeSet::from([0]))],
        };
        let mut asset = asset();
        projection.bind(&mut asset).unwrap();
        assert_eq!(asset.regions.len(), 2);
        assert_eq!(
            asset.regions[0].region_id,
            face.analysis_selection_id().unwrap()
        );
        assert_eq!(
            asset.regions[1].region_id,
            solid.analysis_selection_id().unwrap()
        );
        assert_eq!(asset.regions[0].name, "Mount");
        assert!(asset
            .regions
            .iter()
            .all(|region| region.region_id.starts_with("persistent-entity:sha256:")));
        assert!(!entity(PersistentEntityKind::Face, "part-b")
            .matches_analysis_selection(&asset.regions[0].region_id));
        // Rebinding a materialized preview uses source evidence, never a second identity scheme.
        projection.bind(&mut asset).unwrap();
        assert_eq!(asset.regions.len(), 2);
    }
    #[test]
    fn missing_exact_identity_is_rejected_without_mutating_the_asset() {
        let projection = ExactSelectionProjection {
            faces: BTreeMap::new(),
            solids: Vec::new(),
        };
        let mut asset = asset();
        let original = asset.clone();
        assert!(projection.bind(&mut asset).is_err());
        assert_eq!(asset, original);
    }
}
