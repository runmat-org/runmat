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
    /// The CAD author's name for each named face and solid, by analysis selection ID, when the
    /// display tessellates a representation that carries none of its own (a materialized B-rep).
    authored_names: Option<BTreeMap<String, String>>,
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
            authored_names: None,
        }
    }

    /// Names the display's faces and bodies from the exact product's authored names (each
    /// entity's first name in `ImportedExactCad::entity_names`) instead of the display import's
    /// CAD ownership. Use it when the display reads a representation without CAD names, such as
    /// the kernel B-rep of a materialized study geometry.
    pub fn with_authored_names(mut self, names: BTreeMap<String, String>) -> Self {
        self.authored_names = Some(names);
        self
    }

    /// A face's CAD ownership from the authored names: its own name, if the author gave it one,
    /// and the name of the body (solid) that holds it, as the display import's CAD ownership
    /// lists them (owners outermost first, the face last).
    fn authored_face_ownership(
        &self,
        names: &BTreeMap<String, String>,
        ordinal: u64,
        face_key: &str,
    ) -> Result<runmat_geometry_core::CadRegionOwnership, GeometryImportError> {
        use runmat_geometry_core::{CadLabelRef, CadRegionOwnership, CadSemanticKind};
        let first = |key: &str| {
            names
                .get(key)
                .filter(|name| !name.trim().is_empty())
                .cloned()
        };
        let mut owner_path = Vec::new();
        for (solid, ordinals) in &self.solids {
            if !ordinals.contains(&ordinal) {
                continue;
            }
            let key = solid.analysis_selection_id().map_err(invalid)?;
            if let Some(name) = first(&key) {
                owner_path.push(CadLabelRef {
                    label_entry: key,
                    name,
                    kind: CadSemanticKind::Body,
                });
            }
        }
        let label = first(face_key).map(|name| CadLabelRef {
            label_entry: face_key.to_string(),
            name,
            kind: CadSemanticKind::Face,
        });
        owner_path.extend(label.clone());
        Ok(CadRegionOwnership {
            face_id: Some(ordinal),
            label,
            owner_path,
            ..Default::default()
        })
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
            if !self.faces.contains_key(&ordinal) {
                return Err(invalid(format!(
                    "display face {ordinal} has no exact face among the {} source faces",
                    self.faces.len()
                )));
            }
            if source_regions.insert(ordinal, region).is_some() {
                return Err(invalid(format!(
                    "display face {ordinal} is claimed by two display regions"
                )));
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
        let mut projected_faces: BTreeMap<u64, Region> = BTreeMap::new();
        for (ordinal, region) in &source_regions {
            let entity = &self.faces[ordinal];
            let key = entity.analysis_selection_id().map_err(invalid)?;
            let mut projected = (*region).clone();
            projected.region_id = key.clone();
            if let Some(names) = &self.authored_names {
                let ownership = self.authored_face_ownership(names, *ordinal, &key)?;
                if let Some(label) = &ownership.label {
                    projected.name = label.name.clone();
                }
                projected.cad_ownership = Some(ownership);
            } else {
                projected
                    .cad_ownership
                    .get_or_insert_with(Default::default)
                    .face_id = Some(*ordinal);
            }
            projected_faces.insert(*ordinal, projected.clone());
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
                .find_map(|ordinal| projected_faces.get(ordinal));
            // Face-local color/material and source-face ordinals never become body evidence.
            let owner_path = metadata
                .and_then(|region| region.cad_ownership.as_ref())
                .map(|ownership| ownership.owner_path.as_slice())
                .unwrap_or_default();
            // An owner is this solid's label when it is the solid's occurrence or the solid itself.
            let label = owner_path
                .iter()
                .rev()
                .find(|label| {
                    entity.assembly_path.contains(&label.label_entry) || label.label_entry == key
                })
                .cloned();
            // A body takes its own authored name, else its product's, else its occurrence's when
            // the product is unnamed, as the exact import names solids (`select name` resolves
            // the same either way).
            let product = owner_path
                .iter()
                .find(|label| label.label_entry == key)
                .or_else(|| {
                    owner_path
                        .iter()
                        .rev()
                        .find(|label| label.kind == runmat_geometry_core::CadSemanticKind::Body)
                })
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
            authored_names: None,
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
            authored_names: None,
        };
        let mut asset = asset();
        let original = asset.clone();
        let error = projection.bind(&mut asset).unwrap_err().to_string();
        assert!(
            error.contains("display face 0 has no exact face"),
            "{error}"
        );
        assert_eq!(asset, original);
    }
    #[test]
    fn a_source_face_claimed_twice_is_rejected_by_ordinal() {
        let projection = ExactSelectionProjection {
            faces: BTreeMap::from([(0, entity(PersistentEntityKind::Face, "part-a"))]),
            solids: Vec::new(),
            authored_names: None,
        };
        let mut asset = asset();
        let mut twin = asset.regions[0].clone();
        twin.region_id = "face_000002".into();
        asset.regions.push(twin);
        let error = projection.bind(&mut asset).unwrap_err().to_string();
        assert!(
            error.contains("display face 0 is claimed by two display regions"),
            "{error}"
        );
    }
    /// A display without CAD names of its own (a materialized B-rep) names its faces and bodies
    /// from the exact product: an unnamed face keeps its number and names its owning body, as
    /// Measure reads it ("Face 1 of WallHook_30mm"); a named face reads its CAD name.
    #[test]
    fn authored_names_name_faces_and_their_owning_body() {
        use runmat_geometry_core::CadSemanticKind;
        let face = entity(PersistentEntityKind::Face, "part-a");
        let solid = entity(PersistentEntityKind::Solid, "part-a");
        let face_key = face.analysis_selection_id().unwrap();
        let solid_key = solid.analysis_selection_id().unwrap();
        let mut asset = asset();
        asset.regions[0].name = "Face 1".into();
        let projection = ExactSelectionProjection {
            faces: BTreeMap::from([(0, face.clone())]),
            solids: vec![(solid.clone(), BTreeSet::from([0]))],
            authored_names: None,
        }
        .with_authored_names(BTreeMap::from([(
            solid_key.clone(),
            "WallHook_30mm".to_string(),
        )]));
        projection.bind(&mut asset).unwrap();
        let ownership = asset.regions[0].cad_ownership.clone().unwrap();
        assert_eq!(asset.regions[0].name, "Face 1");
        assert_eq!(ownership.face_id, Some(0));
        assert_eq!(ownership.label, None);
        assert_eq!(ownership.owner_path.len(), 1);
        assert_eq!(ownership.owner_path[0].name, "WallHook_30mm");
        assert_eq!(ownership.owner_path[0].kind, CadSemanticKind::Body);
        assert_eq!(ownership.owner_path[0].label_entry, solid_key);
        // The body region takes the same name and is its own label.
        assert_eq!(asset.regions[1].name, "WallHook_30mm");
        let body = asset.regions[1].cad_ownership.clone().unwrap();
        assert_eq!(body.label.unwrap().label_entry, solid_key);

        // A face the author named reads that name, its body still owning it.
        let mut asset = self::asset();
        let projection = projection.with_authored_names(BTreeMap::from([
            (solid_key.clone(), "WallHook_30mm".to_string()),
            (face_key.clone(), "Hook seat".to_string()),
        ]));
        projection.bind(&mut asset).unwrap();
        let ownership = asset.regions[0].cad_ownership.clone().unwrap();
        assert_eq!(asset.regions[0].name, "Hook seat");
        let label = ownership.label.unwrap();
        assert_eq!(
            (label.name.as_str(), label.kind),
            ("Hook seat", CadSemanticKind::Face)
        );
        let path: Vec<_> = ownership
            .owner_path
            .iter()
            .map(|l| l.name.as_str())
            .collect();
        assert_eq!(path, ["WallHook_30mm", "Hook seat"]);
    }
}
