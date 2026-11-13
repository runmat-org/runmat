//! Authored local sizing controls and their resolution into canonical metric contributions.
//!
//! A control names one geometry region (the same named regions constraints and loads select)
//! and requests a local target edge length on it. Controls are part of the request identity;
//! a request without controls encodes exactly as before. Once the geometry is materialized the
//! named regions resolve to persistent entities and every control becomes one isotropic metric
//! contribution per entity, so curves, surfaces and the volume consume local sizing through the
//! one metric field they already share. Contributions combine by the request's most-restrictive
//! intersection: where several controls (or the global size) reach the same mesh entity, the
//! smallest requested size wins.

use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId, PersistentEntityKind};
use serde::{Deserialize, Serialize};

use super::{
    MeshingContractError, MeshingRequest, MetricContribution, MetricContributionScope,
    MetricSourceKind, MetricTensor3,
};

/// Topological kind of the named region a control is applied to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshSizeControlTarget {
    /// A solid region: every cell of the region, and the faces and edges that bound it.
    Solid,
    /// A face region: the face triangles, the edges bounding the faces and the cells touching them.
    Face,
    /// An edge region: the edge segments and every triangle and cell touching the edges.
    Edge,
}

impl MeshSizeControlTarget {
    const fn source(self) -> MetricSourceKind {
        match self {
            Self::Solid => MetricSourceKind::Region,
            Self::Face => MetricSourceKind::Face,
            Self::Edge => MetricSourceKind::Curve,
        }
    }

    const fn entity_kind(self) -> PersistentEntityKind {
        match self {
            Self::Solid => PersistentEntityKind::Solid,
            Self::Face => PersistentEntityKind::Face,
            Self::Edge => PersistentEntityKind::Edge,
        }
    }
}

/// One authored local sizing control on a named geometry region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshSizeControl {
    /// Authored control name, unique within the mesh.
    pub control_id: String,
    /// Named geometry region the control applies to.
    pub region_id: String,
    pub target: MeshSizeControlTarget,
    pub target_size_m: f64,
}

impl MeshSizeControl {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.control_id.trim().is_empty() || self.region_id.trim().is_empty() {
            return Err(MeshingContractError::invalid(
                "mesh size control",
                "control and region names must not be empty",
            ));
        }
        if !self.target_size_m.is_finite() || self.target_size_m <= 0.0 {
            return Err(MeshingContractError::invalid(
                "mesh size control target size",
                format!(
                    "control `{}` must request a finite positive size",
                    self.control_id
                ),
            ));
        }
        Ok(())
    }
}

/// Validates a control list: each control valid, control names unique, at most one control per
/// named region, canonically ordered by control name.
pub fn validate_size_controls(controls: &[MeshSizeControl]) -> Result<(), MeshingContractError> {
    let mut regions = BTreeSet::new();
    for control in controls {
        control.validate()?;
        if !regions.insert(control.region_id.as_str()) {
            return Err(MeshingContractError::invalid(
                "mesh size controls",
                format!(
                    "region `{}` has more than one size control",
                    control.region_id
                ),
            ));
        }
    }
    if controls
        .windows(2)
        .any(|pair| pair[0].control_id >= pair[1].control_id)
    {
        return Err(MeshingContractError::invalid(
            "mesh size controls",
            "control names must be unique and ordered",
        ));
    }
    Ok(())
}

/// Persistent entities of one named geometry region, as resolved against the materialized topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedRegionEntities {
    pub target: MeshSizeControlTarget,
    pub entity_ids: Vec<PersistentEntityId>,
}

/// The persistent entities one control was applied to, in canonical scope form.
#[derive(Debug, Clone, PartialEq)]
pub struct AppliedSizeControl {
    pub control_id: String,
    pub target_size_m: f64,
    /// Contribution scope entities: region identities for solid controls, faces or edges otherwise.
    pub scope_entities: Vec<PersistentEntityId>,
}

impl MeshingRequest {
    /// Resolves the request's authored size controls against the materialized geometry: each
    /// control's named region must exist with the control's target kind, and becomes one isotropic
    /// metric contribution per entity. A request without controls is returned unchanged.
    pub fn resolve_size_controls(
        &self,
        regions: &BTreeMap<String, NamedRegionEntities>,
        topology: &ExactBRepTopology,
    ) -> Result<(Self, Vec<AppliedSizeControl>), MeshingContractError> {
        if self.size_controls.is_empty() {
            return Ok((self.clone(), Vec::new()));
        }
        let region_of_solid = topology
            .regions
            .iter()
            .map(|region| (region.solid_id.clone(), region.id.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut contributions = Vec::new();
        let mut applied = Vec::new();
        for control in &self.size_controls {
            let region = regions.get(&control.region_id).ok_or_else(|| {
                MeshingContractError::invalid(
                    "mesh size control region",
                    format!(
                        "control `{}` names unknown geometry region `{}`",
                        control.control_id, control.region_id
                    ),
                )
            })?;
            if region.target != control.target {
                return Err(MeshingContractError::invalid(
                    "mesh size control region",
                    format!(
                        "control `{}` targets {:?} entities but region `{}` holds {:?} entities",
                        control.control_id, control.target, control.region_id, region.target
                    ),
                ));
            }
            let metric =
                MetricTensor3::isotropic_length_m(control.target_size_m).map_err(|error| {
                    MeshingContractError::invalid("mesh size control", error.to_string())
                })?;
            let mut scope_entities = Vec::with_capacity(region.entity_ids.len());
            for entity in &region.entity_ids {
                if entity.kind != control.target.entity_kind() {
                    return Err(MeshingContractError::invalid(
                        "mesh size control region",
                        format!(
                            "region `{}` resolved to a {:?} entity",
                            control.region_id, entity.kind
                        ),
                    ));
                }
                let scope = match control.target {
                    MeshSizeControlTarget::Solid => {
                        let region_id = region_of_solid.get(entity).ok_or_else(|| {
                            MeshingContractError::invalid(
                                "mesh size control region",
                                format!(
                                    "solid of region `{}` has no meshing region",
                                    control.region_id
                                ),
                            )
                        })?;
                        scope_entities.push(region_id.clone());
                        MetricContributionScope::Region {
                            region_id: region_id.clone(),
                        }
                    }
                    MeshSizeControlTarget::Face | MeshSizeControlTarget::Edge => {
                        scope_entities.push(entity.clone());
                        MetricContributionScope::Entity {
                            entity_id: entity.clone(),
                        }
                    }
                };
                contributions.push(MetricContribution {
                    source: control.target.source(),
                    scope,
                    metric,
                });
            }
            applied.push(AppliedSizeControl {
                control_id: control.control_id.clone(),
                target_size_m: control.target_size_m,
                scope_entities,
            });
        }
        let mut resolved = self.clone();
        resolved.metric = self
            .metric
            .intersect_contributions(contributions)
            .map_err(|error| {
                MeshingContractError::invalid("mesh size controls", error.to_string())
            })?;
        resolved.validate()?;
        Ok((resolved, applied))
    }
}

impl NamedRegionEntities {
    /// Named solid, face and edge regions of a materialized geometry, the regions size controls
    /// may select. Vertex and cell regions are not sizing targets.
    pub fn from_resolved_regions(
        regions: &runmat_geometry_core::ResolvedGeometryRegions,
    ) -> BTreeMap<String, Self> {
        use runmat_geometry_core::GeometryRegionRole;
        regions
            .iter()
            .filter_map(|(name, region)| {
                let target = match region.role {
                    GeometryRegionRole::Solid => MeshSizeControlTarget::Solid,
                    GeometryRegionRole::Face => MeshSizeControlTarget::Face,
                    GeometryRegionRole::Edge => MeshSizeControlTarget::Edge,
                    GeometryRegionRole::Vertex | GeometryRegionRole::Cell => return None,
                };
                Some((
                    name.clone(),
                    Self {
                        target,
                        entity_ids: region.entity_ids.clone(),
                    },
                ))
            })
            .collect()
    }
}

/// Sizing evidence for every applied control: one row per scope entity with the requested size
/// and the achieved corner edge lengths of the mesh entities on it (boundary triangles of a
/// face, boundary segments of an edge, tetrahedra of a region).
pub fn size_control_evidence(
    artifact: &super::SolverMeshArtifact,
    applied: &[AppliedSizeControl],
) -> Vec<super::SizingResolutionEvidence> {
    let topology = &artifact.topology;
    let coordinates = topology
        .nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();
    let length = |a: u64, b: u64| -> Option<f64> {
        let (a, b) = (coordinates.get(&a)?, coordinates.get(&b)?);
        Some(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt())
    };
    let mut rows = Vec::new();
    for control in applied {
        for entity in &control.scope_entities {
            let mut edges = BTreeSet::new();
            let mut add = |corners: &[u64]| {
                for (index, &a) in corners.iter().enumerate() {
                    for &b in &corners[index + 1..] {
                        edges.insert((a.min(b), a.max(b)));
                    }
                }
            };
            match entity.kind {
                PersistentEntityKind::Region => topology
                    .volume_elements
                    .iter()
                    .filter(|element| &element.region_id == entity && element.node_ids.len() >= 4)
                    .for_each(|element| add(&element.node_ids[..4])),
                PersistentEntityKind::Face => topology
                    .boundary_faces
                    .iter()
                    .filter(|face| face.provenance.contains(entity) && face.node_ids.len() >= 3)
                    .for_each(|face| add(&face.node_ids[..3])),
                _ => topology
                    .boundary_edges
                    .iter()
                    .filter(|edge| edge.provenance.contains(entity) && edge.node_ids.len() >= 2)
                    .for_each(|edge| add(&edge.node_ids[..2])),
            }
            let lengths = edges
                .into_iter()
                .filter_map(|(a, b)| length(a, b))
                .collect::<Vec<_>>();
            let achieved = (!lengths.is_empty()).then(|| super::AchievedEdgeLengths {
                edge_count: lengths.len() as u64,
                minimum_m: lengths.iter().copied().fold(f64::INFINITY, f64::min),
                mean_m: lengths.iter().sum::<f64>() / lengths.len() as f64,
                maximum_m: lengths.iter().copied().fold(0.0, f64::max),
            });
            rows.push(super::SizingResolutionEvidence {
                scope: entity.clone(),
                requested_size_m: control.target_size_m,
                resolved_size_m: control.target_size_m,
                achieved_maximum_size_m: achieved.map_or(control.target_size_m, |a| a.maximum_m),
                clipped_contribution_count: 0,
                rejected_contribution_count: 0,
                control_id: Some(control.control_id.clone()),
                achieved,
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{resolve_meshing_request, CanonicalMeshingContract, MeshingRequestSettings};
    use runmat_geometry_core::GeometryTolerancePolicy;

    fn tolerance() -> GeometryTolerancePolicy {
        GeometryTolerancePolicy {
            source_tolerance_m: runmat_types::physical::Length::new(0.0),
            absolute_floor_m: runmat_types::physical::Length::new(1.0e-12),
            model_relative_term: 1.0e-12,
            requested_deviation_m: runmat_types::physical::Length::new(1.0e-4),
            maximum_healing_displacement_m: runmat_types::physical::Length::new(1.0e-6),
        }
    }

    fn id(kind: PersistentEntityKind, name: &str) -> PersistentEntityId {
        PersistentEntityId::new(kind, name.into(), Vec::new())
    }

    fn control(
        name: &str,
        region: &str,
        target: MeshSizeControlTarget,
        size: f64,
    ) -> MeshSizeControl {
        MeshSizeControl {
            control_id: name.into(),
            region_id: region.into(),
            target,
            target_size_m: size,
        }
    }

    fn topology() -> ExactBRepTopology {
        let entity = |kind, name| serde_json::to_value(id(kind, name)).unwrap();
        serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "root_assembly_id": entity(PersistentEntityKind::Assembly, "root"),
            "assemblies": [], "instances": [], "bodies": [], "lumps": [], "solids": [],
            "regions": [{
                "id": entity(PersistentEntityKind::Region, "region-1"),
                "solid_id": entity(PersistentEntityKind::Solid, "solid-1"),
            }],
            "shells": [], "faces": [], "wires": [], "coedges": [], "edges": [], "vertices": [],
        }))
        .unwrap()
    }

    fn request(settings: MeshingRequestSettings) -> MeshingRequest {
        resolve_meshing_request(tolerance(), settings).unwrap()
    }

    #[test]
    fn requests_without_controls_keep_their_identity_and_controls_change_it() {
        let defaults = MeshingRequestSettings::default();
        let plain = request(defaults.clone());
        let encoded = plain.canonical_encode().unwrap();
        // Absent optional fields encode exactly as before the controls existed.
        let value = serde_json::to_value(&plain).unwrap();
        assert!(value.get("size_controls").is_none());
        assert!(value["metric"].get("cells_across_gap").is_none());
        let with = |settings: MeshingRequestSettings| request(settings).canonical_digest().unwrap();
        let face = MeshingRequestSettings {
            size_controls: vec![control(
                "hole",
                "hole_wall",
                MeshSizeControlTarget::Face,
                0.002,
            )],
            ..defaults.clone()
        };
        let finer_face = MeshingRequestSettings {
            size_controls: vec![control(
                "hole",
                "hole_wall",
                MeshSizeControlTarget::Face,
                0.001,
            )],
            ..defaults.clone()
        };
        let gap = MeshingRequestSettings {
            cells_across_gap: Some(3),
            ..defaults.clone()
        };
        let digests = [
            plain.canonical_digest().unwrap(),
            with(face),
            with(finer_face),
            with(gap),
        ];
        for (index, digest) in digests.iter().enumerate() {
            assert!(digests[index + 1..].iter().all(|other| other != digest));
        }
        assert_eq!(request(defaults).canonical_encode().unwrap(), encoded);
    }

    #[test]
    fn controls_reject_non_positive_sizes_conflicts_and_bad_gap_counts() {
        let defaults = MeshingRequestSettings::default();
        for size in [0.0, -1.0, f64::NAN] {
            let settings = MeshingRequestSettings {
                size_controls: vec![control("a", "r", MeshSizeControlTarget::Face, size)],
                ..defaults.clone()
            };
            assert!(settings.validate().is_err());
        }
        let conflict = MeshingRequestSettings {
            size_controls: vec![
                control("a", "r", MeshSizeControlTarget::Face, 0.001),
                control("b", "r", MeshSizeControlTarget::Face, 0.002),
            ],
            ..defaults.clone()
        };
        assert!(conflict.validate().is_err());
        for cells in [0, 65] {
            let settings = MeshingRequestSettings {
                cells_across_gap: Some(cells),
                ..defaults.clone()
            };
            assert!(settings.validate().is_err());
        }
    }

    #[test]
    fn controls_resolve_to_scoped_contributions_on_the_named_entities() {
        let settings = MeshingRequestSettings {
            size_controls: vec![
                control("body", "core", MeshSizeControlTarget::Solid, 0.004),
                control("edge", "rim", MeshSizeControlTarget::Edge, 0.001),
                control("face", "wall", MeshSizeControlTarget::Face, 0.002),
            ],
            ..MeshingRequestSettings::default()
        };
        let request = request(settings);
        let regions = BTreeMap::from([
            (
                "core".to_string(),
                NamedRegionEntities {
                    target: MeshSizeControlTarget::Solid,
                    entity_ids: vec![id(PersistentEntityKind::Solid, "solid-1")],
                },
            ),
            (
                "rim".to_string(),
                NamedRegionEntities {
                    target: MeshSizeControlTarget::Edge,
                    entity_ids: vec![
                        id(PersistentEntityKind::Edge, "e1"),
                        id(PersistentEntityKind::Edge, "e2"),
                    ],
                },
            ),
            (
                "wall".to_string(),
                NamedRegionEntities {
                    target: MeshSizeControlTarget::Face,
                    entity_ids: vec![id(PersistentEntityKind::Face, "f1")],
                },
            ),
        ]);
        let (resolved, applied) = request
            .resolve_size_controls(&regions, &topology())
            .unwrap();
        assert_eq!(resolved.metric.contributions.len(), 4);
        assert!(resolved.metric.contributions.iter().any(|contribution| {
            contribution.source == MetricSourceKind::Region
                && contribution.scope
                    == MetricContributionScope::Region {
                        region_id: id(PersistentEntityKind::Region, "region-1"),
                    }
                && (contribution.metric.xx - 1.0 / (0.004 * 0.004)).abs() < 1.0e-6
        }));
        assert_eq!(applied.len(), 3);
        assert_eq!(applied[1].scope_entities.len(), 2);
        // Resolution is idempotent: re-resolving adds no new contributions.
        let (again, _) = resolved
            .resolve_size_controls(&regions, &topology())
            .unwrap();
        assert_eq!(again, resolved);
        // Unknown regions and kind mismatches are refused.
        let mut missing = regions.clone();
        missing.remove("rim");
        assert!(request
            .resolve_size_controls(&missing, &topology())
            .is_err());
        let mut mismatch = regions.clone();
        mismatch.get_mut("wall").unwrap().target = MeshSizeControlTarget::Edge;
        assert!(request
            .resolve_size_controls(&mismatch, &topology())
            .is_err());
    }
}
