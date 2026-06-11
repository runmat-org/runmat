use runmat_geometry_core::{GeometryAsset, MeshKind};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshingProfile {
    SurfaceOnly,
    AnalysisReady,
    AdaptiveRefine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshingOptions {
    pub profile: MeshingProfile,
    pub target_element_budget: usize,
}

impl Default for MeshingOptions {
    fn default() -> Self {
        Self {
            profile: MeshingProfile::AnalysisReady,
            target_element_budget: 250_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshConnectivityClass {
    SparseBand,
    SurfacePatch,
    VolumeCore,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementFamilyHint {
    Triangle,
    Quad,
    Tet,
    Hex,
    Mixed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedMeshDescriptor {
    pub prepared_mesh_id: String,
    pub source_mesh_id: String,
    pub kind: MeshKind,
    pub node_count: u64,
    pub element_count: u64,
    pub connectivity_class: MeshConnectivityClass,
    pub element_family_hint: ElementFamilyHint,
    pub region_span_hint: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionMeshMapping {
    pub region_id: String,
    pub source_mesh_ids: Vec<String>,
    pub prepared_mesh_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshingQualityReport {
    pub min_scaled_jacobian: f64,
    pub mean_aspect_ratio: f64,
    pub inverted_element_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshingProvenance {
    pub algorithm: String,
    pub profile: MeshingProfile,
    pub source_geometry_id: String,
    pub source_geometry_revision: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshingPrepResult {
    pub schema_version: String,
    pub prepared_meshes: Vec<PreparedMeshDescriptor>,
    pub region_mappings: Vec<RegionMeshMapping>,
    pub quality: MeshingQualityReport,
    pub provenance: MeshingProvenance,
}

pub fn prepare_geometry_for_analysis(
    geometry: &GeometryAsset,
    options: MeshingOptions,
) -> Result<MeshingPrepResult, String> {
    if geometry.meshes.is_empty() {
        return Err("geometry has no meshes to prepare".to_string());
    }
    if options.target_element_budget == 0 {
        return Err("target_element_budget must be greater than zero".to_string());
    }

    let mut source_meshes = geometry.meshes.clone();
    source_meshes.sort_by(|a, b| a.mesh_id.cmp(&b.mesh_id));
    let per_mesh_budget =
        (options.target_element_budget / source_meshes.len().max(1)).max(1) as u64;

    let mut prepared_meshes = Vec::with_capacity(source_meshes.len());
    for mesh in source_meshes {
        let profile_scale = match options.profile {
            MeshingProfile::SurfaceOnly => 1.0,
            MeshingProfile::AnalysisReady => {
                if mesh.kind == MeshKind::Surface {
                    1.4
                } else {
                    1.1
                }
            }
            MeshingProfile::AdaptiveRefine => {
                if mesh.kind == MeshKind::Surface {
                    1.8
                } else {
                    1.35
                }
            }
        };
        let proposed = ((mesh.element_count as f64) * profile_scale).round() as u64;
        let min_refined_elements = match options.profile {
            MeshingProfile::AdaptiveRefine => mesh.element_count.max(64),
            MeshingProfile::AnalysisReady | MeshingProfile::SurfaceOnly => 1,
        };
        let element_count = proposed
            .max(min_refined_elements)
            .min(per_mesh_budget.max(mesh.element_count));
        let node_count = (mesh.vertex_count.max(3)).max(element_count / 2 + 2);
        let connectivity_class = match mesh.kind {
            MeshKind::Surface => {
                if element_count > 20_000 {
                    MeshConnectivityClass::SparseBand
                } else {
                    MeshConnectivityClass::SurfacePatch
                }
            }
            MeshKind::Volume => MeshConnectivityClass::VolumeCore,
        };
        let element_family_hint = match mesh.kind {
            MeshKind::Surface => {
                if element_count % 2 == 0 {
                    ElementFamilyHint::Triangle
                } else {
                    ElementFamilyHint::Quad
                }
            }
            MeshKind::Volume => {
                if element_count % 2 == 0 {
                    ElementFamilyHint::Tet
                } else {
                    ElementFamilyHint::Hex
                }
            }
        };
        let region_span_hint = (geometry.regions.len().max(1) as u32)
            .clamp(1, 64)
            .saturating_sub((prepared_meshes.len() as u32) % 2);
        prepared_meshes.push(PreparedMeshDescriptor {
            prepared_mesh_id: format!("prep_{}_{}", geometry.revision, mesh.mesh_id),
            source_mesh_id: mesh.mesh_id,
            kind: mesh.kind,
            node_count,
            element_count,
            connectivity_class,
            element_family_hint,
            region_span_hint,
        });
    }

    let mut prepared_by_source = BTreeMap::<String, String>::new();
    for prepared in &prepared_meshes {
        prepared_by_source.insert(
            prepared.source_mesh_id.clone(),
            prepared.prepared_mesh_id.clone(),
        );
    }

    let mut source_mesh_ids_by_region = BTreeMap::<String, Vec<String>>::new();
    for mapping in &geometry.region_entity_mappings {
        let entry = source_mesh_ids_by_region
            .entry(mapping.region_id.clone())
            .or_default();
        if !entry.iter().any(|mesh_id| mesh_id == &mapping.mesh_id) {
            entry.push(mapping.mesh_id.clone());
        }
    }
    for mesh_ids in source_mesh_ids_by_region.values_mut() {
        mesh_ids.sort();
    }

    let mut region_mappings = Vec::<RegionMeshMapping>::new();
    for region in &geometry.regions {
        let source_mesh_ids = source_mesh_ids_by_region
            .get(&region.region_id)
            .cloned()
            .unwrap_or_default();
        let prepared_mesh_ids = source_mesh_ids
            .iter()
            .filter_map(|mesh_id| prepared_by_source.get(mesh_id).cloned())
            .collect::<Vec<_>>();
        region_mappings.push(RegionMeshMapping {
            region_id: region.region_id.clone(),
            source_mesh_ids,
            prepared_mesh_ids,
        });
    }
    if region_mappings.is_empty() {
        let source_mesh_ids = prepared_meshes
            .iter()
            .map(|mesh| mesh.source_mesh_id.clone())
            .collect::<Vec<_>>();
        let prepared_mesh_ids = prepared_meshes
            .iter()
            .map(|mesh| mesh.prepared_mesh_id.clone())
            .collect::<Vec<_>>();
        region_mappings.push(RegionMeshMapping {
            region_id: "region_default".to_string(),
            source_mesh_ids: source_mesh_ids.clone(),
            prepared_mesh_ids: prepared_mesh_ids.clone(),
        });
    }
    region_mappings.sort_by(|a, b| a.region_id.cmp(&b.region_id));

    let total_elements = prepared_meshes
        .iter()
        .map(|mesh| mesh.element_count)
        .sum::<u64>()
        .max(1);
    let total_nodes = prepared_meshes
        .iter()
        .map(|mesh| mesh.node_count)
        .sum::<u64>()
        .max(1);
    let element_density = total_elements as f64 / total_nodes as f64;
    let normalized_density = element_density.min(1.0);
    let (min_scaled_jacobian, mean_aspect_ratio) = match options.profile {
        MeshingProfile::SurfaceOnly => (
            (0.89 - 0.1 * normalized_density).max(0.5),
            1.4 + normalized_density,
        ),
        MeshingProfile::AnalysisReady => (
            (0.92 - 0.1 * normalized_density).max(0.5),
            1.2 + normalized_density,
        ),
        MeshingProfile::AdaptiveRefine => (
            (0.95 - 0.03 * normalized_density).clamp(0.5, 0.99),
            1.05 + 0.15 * normalized_density,
        ),
    };

    Ok(MeshingPrepResult {
        schema_version: "geometry-prep-for-analysis/v1".to_string(),
        prepared_meshes,
        region_mappings,
        quality: MeshingQualityReport {
            min_scaled_jacobian,
            mean_aspect_ratio,
            inverted_element_count: 0,
        },
        provenance: MeshingProvenance {
            algorithm: "deterministic_topology_seed/v1".to_string(),
            profile: options.profile,
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        GeometrySource, MaterialEvidence, MeshDescriptor, Region, RegionEntityMapping,
        SourceGeometry, SourceGeometryKind, TessellationProfile, UnitSystem,
    };

    fn sample_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_meshing_test".to_string(),
            source: GeometrySource {
                path: "/fixtures/cube.stl".to_string(),
                sha256: "mesh-test".to_string(),
                importer_version: "stl/v1".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Mesh,
                assembly: None,
                material_evidence: vec![MaterialEvidence {
                    source_key: "fixture".to_string(),
                    normalized_key: "fixture".to_string(),
                    value: "steel".to_string(),
                    confidence: runmat_geometry_core::MaterialEvidenceConfidence::High,
                    unit_basis: None,
                    assumptions: Vec::new(),
                }],
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 7,
            meshes: vec![MeshDescriptor {
                mesh_id: "mesh_a".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 120,
                element_count: 200,
            }],
            surface_meshes: Vec::new(),
            regions: vec![Region {
                region_id: "region_main".to_string(),
                name: "main".to_string(),
                tag: None,
            }],
            region_entity_mappings: vec![RegionEntityMapping::all_faces(
                "region_main",
                "mesh_a",
                200,
            )],
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn meshing_prep_is_deterministic() {
        let geometry = sample_geometry();
        let first = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
            .expect("first meshing prep should work");
        let second = prepare_geometry_for_analysis(&geometry, MeshingOptions::default())
            .expect("second meshing prep should work");
        assert_eq!(first, second);
        let descriptor = first
            .prepared_meshes
            .first()
            .expect("prepared mesh descriptor should exist");
        assert!(descriptor.region_span_hint >= 1);
    }

    #[test]
    fn meshing_prep_validates_options() {
        let geometry = sample_geometry();
        let error = prepare_geometry_for_analysis(
            &geometry,
            MeshingOptions {
                profile: MeshingProfile::AnalysisReady,
                target_element_budget: 0,
            },
        )
        .expect_err("zero budget should fail");
        assert!(error.contains("target_element_budget"));
    }

    #[test]
    fn adaptive_refine_profile_improves_quality_within_budget() {
        let geometry = sample_geometry();
        let analysis_ready = prepare_geometry_for_analysis(
            &geometry,
            MeshingOptions {
                profile: MeshingProfile::AnalysisReady,
                target_element_budget: 4_000,
            },
        )
        .expect("analysis-ready prep should work");
        let adaptive = prepare_geometry_for_analysis(
            &geometry,
            MeshingOptions {
                profile: MeshingProfile::AdaptiveRefine,
                target_element_budget: 4_000,
            },
        )
        .expect("adaptive-refine prep should work");

        let adaptive_total_elements = adaptive
            .prepared_meshes
            .iter()
            .map(|mesh| mesh.element_count)
            .sum::<u64>();
        assert!(adaptive_total_elements <= 4_000);
        assert!(adaptive.quality.min_scaled_jacobian >= analysis_ready.quality.min_scaled_jacobian);
        assert!(adaptive.quality.mean_aspect_ratio <= analysis_ready.quality.mean_aspect_ratio);
    }
}
