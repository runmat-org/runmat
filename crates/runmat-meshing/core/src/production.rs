use runmat_geometry_core::GeometryAsset;
use serde::{Deserialize, Serialize};

use crate::{
    curve::{
        discretize_topology_curves, CurveDiscretization, CurveDiscretizationError,
        CurveDiscretizationOptions,
    },
    options::{MeshTargetSize, VolumeMeshingOptions},
    source_topology::{extract_source_topology, SourceTopologyError, SourceTopologyModel},
    surface::{
        discretize_topology_surfaces, SurfaceDiscretization, SurfaceDiscretizationError,
        SurfaceDiscretizationOptions,
    },
    volume_candidate::{
        prepare_volume_candidates, VolumeCandidateError, VolumeCandidateOptions, VolumeCandidateSet,
    },
    AnalysisMeshArtifact,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductionMeshPreparation {
    pub topology: SourceTopologyModel,
    pub curves: CurveDiscretization,
    pub surface: SurfaceDiscretization,
    pub volume_candidates: VolumeCandidateSet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProductionMeshError {
    Topology(SourceTopologyError),
    Curve(CurveDiscretizationError),
    Surface(SurfaceDiscretizationError),
    VolumeCandidate(VolumeCandidateError),
    TetGenerationPending {
        component_count: usize,
        surface_element_count: usize,
        curve_element_count: usize,
    },
}

impl std::fmt::Display for ProductionMeshError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topology(err) => write!(formatter, "source topology extraction failed: {err}"),
            Self::Curve(err) => write!(formatter, "curve discretization failed: {err}"),
            Self::Surface(err) => write!(formatter, "surface discretization failed: {err}"),
            Self::VolumeCandidate(err) => {
                write!(formatter, "volume candidate preparation failed: {err}")
            }
            Self::TetGenerationPending {
                component_count,
                surface_element_count,
                curve_element_count,
            } => write!(
                formatter,
                "production Tet generation is pending after preparing {component_count} volume component(s), {surface_element_count} surface element(s), and {curve_element_count} curve element(s)"
            ),
        }
    }
}

impl std::error::Error for ProductionMeshError {}

pub fn prepare_production_mesh(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
) -> Result<ProductionMeshPreparation, ProductionMeshError> {
    let topology = extract_source_topology(geometry).map_err(ProductionMeshError::Topology)?;
    let curves = discretize_topology_curves(&topology, curve_options_for_mesh(&topology, options))
        .map_err(ProductionMeshError::Curve)?;
    let surface = discretize_topology_surfaces(&topology, SurfaceDiscretizationOptions::default())
        .map_err(ProductionMeshError::Surface)?;
    let volume_candidates = prepare_volume_candidates(&surface, VolumeCandidateOptions::default())
        .map_err(ProductionMeshError::VolumeCandidate)?;

    Ok(ProductionMeshPreparation {
        topology,
        curves,
        surface,
        volume_candidates,
    })
}

pub fn generate_production_analysis_mesh(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, ProductionMeshError> {
    let preparation = prepare_production_mesh(geometry, options)?;
    Err(ProductionMeshError::TetGenerationPending {
        component_count: preparation.volume_candidates.components.len(),
        surface_element_count: preparation.surface.elements.len(),
        curve_element_count: preparation.curves.elements.len(),
    })
}

fn curve_options_for_mesh(
    topology: &SourceTopologyModel,
    options: &VolumeMeshingOptions,
) -> CurveDiscretizationOptions {
    let target_size_m = match options.target_size {
        MeshTargetSize::LengthM(length_m) if length_m.is_finite() && length_m > 0.0 => length_m,
        MeshTargetSize::LengthM(_) | MeshTargetSize::Auto => {
            let span = (0..3)
                .map(|axis| topology.bounds_max_m[axis] - topology.bounds_min_m[axis])
                .fold(0.0_f64, f64::max);
            (span / 20.0).max(1.0e-6)
        }
    };
    CurveDiscretizationOptions {
        target_size_m,
        min_segments_per_edge: 1,
        max_segments_per_edge: options.max_elements.max(1).min(4096),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_geometry_core::{
        EntityIdRange, EntityKind, GeometryAsset, GeometrySource, MeshDescriptor, MeshKind, Region,
        RegionEntityMapping, SourceGeometry, SourceGeometryKind, SurfaceMesh, TessellationProfile,
        UnitSystem,
    };

    #[test]
    fn preparation_runs_topology_curve_surface_and_volume_candidate_stages() {
        let preparation =
            prepare_production_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
                .expect("production preparation should run");

        assert_eq!(preparation.topology.faces.len(), 12);
        assert_eq!(preparation.surface.elements.len(), 12);
        assert_eq!(preparation.volume_candidates.components.len(), 1);
        assert!((preparation.volume_candidates.total_volume_m3 - 1.0).abs() < 1.0e-12);
        assert!(!preparation.curves.elements.is_empty());
    }

    #[test]
    fn production_mesh_fails_at_tet_generation_boundary_for_now() {
        let err =
            generate_production_analysis_mesh(&cube_geometry(), &VolumeMeshingOptions::default())
                .expect_err("Tet generation is not implemented yet");

        match err {
            ProductionMeshError::TetGenerationPending {
                component_count,
                surface_element_count,
                curve_element_count,
            } => {
                assert_eq!(component_count, 1);
                assert_eq!(surface_element_count, 12);
                assert!(curve_element_count > 0);
            }
            other => panic!("unexpected production error: {other:?}"),
        }
    }

    fn cube_geometry() -> GeometryAsset {
        GeometryAsset {
            geometry_id: "geo_production_cube".to_string(),
            source: GeometrySource {
                path: "/fixtures/generic_cube.step".to_string(),
                sha256: "generic-cube".to_string(),
                importer_version: "test".to_string(),
            },
            source_geometry: SourceGeometry {
                kind: SourceGeometryKind::Cad,
                assembly: None,
                material_evidence: Vec::new(),
            },
            tessellation_profile: TessellationProfile::default(),
            units: UnitSystem::Meter,
            revision: 1,
            meshes: vec![MeshDescriptor {
                mesh_id: "cube_surface".to_string(),
                kind: MeshKind::Surface,
                vertex_count: 8,
                element_count: 12,
            }],
            surface_meshes: vec![SurfaceMesh::new(
                "cube_surface",
                vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                vec![
                    [0, 2, 1],
                    [0, 3, 2],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 0, 4],
                    [3, 4, 7],
                ],
            )],
            regions: vec![
                Region {
                    region_id: "root".to_string(),
                    name: "root".to_string(),
                    tag: None,
                    cad_ownership: None,
                },
                Region {
                    region_id: "tip".to_string(),
                    name: "tip".to_string(),
                    tag: None,
                    cad_ownership: None,
                },
            ],
            region_entity_mappings: vec![
                RegionEntityMapping::new(
                    "root",
                    "cube_surface",
                    EntityKind::Face,
                    vec![EntityIdRange::new(0, 6)],
                ),
                RegionEntityMapping::new(
                    "tip",
                    "cube_surface",
                    EntityKind::Face,
                    vec![EntityIdRange::new(6, 6)],
                ),
            ],
            diagnostics: Vec::new(),
        }
    }
}
