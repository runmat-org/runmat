use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use runmat_meshing_core::{
    artifact::{
        AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode, AnalysisVolumeElement,
    },
    backend::MeshBackendKind,
    boundary::{BoundaryMeshInput, BoundaryMeshInputError},
    options::{MeshKindRequest, MeshProfile, MeshTargetSize, VolumeMeshingOptions},
    predicate::tetrahedron_scaled_jacobian,
    provenance::{MeshEntityProvenance, SourceEntityKind},
    quality::boundary::{
        evaluate_boundary_quality_candidate, BoundaryQualityCandidateConstraints,
        BoundaryQualityCandidateOptions,
    },
    quality::{AnalysisMeshQualityReport, ElementQuality, QualityThresholds},
    size::field::{
        AnisotropicSizingSample, MeshSizingField, SizingSample, SizingSampleApplication,
        SizingSampleRejection,
    },
    topology::{BoundaryElementKind, VolumeElementKind},
};

pub trait VolumeMesher {
    fn mesh(
        &self,
        input: &BoundaryMeshInput,
        options: &VolumeMeshingOptions,
    ) -> Result<AnalysisMeshArtifact, MeshingError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshingError {
    BoundaryInput(String),
    UnsupportedMeshKind(MeshKindRequest),
    UnsupportedElementKind(VolumeElementKind),
    InvalidElementBudget,
    InvalidTargetSize,
    EmptyBoundaryRegions,
    UnsupportedBackend(MeshBackendKind),
}

impl std::fmt::Display for MeshingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoundaryInput(message) => write!(formatter, "invalid boundary mesh: {message}"),
            Self::UnsupportedMeshKind(kind) => {
                write!(formatter, "unsupported analysis mesh kind: {kind:?}")
            }
            Self::UnsupportedElementKind(kind) => {
                write!(formatter, "unsupported volume element kind: {kind:?}")
            }
            Self::InvalidElementBudget => write!(formatter, "max_elements must be greater than 0"),
            Self::InvalidTargetSize => {
                write!(
                    formatter,
                    "target_size must be auto or a finite positive length"
                )
            }
            Self::EmptyBoundaryRegions => write!(formatter, "boundary mesh has no region ids"),
            Self::UnsupportedBackend(backend) => {
                write!(
                    formatter,
                    "structured Tetrahedron meshing does not own backend {backend:?}"
                )
            }
        }
    }
}

impl std::error::Error for MeshingError {}

impl From<BoundaryMeshInputError> for MeshingError {
    fn from(value: BoundaryMeshInputError) -> Self {
        Self::BoundaryInput(value.to_string())
    }
}

mod sizing;
use sizing::{append_geometry_focus_sizing_samples, structured_grid};

mod metrics;
#[cfg(test)]
use metrics::element_quality_for_nodes;
use metrics::{
    boundary_projection_errors, orient_tetrahedron, project_boundary_nodes_if_quality_improves,
    quality_report, tetrahedron_aspect_ratio, tetrahedron_points, tetrahedron_volume,
};

mod boundary;
#[cfg(test)]
use boundary::{
    boundary_triangle_centroid_cells, largest_connected_occupied_component,
    point_inside_closed_surface,
};
use boundary::{grid_boundary_faces, occupied_cells};

mod geometry;
use geometry::{
    add, boundary_max_span, cross, distance, dot, lerp, midpoint, norm, scale, sub,
    triangle_centroid, triangle_edges, triangle_min_edge, triangle_unit_normal, triangle_vertices,
};

mod grid;
use grid::{grid_nodes, node_id_at, StructuredGrid};

mod generate;
use generate::generate_grid_tetrahedra;

mod artifact;
use artifact::build_analysis_mesh_artifact;

mod validation;
use validation::{
    validate_boundary_regions, validate_structured_meshing_input, validate_volume_meshing_options,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct StructuredTetrahedronMesher;

impl VolumeMesher for StructuredTetrahedronMesher {
    fn mesh(
        &self,
        input: &BoundaryMeshInput,
        options: &VolumeMeshingOptions,
    ) -> Result<AnalysisMeshArtifact, MeshingError> {
        self.mesh_with_sizing(input, options, None)
    }
}

impl StructuredTetrahedronMesher {
    pub fn mesh_with_sizing(
        &self,
        input: &BoundaryMeshInput,
        options: &VolumeMeshingOptions,
        sizing: Option<&MeshSizingField>,
    ) -> Result<AnalysisMeshArtifact, MeshingError> {
        validate_structured_meshing_input(input, options)?;

        let mut mesh_sizing = sizing.cloned().unwrap_or_default();
        append_geometry_focus_sizing_samples(input, options, &mut mesh_sizing);

        let grid = structured_grid(input, options, Some(&mut mesh_sizing))?;
        let nodes = grid_nodes(&grid);
        let occupied_cells = occupied_cells(input, &grid);
        let generated_tetrahedra = generate_grid_tetrahedra(input, &grid, &nodes, &occupied_cells);

        let boundary_faces = grid_boundary_faces(
            input,
            &grid,
            &occupied_cells,
            &generated_tetrahedra.cell_tetrahedron_ids,
            &|i, j, k| node_id_at(&grid, i, j, k),
        );
        let original_quality = quality_report(
            generated_tetrahedra.element_quality,
            boundary_projection_errors(input, &boundary_faces, &nodes),
        );
        let (nodes, quality) = project_boundary_nodes_if_quality_improves(
            input,
            nodes,
            &generated_tetrahedra.volume_elements,
            &boundary_faces,
            original_quality,
        );
        Ok(build_analysis_mesh_artifact(
            input,
            options,
            &grid,
            nodes,
            generated_tetrahedra.volume_elements,
            boundary_faces,
            quality,
            mesh_sizing,
        ))
    }
}

pub fn generate_analysis_mesh(
    geometry: &runmat_geometry_core::GeometryAsset,
    options: VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, MeshingError> {
    validate_structured_backend(options.backend)?;
    validate_volume_meshing_options(&options)?;
    let input = BoundaryMeshInput::from_geometry(geometry)?;
    validate_boundary_regions(&input)?;
    StructuredTetrahedronMesher.mesh(&input, &options)
}

pub fn generate_analysis_mesh_with_sizing(
    geometry: &runmat_geometry_core::GeometryAsset,
    options: VolumeMeshingOptions,
    sizing: &MeshSizingField,
) -> Result<AnalysisMeshArtifact, MeshingError> {
    validate_structured_backend(options.backend)?;
    validate_volume_meshing_options(&options)?;
    let input = BoundaryMeshInput::from_geometry(geometry)?;
    validate_boundary_regions(&input)?;
    StructuredTetrahedronMesher.mesh_with_sizing(&input, &options, Some(sizing))
}

fn validate_structured_backend(backend: MeshBackendKind) -> Result<(), MeshingError> {
    if backend == MeshBackendKind::StructuredTetrahedronFallback {
        Ok(())
    } else {
        Err(MeshingError::UnsupportedBackend(backend))
    }
}

#[cfg(test)]
mod tests;
