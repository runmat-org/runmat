use runmat_geometry_core::GeometryAsset;
use runmat_meshing_cad::{build_cad_evaluation_model, build_cad_topology, extract_source_topology};
use runmat_meshing_core::{
    AnalysisMeshArtifact, MeshBackendKind, MeshSizingField, VolumeMeshingOptions,
};
use runmat_meshing_curve::discretize_topology_curves_with_sizing;
use runmat_meshing_plc::build::build_protected_boundary_complex;
use runmat_meshing_surface::{
    discretize_cad_topology_surfaces_with_curves, validate_cad_topology_surface_discretization,
    SurfaceValidationOptions,
};
use runmat_meshing_tetrahedron::{recover::build_recovery_queue_from_plc, structured_grid};

mod artifact;
mod error;
mod options;
mod sizing;
mod stage_options;
mod tetrahedron_stage;

use artifact::analysis_artifact_from_tetrahedron_mesh;
pub use error::SolidMeshingError;
use options::validate_solid_options;
use stage_options::{curve_discretization_options, surface_discretization_options};
use tetrahedron_stage::generate_solid_tetrahedron_mesh;

pub fn generate_analysis_mesh(
    geometry: &GeometryAsset,
    options: VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, SolidMeshingError> {
    match options.backend {
        MeshBackendKind::Auto | MeshBackendKind::Solid => generate_solid_analysis_mesh(
            geometry,
            &VolumeMeshingOptions {
                backend: MeshBackendKind::Solid,
                ..options
            },
        ),
        MeshBackendKind::StructuredGridTetrahedron => {
            structured_grid::generate_analysis_mesh(geometry, options)
                .map_err(SolidMeshingError::StructuredGrid)
        }
    }
}

pub fn generate_analysis_mesh_with_sizing(
    geometry: &GeometryAsset,
    options: VolumeMeshingOptions,
    sizing: &MeshSizingField,
) -> Result<AnalysisMeshArtifact, SolidMeshingError> {
    match options.backend {
        MeshBackendKind::Auto | MeshBackendKind::Solid => generate_solid_analysis_mesh_with_sizing(
            geometry,
            &VolumeMeshingOptions {
                backend: MeshBackendKind::Solid,
                ..options
            },
            sizing,
        ),
        MeshBackendKind::StructuredGridTetrahedron => {
            structured_grid::generate_analysis_mesh_with_sizing(geometry, options, sizing)
                .map_err(SolidMeshingError::StructuredGrid)
        }
    }
}

pub fn generate_solid_analysis_mesh(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
) -> Result<AnalysisMeshArtifact, SolidMeshingError> {
    generate_solid_analysis_mesh_with_sizing(geometry, options, &MeshSizingField::default())
}

pub fn generate_solid_analysis_mesh_with_sizing(
    geometry: &GeometryAsset,
    options: &VolumeMeshingOptions,
    sizing: &MeshSizingField,
) -> Result<AnalysisMeshArtifact, SolidMeshingError> {
    validate_solid_options(options)?;

    let topology = extract_source_topology(geometry).map_err(SolidMeshingError::SourceTopology)?;
    let cad_topology =
        build_cad_topology(geometry, &topology).map_err(SolidMeshingError::CadTopology)?;
    let cad_evaluation = build_cad_evaluation_model(&cad_topology, &topology)
        .map_err(SolidMeshingError::CadEvaluation)?;
    let curve_options = curve_discretization_options(options, geometry);
    let curves = discretize_topology_curves_with_sizing(&topology, curve_options, Some(sizing))
        .map_err(SolidMeshingError::Curve)?;
    let surface = discretize_cad_topology_surfaces_with_curves(
        &cad_topology,
        &topology,
        &cad_evaluation,
        &curves,
        surface_discretization_options(),
    )
    .map_err(SolidMeshingError::Surface)?;
    validate_cad_topology_surface_discretization(
        &cad_topology,
        &topology,
        &surface,
        SurfaceValidationOptions::default(),
    )
    .map_err(SolidMeshingError::SurfaceValidation)?;
    let plc = build_protected_boundary_complex(&surface)
        .map_err(SolidMeshingError::ProtectedBoundaryComplex)?;
    let tetrahedron_mesh = generate_solid_tetrahedron_mesh(&plc)?;
    let recovery_queue = build_recovery_queue_from_plc(&plc, &tetrahedron_mesh)
        .map_err(SolidMeshingError::TetrahedronRecovery)?;

    Ok(analysis_artifact_from_tetrahedron_mesh(
        geometry,
        sizing,
        &surface,
        &recovery_queue,
        tetrahedron_mesh,
    ))
}

#[cfg(test)]
mod tests;
