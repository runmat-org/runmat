use crate::{
    contracts::{artifact::ANALYSIS_MESH_SCHEMA_VERSION, AnalysisMeshArtifact},
    quality::QualityThresholds,
};

mod components;
use components::validate_volume_component_count;

mod connectivity;
pub use connectivity::{volume_component_count, volume_component_element_counts};

mod coverage;
use coverage::{
    validate_boundary_area_coverage, validate_bounds_coverage, validate_coverage_samples,
    validate_volume_coverage,
};

mod elements;
use elements::validate_volume_elements;

mod edges;
use edges::validate_boundary_edges;

mod faces;
use faces::validate_boundary_faces;

mod geometry;
pub use geometry::mesh_contains_point;

mod nodes;
use nodes::validate_nodes;

mod plc_input;
use plc_input::validate_plc_input_evidence;

mod quality;
use quality::validate_quality;

mod recovery;
use recovery::{
    validate_boundary_edge_recovery, validate_boundary_face_recovery,
    validate_no_rolled_back_material_interface_partitions,
    validate_no_unrecovered_tetrahedron_components, validate_no_unrepaired_exact_quality,
};

mod regions;
use regions::{validate_required_boundary_regions, validate_required_material_regions};

mod source_provenance;
use source_provenance::validate_boundary_source_provenance;

mod types;
pub use types::{
    analysis_mesh_validation_error_code, AnalysisMeshValidationError, AnalysisMeshValidationOptions,
};

pub fn validate_analysis_mesh(
    mesh: &AnalysisMeshArtifact,
    thresholds: QualityThresholds,
) -> Result<(), AnalysisMeshValidationError> {
    validate_analysis_mesh_with_options(
        mesh,
        AnalysisMeshValidationOptions {
            quality: thresholds,
            ..AnalysisMeshValidationOptions::default()
        },
    )
}

pub fn validate_analysis_mesh_with_options(
    mesh: &AnalysisMeshArtifact,
    options: AnalysisMeshValidationOptions,
) -> Result<(), AnalysisMeshValidationError> {
    if mesh.schema_version != ANALYSIS_MESH_SCHEMA_VERSION {
        return Err(AnalysisMeshValidationError::UnsupportedSchema {
            schema_version: mesh.schema_version.clone(),
        });
    }
    if mesh.nodes.is_empty() {
        return Err(AnalysisMeshValidationError::EmptyNodes);
    }
    if mesh.volume_elements.is_empty() {
        return Err(AnalysisMeshValidationError::EmptyVolumeElements);
    }
    if let Some(max_element_count) = options.max_volume_element_count {
        if mesh.volume_elements.len() > max_element_count {
            return Err(AnalysisMeshValidationError::ElementBudgetExceeded {
                element_count: mesh.volume_elements.len(),
                max_element_count,
            });
        }
    }

    let node_ids = validate_nodes(mesh)?;

    let element_ids = validate_volume_elements(mesh, &node_ids)?;

    let face_ids = validate_boundary_faces(mesh, &node_ids, &element_ids)?;

    let recovered_boundary_edges = validate_boundary_edges(mesh, &node_ids, &face_ids)?;

    validate_required_boundary_regions(mesh, &options.required_boundary_region_ids)?;
    validate_required_material_regions(mesh, &options.required_material_region_ids)?;
    validate_plc_input_evidence(mesh)?;
    validate_boundary_source_provenance(mesh, options.require_boundary_source_edge_provenance)?;
    validate_no_unrecovered_tetrahedron_components(
        mesh,
        options.require_no_unrecovered_tetrahedron_components,
    )?;
    validate_no_rolled_back_material_interface_partitions(mesh)?;
    validate_no_unrepaired_exact_quality(mesh, options.require_no_unrepaired_exact_quality)?;
    validate_volume_component_count(mesh, options.max_volume_component_count)?;
    validate_coverage_samples(
        mesh,
        &options.coverage_sample_points_m,
        options.min_coverage_sample_ratio,
    )?;
    validate_bounds_coverage(
        mesh,
        options.expected_bounds_m,
        options.min_bounds_coverage_ratio,
    )?;
    validate_volume_coverage(
        mesh,
        options.expected_volume_m3,
        options.min_volume_coverage_ratio,
    )?;
    validate_boundary_area_coverage(
        mesh,
        options.expected_boundary_area_m2,
        options.min_boundary_area_ratio,
    )?;
    validate_boundary_face_recovery(mesh, options.min_boundary_face_recovery_ratio)?;
    validate_boundary_edge_recovery(
        mesh,
        &recovered_boundary_edges,
        options.min_boundary_edge_recovery_ratio,
    )?;
    validate_quality(mesh, options.quality)
}

#[cfg(test)]
mod tests;
