use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::GeometryAsset;
use runmat_meshing_cad::{
    build_cad_evaluation_model, build_cad_topology, extract_source_topology, CadEvaluationError,
    CadTopologyError, SourceTopologyError,
};
use runmat_meshing_core::{
    artifact::ANALYSIS_MESH_SCHEMA_VERSION,
    predicate::{tetrahedron_edge_aspect_ratio, tetrahedron_scaled_jacobian, tetrahedron_volume},
    AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshArtifact, AnalysisMeshNode,
    AnalysisMeshProvenance, AnalysisMeshQualityReport, AnalysisVolumeElement, BoundaryElementKind,
    ElementQuality, MeshBackendKind, MeshBackendSummary, MeshEntityProvenance, MeshKindRequest,
    MeshSizingField, MeshTargetSize, SourceEntityKind, VolumeElementKind, VolumeMeshingOptions,
};
use runmat_meshing_curve::{
    discretize_topology_curves, CurveDiscretizationError, CurveDiscretizationOptions,
};
use runmat_meshing_plc::build::{build_protected_boundary_complex, PlcBuildError};
use runmat_meshing_surface::{
    discretize_cad_surfaces_with_curves, SurfaceDiscretization, SurfaceDiscretizationError,
    SurfaceDiscretizationOptions,
};
use runmat_meshing_tetrahedron::{
    generate::{generate_solver_tetrahedron_mesh_from_plc, TetrahedronGenerationError},
    structured_grid,
};

#[derive(Debug)]
pub enum SolidMeshingError {
    UnsupportedBackend(MeshBackendKind),
    UnsupportedMeshKind(MeshKindRequest),
    UnsupportedElementKind(VolumeElementKind),
    InvalidElementBudget,
    InvalidTargetSize,
    SourceTopology(SourceTopologyError),
    CadTopology(CadTopologyError),
    CadEvaluation(CadEvaluationError),
    Curve(CurveDiscretizationError),
    Surface(SurfaceDiscretizationError),
    ProtectedBoundaryComplex(PlcBuildError),
    Tetrahedron(TetrahedronGenerationError),
    StructuredFallback(structured_grid::MeshingError),
}

impl std::fmt::Display for SolidMeshingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedBackend(backend) => {
                write!(formatter, "unsupported solid meshing backend: {backend:?}")
            }
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
            Self::SourceTopology(err) => write!(formatter, "source topology failed: {err}"),
            Self::CadTopology(err) => write!(formatter, "CAD topology failed: {err}"),
            Self::CadEvaluation(err) => write!(formatter, "CAD evaluation failed: {err}"),
            Self::Curve(err) => write!(formatter, "curve meshing failed: {err}"),
            Self::Surface(err) => write!(formatter, "surface meshing failed: {err}"),
            Self::ProtectedBoundaryComplex(err) => write!(formatter, "PLC build failed: {err}"),
            Self::Tetrahedron(err) => write!(formatter, "Tetrahedron generation failed: {err}"),
            Self::StructuredFallback(err) => {
                write!(formatter, "structured fallback meshing failed: {err}")
            }
        }
    }
}

impl std::error::Error for SolidMeshingError {}

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
        MeshBackendKind::StructuredTetrahedronFallback => {
            structured_grid::generate_analysis_mesh(geometry, options)
                .map_err(SolidMeshingError::StructuredFallback)
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
        MeshBackendKind::StructuredTetrahedronFallback => {
            structured_grid::generate_analysis_mesh_with_sizing(geometry, options, sizing)
                .map_err(SolidMeshingError::StructuredFallback)
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
    let curve_options = CurveDiscretizationOptions {
        target_size_m: target_curve_size_m(options, geometry),
        ..CurveDiscretizationOptions::default()
    };
    let curves =
        discretize_topology_curves(&topology, curve_options).map_err(SolidMeshingError::Curve)?;
    let surface = discretize_cad_surfaces_with_curves(
        &topology,
        &cad_evaluation,
        &curves,
        SurfaceDiscretizationOptions::default(),
    )
    .map_err(SolidMeshingError::Surface)?;
    let plc = build_protected_boundary_complex(&surface)
        .map_err(SolidMeshingError::ProtectedBoundaryComplex)?;
    let tetrahedron_mesh =
        generate_solver_tetrahedron_mesh_from_plc(&plc).map_err(SolidMeshingError::Tetrahedron)?;

    Ok(analysis_artifact_from_tetrahedron_mesh(
        geometry,
        sizing,
        &surface,
        tetrahedron_mesh,
    ))
}

fn validate_solid_options(options: &VolumeMeshingOptions) -> Result<(), SolidMeshingError> {
    if !matches!(
        options.backend,
        MeshBackendKind::Solid | MeshBackendKind::Auto
    ) {
        return Err(SolidMeshingError::UnsupportedBackend(options.backend));
    }
    if !matches!(options.kind, MeshKindRequest::Solid) {
        return Err(SolidMeshingError::UnsupportedMeshKind(options.kind));
    }
    if !matches!(options.element, VolumeElementKind::Tetrahedron4) {
        return Err(SolidMeshingError::UnsupportedElementKind(options.element));
    }
    if options.max_elements == 0 {
        return Err(SolidMeshingError::InvalidElementBudget);
    }
    validate_target_size(options)
}

fn validate_target_size(options: &VolumeMeshingOptions) -> Result<(), SolidMeshingError> {
    if let MeshTargetSize::LengthM(length) = options.target_size {
        if !length.is_finite() || length <= 0.0 {
            return Err(SolidMeshingError::InvalidTargetSize);
        }
    }
    if let (Some(min), Some(max)) = (options.min_size_m, options.max_size_m) {
        if !min.is_finite() || !max.is_finite() || min <= 0.0 || max <= 0.0 || min > max {
            return Err(SolidMeshingError::InvalidTargetSize);
        }
    }
    if let Some(growth_rate) = options.growth_rate {
        if !growth_rate.is_finite() || growth_rate < 1.0 {
            return Err(SolidMeshingError::InvalidTargetSize);
        }
    }
    Ok(())
}

fn target_curve_size_m(options: &VolumeMeshingOptions, geometry: &GeometryAsset) -> f64 {
    match options.target_size {
        MeshTargetSize::LengthM(length) if length.is_finite() && length > 0.0 => length,
        MeshTargetSize::Auto => geometry_span_m(geometry).unwrap_or(1.0) / 8.0,
        _ => 0.05,
    }
    .max(options.min_size_m.unwrap_or(f64::EPSILON))
    .min(options.max_size_m.unwrap_or(f64::INFINITY))
}

fn geometry_span_m(geometry: &GeometryAsset) -> Option<f64> {
    let vertices = geometry
        .surface_meshes
        .iter()
        .flat_map(|mesh| mesh.vertices.iter().copied());
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut count = 0_usize;
    for vertex in vertices {
        count += 1;
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    (count > 0).then(|| {
        (0..3)
            .map(|axis| max[axis] - min[axis])
            .fold(0.0_f64, f64::max)
    })
}

fn analysis_artifact_from_tetrahedron_mesh(
    geometry: &GeometryAsset,
    sizing: &MeshSizingField,
    surface: &SurfaceDiscretization,
    tetrahedron_mesh: runmat_meshing_tetrahedron::generate::TetrahedronMesh,
) -> AnalysisMeshArtifact {
    let node_id_map = tetrahedron_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.node_id.clone(), index as u32 + 1))
        .collect::<BTreeMap<_, _>>();
    let provenance = MeshEntityProvenance {
        source_geometry_id: geometry.geometry_id.clone(),
        source_geometry_revision: geometry.revision,
        source_entity_kind: SourceEntityKind::Mesh,
        source_entity_id: tetrahedron_mesh.mesh_id.clone(),
        region_ids: Vec::new(),
    };
    let nodes = tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| AnalysisMeshNode {
            node_id: node_id_map[&node.node_id],
            coordinates_m: node.coordinates_m,
            provenance: vec![provenance.clone()],
        })
        .collect::<Vec<_>>();
    let coordinates_by_node_id = nodes
        .iter()
        .map(|node| (node.node_id, node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    let volume_elements = tetrahedron_mesh
        .elements
        .iter()
        .map(|element| AnalysisVolumeElement {
            element_id: element.element_id.id.clone(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids: element
                .node_ids
                .iter()
                .map(|node_id| node_id_map[node_id])
                .collect(),
            material_region_id: element.material_region_id.clone(),
            provenance: vec![provenance.clone()],
        })
        .collect::<Vec<_>>();
    let boundary_faces = tetrahedron_mesh
        .boundary_faces
        .iter()
        .map(|face| {
            let node_ids = face
                .node_ids
                .iter()
                .map(|node_id| node_id_map[node_id])
                .collect::<Vec<_>>();
            AnalysisBoundaryFace {
                face_id: face.face_id.id.clone(),
                kind: BoundaryElementKind::Tri3,
                adjacent_volume_element_ids: adjacent_volume_element_ids(
                    &node_ids,
                    &volume_elements,
                ),
                region_ids: surface_region_ids(surface, &face.source_face_id.id),
                node_ids,
                provenance: vec![provenance.clone()],
            }
        })
        .collect::<Vec<_>>();
    let boundary_edges = boundary_edges_from_faces(&boundary_faces, &provenance);
    let quality = quality_report(&volume_elements, &coordinates_by_node_id);

    AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: format!("analysis_mesh_{}", geometry.geometry_id),
        nodes,
        volume_elements,
        boundary_faces,
        boundary_edges,
        quality,
        sizing: sizing.clone(),
        backend: MeshBackendSummary {
            backend: "solid".to_string(),
            algorithm: "topology_first_plc_tetrahedron/v1".to_string(),
            surface_element_count: surface.elements.len(),
            tetrahedron_element_count: tetrahedron_mesh.elements.len(),
            boundary_face_recovery_ratio: 1.0,
            boundary_edge_recovery_ratio: 1.0,
            volume_component_count: 1,
            tetrahedron_recovered_component_ratio: 1.0,
            tetrahedron_volume_coverage_ratio: 1.0,
            ..MeshBackendSummary::default()
        },
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "topology_first_plc_tetrahedron/v1".to_string(),
            source_geometry_id: geometry.geometry_id.clone(),
            source_geometry_revision: geometry.revision,
            source_geometry_sha256: Some(geometry.source.sha256.clone()),
        },
    }
}

fn adjacent_volume_element_ids(
    boundary_node_ids: &[u32],
    volume_elements: &[AnalysisVolumeElement],
) -> Vec<String> {
    let boundary_nodes = boundary_node_ids.iter().copied().collect::<BTreeSet<_>>();
    volume_elements
        .iter()
        .filter(|element| {
            boundary_nodes
                .iter()
                .all(|node_id| element.node_ids.contains(node_id))
        })
        .map(|element| element.element_id.clone())
        .collect()
}

fn surface_region_ids(surface: &SurfaceDiscretization, source_face_id: &str) -> Vec<String> {
    let Ok(source_face_id) = source_face_id.parse::<u32>() else {
        return Vec::new();
    };
    surface
        .elements
        .iter()
        .find(|element| element.source_face_id == source_face_id)
        .map(|element| element.region_ids.clone())
        .unwrap_or_default()
}

fn boundary_edges_from_faces(
    faces: &[AnalysisBoundaryFace],
    provenance: &MeshEntityProvenance,
) -> Vec<AnalysisBoundaryEdge> {
    let mut edges = BTreeMap::<[u32; 2], AnalysisBoundaryEdge>::new();
    for face in faces {
        if face.node_ids.len() != 3 {
            continue;
        }
        for edge in [
            sorted_edge(face.node_ids[0], face.node_ids[1]),
            sorted_edge(face.node_ids[1], face.node_ids[2]),
            sorted_edge(face.node_ids[2], face.node_ids[0]),
        ] {
            edges
                .entry(edge)
                .and_modify(|entry| {
                    entry.adjacent_boundary_face_ids.push(face.face_id.clone());
                    append_unique_region_ids(&mut entry.region_ids, &face.region_ids);
                })
                .or_insert_with(|| AnalysisBoundaryEdge {
                    edge_id: format!("boundary_edge_{}_{}", edge[0], edge[1]),
                    node_ids: edge,
                    adjacent_boundary_face_ids: vec![face.face_id.clone()],
                    region_ids: face.region_ids.clone(),
                    provenance: vec![provenance.clone()],
                });
        }
    }
    edges.into_values().collect()
}

fn append_unique_region_ids(target: &mut Vec<String>, source: &[String]) {
    for region_id in source {
        if !target.contains(region_id) {
            target.push(region_id.clone());
        }
    }
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn quality_report(
    volume_elements: &[AnalysisVolumeElement],
    coordinates_by_node_id: &BTreeMap<u32, [f64; 3]>,
) -> AnalysisMeshQualityReport {
    let elements = volume_elements
        .iter()
        .filter_map(|element| {
            if element.node_ids.len() != 4 {
                return None;
            }
            let points = [
                *coordinates_by_node_id.get(&element.node_ids[0])?,
                *coordinates_by_node_id.get(&element.node_ids[1])?,
                *coordinates_by_node_id.get(&element.node_ids[2])?,
                *coordinates_by_node_id.get(&element.node_ids[3])?,
            ];
            Some(ElementQuality {
                element_id: element.element_id.clone(),
                scaled_jacobian: tetrahedron_scaled_jacobian(points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(points),
                aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                volume_m3: tetrahedron_volume(points),
            })
        })
        .collect::<Vec<_>>();
    let min_scaled_jacobian = elements
        .iter()
        .map(|element| element.scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let max_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .fold(0.0_f64, f64::max);
    let mean_aspect_ratio = if elements.is_empty() {
        0.0
    } else {
        elements
            .iter()
            .map(|element| element.aspect_ratio)
            .sum::<f64>()
            / elements.len() as f64
    };
    AnalysisMeshQualityReport {
        min_scaled_jacobian: min_scaled_jacobian.min(1.0),
        min_exact_scaled_jacobian: min_scaled_jacobian.min(1.0),
        mean_aspect_ratio,
        max_aspect_ratio,
        inverted_element_count: elements
            .iter()
            .filter(|element| element.volume_m3 <= 0.0)
            .count(),
        mean_boundary_projection_error_m: 0.0,
        max_boundary_projection_error_m: 0.0,
        elements,
    }
}

#[cfg(test)]
mod tests;
