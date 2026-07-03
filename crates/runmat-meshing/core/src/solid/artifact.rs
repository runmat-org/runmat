use super::*;

pub(super) fn analysis_mesh_from_preparation(
    preparation: &SolidMeshPreparation,
    options: &VolumeMeshingOptions,
    sizing: Option<&MeshSizingField>,
) -> Result<AnalysisMeshArtifact, SolidMeshError> {
    let mut node_id_map = BTreeMap::<TopologyEntityId, u32>::new();
    let nodes = preparation
        .solver_tetrahedron_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let node_id = index as u32 + 1;
            node_id_map.insert(node.node_id.clone(), node_id);
            AnalysisMeshNode {
                node_id,
                coordinates_m: node.coordinates_m,
                provenance: vec![MeshEntityProvenance {
                    source_geometry_id: preparation.topology.source_geometry_id.clone(),
                    source_geometry_revision: preparation.topology.source_geometry_revision,
                    source_entity_kind: match node.node_id.stage {
                        crate::contracts::MeshingStage::TetrahedronMesh => SourceEntityKind::Body,
                        _ => SourceEntityKind::Mesh,
                    },
                    source_entity_id: node.node_id.id.clone(),
                    region_ids: Vec::new(),
                }],
            }
        })
        .collect::<Vec<_>>();

    let mut source_surface_to_tetrahedron = BTreeMap::<u32, Vec<String>>::new();
    let mut volume_elements = Vec::<AnalysisVolumeElement>::new();
    let mut quality_elements = Vec::<ElementQuality>::new();
    let mut tetrahedron_centroids = Vec::<(String, [f64; 3])>::new();
    let tetrahedron_nodes_by_id = preparation
        .solver_tetrahedron_mesh
        .nodes
        .iter()
        .map(|node| (node.node_id.clone(), node.coordinates_m))
        .collect::<BTreeMap<_, _>>();
    for (tetrahedron_index, tetrahedron) in preparation
        .solver_tetrahedron_mesh
        .elements
        .iter()
        .enumerate()
    {
        let element_id = format!("solid_tetrahedron_{}", tetrahedron_index + 1);
        let node_ids = tetrahedron
            .node_ids
            .iter()
            .map(|node_id| {
                node_id_map.get(node_id).copied().ok_or_else(|| {
                    SolidMeshError::MissingTetrahedronNode {
                        node_id: node_id.id.clone(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(source_surface_element_id) =
            solver_tetrahedron_source_surface_element_id(preparation, tetrahedron_index)
        {
            source_surface_to_tetrahedron
                .entry(source_surface_element_id)
                .or_default()
                .push(element_id.clone());
        }
        volume_elements.push(AnalysisVolumeElement {
            element_id: element_id.clone(),
            kind: VolumeElementKind::Tetrahedron4,
            node_ids,
            material_region_id: tetrahedron.material_region_id.clone(),
            provenance: vec![MeshEntityProvenance {
                source_geometry_id: preparation.topology.source_geometry_id.clone(),
                source_geometry_revision: preparation.topology.source_geometry_revision,
                source_entity_kind: SourceEntityKind::Body,
                source_entity_id: tetrahedron.element_id.id.clone(),
                region_ids: vec![tetrahedron.material_region_id.clone()],
            }],
        });
        let tetrahedron_points = [
            *tetrahedron_nodes_by_id
                .get(&tetrahedron.node_ids[0])
                .ok_or_else(|| SolidMeshError::MissingTetrahedronNode {
                    node_id: tetrahedron.node_ids[0].id.clone(),
                })?,
            *tetrahedron_nodes_by_id
                .get(&tetrahedron.node_ids[1])
                .ok_or_else(|| SolidMeshError::MissingTetrahedronNode {
                    node_id: tetrahedron.node_ids[1].id.clone(),
                })?,
            *tetrahedron_nodes_by_id
                .get(&tetrahedron.node_ids[2])
                .ok_or_else(|| SolidMeshError::MissingTetrahedronNode {
                    node_id: tetrahedron.node_ids[2].id.clone(),
                })?,
            *tetrahedron_nodes_by_id
                .get(&tetrahedron.node_ids[3])
                .ok_or_else(|| SolidMeshError::MissingTetrahedronNode {
                    node_id: tetrahedron.node_ids[3].id.clone(),
                })?,
        ];
        tetrahedron_centroids.push((element_id.clone(), tetrahedron_centroid(tetrahedron_points)));
        let aspect_ratio = tetrahedron_edge_aspect_ratio(tetrahedron_points);
        quality_elements.push(ElementQuality {
            element_id,
            scaled_jacobian: (1.0 / aspect_ratio.max(1.0)).min(1.0),
            exact_scaled_jacobian: tetrahedron_scaled_jacobian(tetrahedron_points),
            aspect_ratio,
            volume_m3: tetrahedron_volume(tetrahedron_points),
        });
    }

    let boundary_faces = preparation
        .surface
        .elements
        .iter()
        .map(|element| {
            let node_ids = element
                .node_ids
                .iter()
                .map(|node_id| {
                    let plc_node_id = plc_node_topology_id(*node_id);
                    node_id_map.get(&plc_node_id).copied().ok_or_else(|| {
                        SolidMeshError::MissingTetrahedronNode {
                            node_id: plc_node_id.id,
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut adjacent_volume_element_ids = source_surface_to_tetrahedron
                .get(&element.element_id)
                .cloned()
                .unwrap_or_default();
            if adjacent_volume_element_ids.is_empty() {
                if let Some(nearest_tetrahedron_id) = nearest_tetrahedron_for_boundary_element(
                    element,
                    &preparation.surface,
                    &tetrahedron_centroids,
                ) {
                    adjacent_volume_element_ids.push(nearest_tetrahedron_id);
                }
            }
            Ok(AnalysisBoundaryFace {
                face_id: format!("solid_boundary_{}", element.element_id + 1),
                kind: BoundaryElementKind::Tri3,
                node_ids,
                adjacent_volume_element_ids,
                region_ids: element.region_ids.clone(),
                provenance: vec![MeshEntityProvenance {
                    source_geometry_id: preparation.topology.source_geometry_id.clone(),
                    source_geometry_revision: preparation.topology.source_geometry_revision,
                    source_entity_kind: SourceEntityKind::Face,
                    source_entity_id: element.source_face_id.to_string(),
                    region_ids: element.region_ids.clone(),
                }],
            })
        })
        .collect::<Result<Vec<_>, SolidMeshError>>()?;

    let boundary_edges = boundary_edges_from_faces(&boundary_faces, preparation);
    let quality = quality_report(
        quality_elements,
        preparation
            .surface
            .elements
            .iter()
            .map(|element| element.max_projection_error_m),
    );
    let backend = solid_backend_summary(preparation, &boundary_faces, &boundary_edges, &quality);

    let mesh = AnalysisMeshArtifact {
        schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
        mesh_id: format!("solid_{}", preparation.topology.mesh_id),
        nodes,
        volume_elements,
        boundary_faces,
        boundary_edges,
        quality,
        sizing: solid_mesh_sizing(
            options,
            preparation.effective_sizing.as_ref().or(sizing),
            preparation,
        ),
        backend,
        adaptive_iterations: Vec::new(),
        provenance: AnalysisMeshProvenance {
            algorithm: "plc_tetrahedron/v1".to_string(),
            source_geometry_id: preparation.topology.source_geometry_id.clone(),
            source_geometry_revision: preparation.topology.source_geometry_revision,
            source_geometry_sha256: preparation.topology.source_geometry_sha256.clone(),
        },
    };
    validate_analysis_mesh_with_options(&mesh, solid_validation_options(preparation, options))
        .map_err(SolidMeshError::Validation)?;
    Ok(mesh)
}

fn solid_backend_summary(
    preparation: &SolidMeshPreparation,
    boundary_faces: &[AnalysisBoundaryFace],
    boundary_edges: &[AnalysisBoundaryEdge],
    quality: &AnalysisMeshQualityReport,
) -> MeshBackendSummary {
    MeshBackendSummary {
        backend: "solid".to_string(),
        algorithm: "plc_tetrahedron/v1".to_string(),
        source_topology_vertex_count: preparation.topology.vertices.len(),
        source_topology_edge_count: preparation.topology.edges.len(),
        source_topology_face_count: preparation.topology.faces.len(),
        cad_topology_source: cad_topology_source_label(preparation.cad_topology.source).to_string(),
        cad_vertex_count: preparation.cad_topology.report.vertex_count,
        cad_edge_count: preparation.cad_topology.report.edge_count,
        cad_face_count: preparation.cad_topology.report.face_count,
        cad_shell_count: preparation.cad_topology.report.shell_count,
        cad_volume_count: preparation.cad_topology.report.volume_count,
        cad_semantic_face_count: preparation.cad_topology.report.semantic_face_count,
        cad_imported_face_count: preparation.cad_topology.report.imported_face_count,
        cad_evaluator_face_count: preparation.cad_topology.report.evaluator_face_count,
        cad_generic_face_count: preparation.cad_topology.report.generic_face_count,
        cad_closed_shell_count: preparation.cad_topology.report.closed_shell_count,
        cad_evaluation_source: cad_evaluation_source_label(preparation.cad_evaluation.source)
            .to_string(),
        cad_face_frame_count: preparation.cad_evaluation_report.face_frame_count,
        cad_evaluation_evaluator_face_count: preparation.cad_evaluation_report.evaluator_face_count,
        cad_evaluation_live_query_face_count: preparation
            .cad_evaluation_report
            .live_query_face_count,
        cad_evaluation_exact_query_face_count: preparation
            .cad_evaluation_report
            .exact_query_face_count,
        cad_evaluation_point_supported_face_count: preparation
            .cad_evaluation_report
            .point_evaluation_supported_face_count,
        cad_evaluation_projection_supported_face_count: preparation
            .cad_evaluation_report
            .projection_supported_face_count,
        cad_evaluation_normal_supported_face_count: preparation
            .cad_evaluation_report
            .normal_supported_face_count,
        cad_evaluation_derivative_supported_face_count: preparation
            .cad_evaluation_report
            .derivative_supported_face_count,
        cad_evaluation_curvature_supported_face_count: preparation
            .cad_evaluation_report
            .curvature_supported_face_count,
        cad_evaluation_missing_exact_query_face_count: preparation
            .cad_evaluation_report
            .missing_exact_query_face_count,
        cad_evaluation_missing_derivative_query_face_count: preparation
            .cad_evaluation_report
            .missing_derivative_query_face_count,
        cad_evaluation_missing_curvature_query_face_count: preparation
            .cad_evaluation_report
            .missing_curvature_query_face_count,
        cad_evaluation_sample_count: preparation.cad_evaluation_report.evaluator_sample_count,
        cad_evaluation_rejected_sample_count: preparation
            .cad_evaluation_report
            .evaluator_rejected_sample_count,
        cad_projection_query_count: preparation.cad_evaluation_report.projection_query_count,
        cad_derivative_query_count: preparation.cad_evaluation_report.derivative_query_count,
        cad_curvature_query_count: preparation.cad_evaluation_report.curvature_query_count,
        cad_uv_domain_face_count: preparation.cad_evaluation_report.uv_domain_face_count,
        cad_uv_projection_out_of_bounds_count: preparation
            .cad_evaluation_report
            .uv_projection_out_of_bounds_count,
        cad_max_projection_error_m: preparation.cad_evaluation_report.max_projection_error_m,
        cad_max_normal_deviation: preparation.cad_evaluation_report.max_normal_deviation,
        cad_max_curvature_estimate_1_per_m: preparation
            .cad_evaluation_report
            .max_curvature_estimate_1_per_m
            .unwrap_or(0.0),
        curve_element_count: preparation.curves.elements.len(),
        surface_element_count: preparation.surface.elements.len(),
        surface_source_edge_loop_count: preparation.surface_validation.source_edge_loop_count,
        surface_closed_edge_loop_count: preparation
            .surface_validation
            .closed_source_edge_loop_count,
        surface_conforming_source_edge_count: preparation
            .surface_validation
            .conforming_source_edge_count,
        surface_missing_source_edge_count: preparation.surface_validation.missing_source_edge_count,
        surface_projection_error_m: preparation.surface_validation.max_projection_error_m,
        surface_face_coverage_ratio: preparation.surface_validation.face_coverage_ratio,
        surface_cad_face_count: surface_cad_face_count(&preparation.surface),
        surface_exact_cad_sample_node_count: preparation.surface.exact_cad_sample_node_count,
        surface_rejected_exact_cad_sample_count: preparation
            .surface
            .rejected_exact_cad_sample_count,
        surface_max_cad_projection_error_m: surface_max_cad_projection_error_m(
            &preparation.surface,
        ),
        volume_component_count: preparation.tetrahedron_stage.volume_component_count,
        interior_seed_point_count: preparation.tetrahedron_stage.interior_seed_point_count,
        tetrahedron_element_count: preparation.solver_tetrahedron_mesh.elements.len(),
        tetrahedron_recovered_component_ratio: preparation
            .tetrahedron_stage
            .recovered_component_ratio,
        tetrahedron_fan_fallback_component_count: 0,
        tetrahedron_volume_coverage_ratio: 1.0,
        tetrahedron_refinement_pass_count: 0,
        tetrahedron_refinement_point_count: 0,
        tetrahedron_requested_refinement_point_count: preparation
            .tetrahedron_stage
            .requested_refinement_point_count,
        tetrahedron_accepted_requested_refinement_location_count: preparation
            .tetrahedron_stage
            .accepted_requested_refinement_point_count,
        tetrahedron_accepted_requested_refinement_point_count: preparation
            .tetrahedron_stage
            .accepted_requested_refinement_point_count,
        tetrahedron_accepted_requested_refinement_surrogate_point_count: preparation
            .tetrahedron_stage
            .accepted_requested_refinement_surrogate_point_count,
        tetrahedron_rejected_requested_refinement_point_count: preparation
            .tetrahedron_stage
            .rejected_requested_refinement_point_count,
        tetrahedron_requested_refinement_rejected_by_reason: preparation
            .tetrahedron_stage
            .requested_refinement_rejected_by_reason
            .clone(),
        tetrahedron_dropped_requested_refinement_point_count: preparation
            .tetrahedron_stage
            .dropped_requested_refinement_point_count,
        tetrahedron_requested_refinement_dropped_by_reason: preparation
            .tetrahedron_stage
            .requested_refinement_dropped_by_reason
            .clone(),
        tetrahedron_max_radius_edge_ratio: quality.max_aspect_ratio,
        tetrahedron_sizing_violation_count: 0,
        tetrahedron_min_exact_scaled_jacobian: quality.min_exact_scaled_jacobian,
        tetrahedron_exact_scaled_jacobian_below_threshold_count:
            exact_scaled_jacobian_below_threshold_count(quality),
        tetrahedron_exact_scaled_jacobian_bins: exact_scaled_jacobian_bins(quality),
        tetrahedron_optimization_pass_count: usize::from(
            preparation.solver_tetrahedron_mesh.quality_optimized,
        ),
        tetrahedron_smoothed_point_count: 0,
        tetrahedron_sliver_count: 0,
        tetrahedron_sliver_removed_count: 0,
        tetrahedron_optimization_target_seed_count: 0,
        tetrahedron_optimization_skipped_target_seed_count: 0,
        tetrahedron_optimization_rejected_edit_count: 0,
        tetrahedron_optimization_initial_max_aspect_ratio: quality.max_aspect_ratio,
        tetrahedron_optimization_final_max_aspect_ratio: quality.max_aspect_ratio,
        tetrahedron_optimization_initial_min_exact_scaled_jacobian: quality
            .min_exact_scaled_jacobian,
        tetrahedron_optimization_final_min_exact_scaled_jacobian: quality.min_exact_scaled_jacobian,
        tetrahedron_untangling_pass_count: 0,
        tetrahedron_untangling_initial_near_singular_count: 0,
        tetrahedron_untangling_final_near_singular_count: 0,
        tetrahedron_untangling_relocated_seed_count: 0,
        tetrahedron_untangling_reconnected_edge_star_count: 0,
        tetrahedron_untangling_reconnected_boundary_adjacent_cavity_count: 0,
        tetrahedron_untangling_reconnected_node_adjacent_cavity_count: 0,
        tetrahedron_exact_quality_repair_pass_count: 0,
        tetrahedron_exact_quality_reconnected_cavity_count: 0,
        tetrahedron_exact_quality_reconnection_quality_gain_count: 0,
        tetrahedron_exact_quality_face_neighbor_reconnected_cavity_count: 0,
        tetrahedron_exact_quality_connected_reconnected_cavity_count: 0,
        tetrahedron_exact_quality_node_adjacent_reconnected_cavity_count: 0,
        tetrahedron_exact_quality_boundary_adjacent_reconnected_cavity_count: 0,
        tetrahedron_exact_quality_expanded_connected_reconnected_cavity_count: 0,
        tetrahedron_exact_quality_split_cavity_count: 0,
        tetrahedron_exact_quality_seed_star_collapse_count: 0,
        tetrahedron_exact_quality_seed_star_relocation_count: 0,
        tetrahedron_exact_quality_unrepaired_total_count: 0,
        tetrahedron_exact_quality_unrepaired_general_cavity_count: 0,
        tetrahedron_exact_quality_unrepaired_boundary_adjacent_count: 0,
        tetrahedron_exact_quality_unrepaired_node_adjacent_count: 0,
        tetrahedron_exact_quality_unrepaired_interior_seed_count: 0,
        tetrahedron_exact_quality_unrepaired_edge_star_count: 0,
        boundary_face_recovery_ratio: boundary_face_recovery_ratio(boundary_faces),
        boundary_edge_recovery_ratio: boundary_edge_recovery_ratio(boundary_faces, boundary_edges),
    }
}

fn cad_topology_source_label(source: CadTopologySource) -> &'static str {
    match source {
        CadTopologySource::SemanticCad => "semantic_cad",
        CadTopologySource::GenericCadMesh => "generic_cad_mesh",
        CadTopologySource::MeshFallback => "mesh_fallback",
    }
}

fn cad_evaluation_source_label(source: CadEvaluationSource) -> &'static str {
    match source {
        CadEvaluationSource::ParametricCad => "parametric_cad",
        CadEvaluationSource::ImportedEvaluatorSamples => "imported_evaluator_samples",
        CadEvaluationSource::PlanarFacetApproximation => "planar_facet_approximation",
    }
}

pub(super) fn surface_cad_face_count(surface: &SurfaceDiscretization) -> usize {
    surface
        .elements
        .iter()
        .filter_map(|element| element.cad_face_id.as_ref())
        .collect::<BTreeSet<_>>()
        .len()
}

pub(super) fn surface_max_cad_projection_error_m(surface: &SurfaceDiscretization) -> f64 {
    surface
        .elements
        .iter()
        .map(|element| element.max_projection_error_m)
        .fold(0.0_f64, f64::max)
}

fn boundary_face_recovery_ratio(boundary_faces: &[AnalysisBoundaryFace]) -> f64 {
    if boundary_faces.is_empty() {
        return 1.0;
    }
    let recovered_count = boundary_faces
        .iter()
        .filter(|face| !face.adjacent_volume_element_ids.is_empty())
        .count();
    recovered_count as f64 / boundary_faces.len() as f64
}

fn nearest_tetrahedron_for_boundary_element(
    element: &crate::surface::SurfaceElement,
    surface: &SurfaceDiscretization,
    tetrahedron_centroids: &[(String, [f64; 3])],
) -> Option<String> {
    let points = [
        surface.nodes[element.node_ids[0] as usize].coordinates_m,
        surface.nodes[element.node_ids[1] as usize].coordinates_m,
        surface.nodes[element.node_ids[2] as usize].coordinates_m,
    ];
    let centroid = triangle_centroid(points);
    tetrahedron_centroids
        .iter()
        .min_by(|(_, left), (_, right)| {
            distance_squared(*left, centroid).total_cmp(&distance_squared(*right, centroid))
        })
        .map(|(element_id, _)| element_id.clone())
}

fn boundary_edge_recovery_ratio(
    boundary_faces: &[AnalysisBoundaryFace],
    boundary_edges: &[AnalysisBoundaryEdge],
) -> f64 {
    let expected_edges = boundary_face_edge_set(boundary_faces);
    if expected_edges.is_empty() {
        return 1.0;
    }
    let recovered_edges = boundary_edges
        .iter()
        .filter(|edge| !edge.adjacent_boundary_face_ids.is_empty())
        .map(|edge| sorted_edge(edge.node_ids[0], edge.node_ids[1]))
        .collect::<BTreeSet<_>>();
    expected_edges
        .iter()
        .filter(|edge| recovered_edges.contains(*edge))
        .count() as f64
        / expected_edges.len() as f64
}

fn boundary_face_edge_set(boundary_faces: &[AnalysisBoundaryFace]) -> BTreeSet<[u32; 2]> {
    let mut edges = BTreeSet::<[u32; 2]>::new();
    for face in boundary_faces {
        if face.node_ids.len() != 3 {
            continue;
        }
        edges.extend(triangle_edges([
            face.node_ids[0],
            face.node_ids[1],
            face.node_ids[2],
        ]));
    }
    edges
}

fn boundary_edges_from_faces(
    boundary_faces: &[AnalysisBoundaryFace],
    preparation: &SolidMeshPreparation,
) -> Vec<AnalysisBoundaryEdge> {
    let mut edges = BTreeMap::<[u32; 2], BoundaryEdgeAccumulator>::new();
    for face in boundary_faces {
        if face.node_ids.len() != 3 {
            continue;
        }
        for edge in triangle_edges([face.node_ids[0], face.node_ids[1], face.node_ids[2]]) {
            let accumulator = edges
                .entry(edge)
                .or_insert_with(|| BoundaryEdgeAccumulator {
                    node_ids: edge,
                    adjacent_boundary_face_ids: Vec::new(),
                    region_ids: BTreeSet::new(),
                });
            accumulator
                .adjacent_boundary_face_ids
                .push(face.face_id.clone());
            accumulator
                .region_ids
                .extend(face.region_ids.iter().cloned());
        }
    }
    edges
        .into_values()
        .enumerate()
        .map(|(index, edge)| {
            let region_ids = edge.region_ids.into_iter().collect::<Vec<_>>();
            AnalysisBoundaryEdge {
                edge_id: format!("solid_boundary_edge_{}", index + 1),
                node_ids: edge.node_ids,
                adjacent_boundary_face_ids: edge.adjacent_boundary_face_ids,
                region_ids: region_ids.clone(),
                provenance: vec![MeshEntityProvenance {
                    source_geometry_id: preparation.topology.source_geometry_id.clone(),
                    source_geometry_revision: preparation.topology.source_geometry_revision,
                    source_entity_kind: SourceEntityKind::Edge,
                    source_entity_id: format!("boundary_edge_{}", index + 1),
                    region_ids,
                }],
            }
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundaryEdgeAccumulator {
    node_ids: [u32; 2],
    adjacent_boundary_face_ids: Vec<String>,
    region_ids: BTreeSet<String>,
}

fn triangle_edges(node_ids: [u32; 3]) -> [[u32; 2]; 3] {
    [
        sorted_edge(node_ids[0], node_ids[1]),
        sorted_edge(node_ids[1], node_ids[2]),
        sorted_edge(node_ids[2], node_ids[0]),
    ]
}

fn sorted_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

fn plc_node_topology_id(node_id: u32) -> TopologyEntityId {
    TopologyEntityId {
        stage: crate::contracts::MeshingStage::ProtectedBoundaryComplex,
        id: node_id.to_string(),
    }
}

fn solver_tetrahedron_source_surface_element_id(
    preparation: &SolidMeshPreparation,
    tetrahedron_index: usize,
) -> Option<u32> {
    preparation
        .solver_tetrahedron_mesh
        .boundary_faces
        .get(tetrahedron_index)
        .and_then(|face| face.face_id.id.parse::<u32>().ok())
}

pub(super) fn solid_validation_options(
    preparation: &SolidMeshPreparation,
    options: &VolumeMeshingOptions,
) -> AnalysisMeshValidationOptions {
    AnalysisMeshValidationOptions {
        quality: options.validation.quality,
        max_volume_element_count: Some(options.max_elements),
        max_volume_component_count: options
            .validation
            .max_volume_component_count
            .or(Some(preparation.tetrahedron_stage.volume_component_count)),
        coverage_sample_points_m: preparation
            .tetrahedron_stage
            .coverage_sample_points_m
            .clone(),
        min_coverage_sample_ratio: 1.0,
        expected_bounds_m: Some([
            preparation.topology.bounds_min_m,
            preparation.topology.bounds_max_m,
        ]),
        min_bounds_coverage_ratio: options.validation.min_bounds_coverage_ratio,
        expected_volume_m3: Some(preparation.tetrahedron_stage.expected_volume_m3),
        min_volume_coverage_ratio: options.validation.min_volume_coverage_ratio,
        expected_boundary_area_m2: Some(preparation.tetrahedron_stage.expected_boundary_area_m2),
        min_boundary_area_ratio: options.validation.min_boundary_area_ratio,
        min_boundary_face_recovery_ratio: options.validation.min_boundary_face_recovery_ratio,
        min_boundary_edge_recovery_ratio: options.validation.min_boundary_edge_recovery_ratio,
        require_no_fan_fallback: true,
        require_no_unrepaired_exact_quality: true,
        ..AnalysisMeshValidationOptions::default()
    }
}

pub(super) fn quality_report(
    elements: Vec<ElementQuality>,
    boundary_projection_errors_m: impl IntoIterator<Item = f64>,
) -> AnalysisMeshQualityReport {
    if elements.is_empty() {
        return AnalysisMeshQualityReport::default();
    }
    let mut boundary_projection_error_count = 0_usize;
    let mut boundary_projection_error_sum_m = 0.0_f64;
    let mut max_boundary_projection_error_m = 0.0_f64;
    for error_m in boundary_projection_errors_m {
        if !error_m.is_finite() {
            continue;
        }
        boundary_projection_error_count += 1;
        boundary_projection_error_sum_m += error_m;
        max_boundary_projection_error_m = max_boundary_projection_error_m.max(error_m);
    }
    let mean_boundary_projection_error_m = if boundary_projection_error_count == 0 {
        0.0
    } else {
        boundary_projection_error_sum_m / boundary_projection_error_count as f64
    };
    let min_scaled_jacobian = elements
        .iter()
        .map(|element| element.scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let min_exact_scaled_jacobian = elements
        .iter()
        .map(|element| element.exact_scaled_jacobian)
        .fold(f64::INFINITY, f64::min);
    let max_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .fold(0.0_f64, f64::max);
    let mean_aspect_ratio = elements
        .iter()
        .map(|element| element.aspect_ratio)
        .sum::<f64>()
        / elements.len() as f64;
    AnalysisMeshQualityReport {
        min_scaled_jacobian,
        min_exact_scaled_jacobian,
        mean_aspect_ratio,
        max_aspect_ratio,
        inverted_element_count: 0,
        mean_boundary_projection_error_m,
        max_boundary_projection_error_m,
        elements,
    }
}

fn exact_scaled_jacobian_below_threshold_count(quality: &AnalysisMeshQualityReport) -> usize {
    let threshold = QualityThresholds::default().min_scaled_jacobian;
    quality
        .elements
        .iter()
        .filter(|element| element.exact_scaled_jacobian < threshold)
        .count()
}

fn exact_scaled_jacobian_bins(quality: &AnalysisMeshQualityReport) -> BTreeMap<String, usize> {
    let mut bins = BTreeMap::<String, usize>::new();
    for element in &quality.elements {
        *bins
            .entry(scaled_jacobian_bin(element.exact_scaled_jacobian))
            .or_default() += 1;
    }
    bins
}

fn scaled_jacobian_bin(value: f64) -> String {
    if value < 0.0 {
        "lt_0".to_string()
    } else if value < 0.15 {
        "0_to_0_15".to_string()
    } else if value < 0.35 {
        "0_15_to_0_35".to_string()
    } else if value < 0.65 {
        "0_35_to_0_65".to_string()
    } else {
        "gte_0_65".to_string()
    }
}
