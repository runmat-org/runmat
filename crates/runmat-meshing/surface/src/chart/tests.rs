use runmat_geometry_core::{
    ExactSurfaceDefinition, ExactSurfaceImplementation, GeometryEvaluationControl,
    GeometryEvaluationError, GeometryModel, ParameterRange, PersistentEntityId,
    PersistentEntityKind, PortableExactEvaluator, TopologicalOrientation,
};
use runmat_meshing_core::{
    MetricCombinationRule, MetricFieldRequest, MetricTensor3, NeverCancelled, StableDigest,
    SurfaceQualityTargets,
};

use crate::{
    ExactFaceBoundary, ExactFaceBoundaryLoop, ExactFaceBoundarySegment, ExactFacePslgSegmentSource,
};

use super::*;

#[test]
fn periodic_seam_images_lift_into_one_canonical_chart() {
    let (document, mut topology, mut registry) = runmat_geometry_fixtures::exact_circle();
    topology.faces[0].periodic_u = true;
    registry.surfaces[0].implementation = ExactSurfaceImplementation::Portable {
        definition: ExactSurfaceDefinition::Cylinder {
            origin_m: [0.0; 3],
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0],
            axis_m_per_v: [0.0, 0.0, 1.0],
            radius_m: 1.0,
            domains: [
                range(-std::f64::consts::PI, std::f64::consts::PI),
                range(-2.0, 2.0),
            ],
        },
    };
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let source = boundary(
        &topology,
        [[3.0, 0.0], [-3.0, 0.0], [-3.0, 1.0], [3.0, 1.0]],
    );

    let charts = build_exact_face_charts(
        &source,
        &topology,
        &evaluator,
        &Control,
        ExactFaceChartOptions::default(),
    )
    .unwrap();
    assert_eq!(charts.source_face_id, source.source_face_id);
    assert_eq!(charts.charts.len(), 1);
    assert_ne!(charts.charts[0].chart_id, StableDigest::ZERO);
    let chart = &charts.charts[0];
    assert!(chart
        .boundary
        .outer_loop
        .segments
        .windows(2)
        .all(|pair| pair[0].node_uv[1] == pair[1].node_uv[0]));
    assert_eq!(
        chart.boundary.outer_loop.segments.last().unwrap().node_uv[1],
        chart.boundary.outer_loop.segments[0].node_uv[0]
    );
    assert_eq!(chart.pslg.vertices.len(), 4);
    validate_exact_face_charts(
        &charts,
        &source,
        &topology,
        &evaluator,
        &Control,
        ExactFaceChartOptions::default(),
    )
    .unwrap();

    let winding = boundary(
        &topology,
        [
            [0.0, 0.0],
            [std::f64::consts::FRAC_PI_2, 0.0],
            [std::f64::consts::PI, 0.0],
            [-std::f64::consts::FRAC_PI_2, 0.0],
        ],
    );
    assert_eq!(
        build_exact_face_charts(
            &winding,
            &topology,
            &evaluator,
            &Control,
            ExactFaceChartOptions::default(),
        )
        .unwrap_err()
        .kind,
        ExactFaceChartErrorKind::RequiresMultipleCharts
    );

    let mut annulus = winding;
    let mut inner = boundary(
        &topology,
        [
            [0.0, 1.0],
            [-std::f64::consts::FRAC_PI_2, 1.0],
            [-std::f64::consts::PI, 1.0],
            [std::f64::consts::FRAC_PI_2, 1.0],
        ],
    )
    .outer_loop;
    inner.source_wire_id = entity(PersistentEntityKind::Wire, "inner-wire");
    inner.orientation = TopologicalOrientation::Reversed;
    for (index, segment) in inner.segments.iter_mut().enumerate() {
        segment.node_ids = [node(index as u8 + 5), node((index as u8 + 1) % 4 + 5)];
    }
    annulus.inner_loops.push(inner);
    let annulus_charts = build_exact_face_charts(
        &annulus,
        &topology,
        &evaluator,
        &Control,
        ExactFaceChartOptions::default(),
    )
    .unwrap();
    let annulus_pslg = &annulus_charts.charts[0].pslg;
    assert_eq!(annulus_pslg.loops.len(), 1);
    assert_eq!(annulus_pslg.segments.len(), 10);
    let cuts = annulus_pslg
        .segments
        .iter()
        .filter_map(|segment| match &segment.source {
            ExactFacePslgSegmentSource::ChartCut { cut_id } => Some(*cut_id),
            ExactFacePslgSegmentSource::ExactTrim { .. } => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(cuts.len(), 2);
    assert_eq!(cuts[0], cuts[1]);
    assert!(annulus_pslg
        .segments
        .iter()
        .zip(annulus_pslg.segments.iter().cycle().skip(1))
        .all(|(left, right)| left.vertex_indices[1] == right.vertex_indices[0]));

    let delaunay_context = ExactFaceChartDelaunayContext {
        topology: &topology,
        evaluator: &evaluator,
        geometry_control: &Control,
        cancellation: &NeverCancelled,
    };
    let triangulations = triangulate_exact_face_charts(
        &annulus_charts,
        &annulus,
        delaunay_context,
        ExactFaceChartOptions::default(),
        crate::ExactFaceDelaunayOptions::default(),
    )
    .unwrap();
    assert_eq!(triangulations.len(), 1);
    assert!(!triangulations[0].triangulation.triangles.is_empty());
    validate_exact_face_chart_delaunay(
        &triangulations,
        &annulus_charts,
        &annulus,
        delaunay_context,
        ExactFaceChartOptions::default(),
        crate::ExactFaceDelaunayOptions::default(),
    )
    .unwrap();
    let domains = recover_exact_face_chart_domains(
        &triangulations,
        &annulus_charts,
        &annulus,
        delaunay_context,
        ExactFaceChartOptions::default(),
        crate::ExactFaceDelaunayOptions::default(),
    )
    .unwrap();
    assert_eq!(domains.len(), 1);
    assert_eq!(domains[0].constrained.protected_segments.len(), 10);
    assert!(!domains[0].trimmed.triangles.is_empty());
    validate_exact_face_chart_domains(
        &domains,
        &annulus_charts,
        &annulus,
        delaunay_context,
        ExactFaceChartOptions::default(),
        crate::ExactFaceDelaunayOptions::default(),
    )
    .unwrap();
    let metric_request = MetricFieldRequest {
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(100.0).unwrap(),
        maximum_grading_ratio: 2.0,
        contributions: Vec::new(),
    };
    let quality = SurfaceQualityTargets {
        minimum_metric_angle_degrees: 0.01,
        maximum_physical_aspect_ratio: 1.0e9,
        maximum_chordal_deviation_m: 1.0e9,
        maximum_normal_deviation_degrees: 180.0,
    };
    let refinement_context = crate::ExactFaceRefinementContext::new(
        &topology,
        &metric_request,
        &evaluator,
        &Control,
        &NeverCancelled,
    );
    let refined = crate::refine_exact_face_chart_until_blocked(
        &annulus_charts.charts[0],
        &domains[0],
        refinement_context,
        crate::ExactFaceRefinementPolicy {
            quality,
            delaunay: crate::ExactFaceDelaunayOptions::default(),
            refinement: crate::ExactFaceRefinementOptions {
                maximum_interior_insertions: 1,
            },
        },
        crate::ExactFaceChartRefinementOptions {
            maximum_chart_cut_splits: 1,
        },
    )
    .unwrap();
    let crate::ExactFaceChartRefinementOutcome::Converged(refined) = refined else {
        panic!("permissive periodic-annulus refinement must converge")
    };
    assert!(refined
        .mesh
        .geometry
        .vertices
        .iter()
        .any(|vertex| vertex.evaluation.uv != vertex.evaluation.evaluator_uv));
    let acceptance_options = crate::ExactFaceAcceptanceOptions {
        minimum_subdivision_depth: 1,
        maximum_subdivision_depth: 2,
        refinement_margin_ratio: 0.5,
        maximum_samples: 100_000,
    };
    let acceptance = crate::accept_exact_face_chart_mesh(
        &refined,
        refinement_context,
        quality,
        acceptance_options,
    )
    .unwrap();
    assert_eq!(acceptance.chart_id, annulus_charts.charts[0].chart_id);
    assert!(acceptance.acceptance.sample_count > 0);
    let join_context =
        crate::ExactFaceJoinContext::new(refinement_context, quality, acceptance_options);
    let joined = crate::join_exact_face_charts(
        &annulus_charts,
        std::slice::from_ref(refined.as_ref()),
        std::slice::from_ref(&acceptance),
        join_context,
        crate::ExactFaceJoinOptions::default(),
    )
    .unwrap();
    assert_eq!(joined.source_face_id, annulus.source_face_id);
    assert_eq!(joined.boundary_segments.len(), 8);
    assert_eq!(joined.joined_chart_cuts.len(), 1);
    assert_eq!(joined.joined_chart_cuts[0].piece_count, 1);
    assert!(joined
        .nodes
        .iter()
        .any(|node| node.uses.len() == 2 && node.uses[0].uv != node.uses[1].uv));
    assert!(joined
        .triangles
        .iter()
        .all(|triangle| triangle.triangle_id != StableDigest::ZERO));
    let mut reversed_topology = topology.clone();
    reversed_topology.faces[0].orientation = TopologicalOrientation::Reversed;
    let reversed_evaluator =
        PortableExactEvaluator::new(&registry, &reversed_topology, model).unwrap();
    let reversed_joined = crate::join_exact_face_charts(
        &annulus_charts,
        std::slice::from_ref(refined.as_ref()),
        std::slice::from_ref(&acceptance),
        crate::ExactFaceJoinContext::new(
            crate::ExactFaceRefinementContext::new(
                &reversed_topology,
                &metric_request,
                &reversed_evaluator,
                &Control,
                &NeverCancelled,
            ),
            quality,
            acceptance_options,
        ),
        crate::ExactFaceJoinOptions::default(),
    )
    .unwrap();
    assert!(joined
        .triangles
        .iter()
        .zip(&reversed_joined.triangles)
        .all(|(forward, reversed)| {
            forward.triangle_id != reversed.triangle_id
                && forward
                    .unit_normal
                    .into_iter()
                    .zip(reversed.unit_normal)
                    .map(|(left, right)| left * right)
                    .sum::<f64>()
                    < -0.999_999
        }));
    let face_partitions = crate::face_partition_descriptors(&topology, 1).unwrap();
    assert_eq!(face_partitions.len(), 1);
    let face_batch = crate::build_exact_face_mesh_batch(
        &topology,
        face_partitions[0].clone(),
        vec![joined.clone()],
    )
    .unwrap();
    crate::validate_exact_face_mesh_batch(&face_batch, &topology).unwrap();
    let encoded_batch = crate::encode_exact_face_mesh_batch(&face_batch, &topology).unwrap();
    assert_eq!(
        crate::decode_exact_face_mesh_batch(&encoded_batch, &topology).unwrap(),
        face_batch
    );
    assert_eq!(
        crate::surface_mesh::decode_with_byte_limit(
            &encoded_batch,
            &topology,
            encoded_batch.len() - 1,
        )
        .unwrap_err()
        .kind,
        crate::ExactSurfaceMeshErrorKind::InvalidEncoding
    );
    let mut corrupt_batch = encoded_batch;
    corrupt_batch.push(0);
    assert_eq!(
        crate::decode_exact_face_mesh_batch(&corrupt_batch, &topology)
            .unwrap_err()
            .kind,
        crate::ExactSurfaceMeshErrorKind::InvalidEncoding
    );
    let mut tampered_batch = face_batch;
    tampered_batch.schema_version += 1;
    assert_eq!(
        crate::validate_exact_face_mesh_batch(&tampered_batch, &topology)
            .unwrap_err()
            .kind,
        crate::ExactSurfaceMeshErrorKind::InvalidInput
    );
    assert_eq!(
        crate::face_partition_descriptors(&topology, 0)
            .unwrap_err()
            .kind,
        crate::ExactSurfaceMeshErrorKind::InvalidOptions
    );
    assert_eq!(
        crate::join_exact_face_charts(
            &annulus_charts,
            std::slice::from_ref(refined.as_ref()),
            std::slice::from_ref(&acceptance),
            join_context,
            crate::ExactFaceJoinOptions {
                maximum_nodes: 1,
                ..crate::ExactFaceJoinOptions::default()
            },
        )
        .unwrap_err()
        .kind,
        crate::ExactFaceJoinErrorKind::ResourceLimit
    );
    crate::validate_exact_face_mesh(
        &joined,
        &annulus_charts,
        std::slice::from_ref(refined.as_ref()),
        std::slice::from_ref(&acceptance),
        join_context,
        crate::ExactFaceJoinOptions::default(),
    )
    .unwrap();
    let mut tampered_join = joined;
    tampered_join.nodes[0].point_m[0] += 0.25;
    assert_eq!(
        crate::validate_exact_face_mesh(
            &tampered_join,
            &annulus_charts,
            std::slice::from_ref(refined.as_ref()),
            std::slice::from_ref(&acceptance),
            join_context,
            crate::ExactFaceJoinOptions::default(),
        )
        .unwrap_err()
        .kind,
        crate::ExactFaceJoinErrorKind::InvalidInput
    );
    let mut unpaired_cut = (*refined).clone();
    let cut_index = unpaired_cut
        .mesh
        .topology
        .pslg
        .segments
        .iter()
        .position(|segment| matches!(segment.source, ExactFacePslgSegmentSource::ChartCut { .. }))
        .unwrap();
    unpaired_cut.mesh.topology.pslg.segments.remove(cut_index);
    assert_eq!(
        crate::join_exact_face_charts(
            &annulus_charts,
            std::slice::from_ref(&unpaired_cut),
            std::slice::from_ref(&acceptance),
            join_context,
            crate::ExactFaceJoinOptions::default(),
        )
        .unwrap_err()
        .kind,
        crate::ExactFaceJoinErrorKind::InvalidInput
    );
    let cut_segments = annulus_pslg
        .segments
        .iter()
        .enumerate()
        .filter(|(_, segment)| {
            matches!(&segment.source, ExactFacePslgSegmentSource::ChartCut { .. })
        })
        .collect::<Vec<_>>();
    let cut_id = match &cut_segments[0].1.source {
        ExactFacePslgSegmentSource::ChartCut { cut_id } => *cut_id,
        ExactFacePslgSegmentSource::ExactTrim { .. } => unreachable!(),
    };
    let images = cut_segments
        .iter()
        .map(|(index, segment)| crate::ExactChartCutSplitImage {
            pslg_segment_index: *index as u32,
            vertex_indices: segment.vertex_indices,
            midpoint_uv: midpoint(
                segment
                    .vertex_indices
                    .map(|vertex| annulus_pslg.vertices[vertex as usize].uv),
            ),
        })
        .collect::<Vec<_>>();
    let endpoint_node_ids = cut_segments[0]
        .1
        .vertex_indices
        .map(|vertex| annulus_pslg.vertices[vertex as usize].node_id);
    let split = crate::ExactChartCutSplit {
        source_face_id: annulus.source_face_id.clone(),
        cut_id,
        node_id: crate::exact_face_chart_cut_node_id(cut_id, endpoint_node_ids),
        images: [images[0], images[1]],
    };
    let initial_refined = crate::ExactFaceRefinedTopology {
        pslg: annulus_pslg.clone(),
        constrained: domains[0].constrained.clone(),
        trimmed: domains[0].trimmed.clone(),
    };
    let split_refined = crate::split_exact_face_chart_cut(
        &initial_refined,
        &split,
        &NeverCancelled,
        crate::ExactFaceDelaunayOptions::default(),
    )
    .unwrap();
    assert_eq!(
        split_refined.pslg.vertices.len(),
        annulus_pslg.vertices.len() + 2
    );
    assert_eq!(
        split_refined.pslg.segments.len(),
        annulus_pslg.segments.len() + 2
    );
    assert_eq!(
        split_refined
            .pslg
            .segments
            .iter()
            .filter(|segment| {
                matches!(&segment.source, ExactFacePslgSegmentSource::ChartCut { .. })
            })
            .count(),
        4
    );
    crate::validate_exact_face_chart_cut_split_result(
        &split_refined,
        &initial_refined,
        &split,
        &NeverCancelled,
        crate::ExactFaceDelaunayOptions::default(),
    )
    .unwrap();
    let mut tampered_split_refined = split_refined;
    tampered_split_refined
        .pslg
        .vertices
        .iter_mut()
        .find(|vertex| vertex.node_id == split.node_id)
        .unwrap()
        .uv[0] += 0.25;
    assert_eq!(
        crate::validate_exact_face_chart_cut_split_result(
            &tampered_split_refined,
            &initial_refined,
            &split,
            &NeverCancelled,
            crate::ExactFaceDelaunayOptions::default(),
        )
        .unwrap_err()
        .kind,
        crate::ExactFaceRefinementErrorKind::InvalidGeometry
    );
    let mut tampered_domains = domains;
    tampered_domains[0].constrained.protected_segments.pop();
    assert!(matches!(
        validate_exact_face_chart_domains(
            &tampered_domains,
            &annulus_charts,
            &annulus,
            delaunay_context,
            ExactFaceChartOptions::default(),
            crate::ExactFaceDelaunayOptions::default(),
        )
        .unwrap_err()
        .kind,
        ExactFaceChartErrorKind::Delaunay(_)
    ));
    let mut tampered_triangulations = triangulations;
    tampered_triangulations[0].triangulation.triangles[0]
        .vertex_indices
        .swap(1, 2);
    assert!(matches!(
        validate_exact_face_chart_delaunay(
            &tampered_triangulations,
            &annulus_charts,
            &annulus,
            delaunay_context,
            ExactFaceChartOptions::default(),
            crate::ExactFaceDelaunayOptions::default(),
        )
        .unwrap_err()
        .kind,
        ExactFaceChartErrorKind::Delaunay(_)
    ));

    let mut tampered_annulus = annulus_charts;
    let cut = tampered_annulus.charts[0]
        .pslg
        .segments
        .iter_mut()
        .find(|segment| matches!(&segment.source, ExactFacePslgSegmentSource::ChartCut { .. }))
        .unwrap();
    cut.source = ExactFacePslgSegmentSource::ChartCut { cut_id: node(99) };
    assert_eq!(
        validate_exact_face_charts(
            &tampered_annulus,
            &annulus,
            &topology,
            &evaluator,
            &Control,
            ExactFaceChartOptions::default(),
        )
        .unwrap_err()
        .kind,
        ExactFaceChartErrorKind::InvalidInput
    );
}

#[test]
fn nonperiodic_chart_is_identity_and_tamper_evident() {
    let (document, topology, registry) = runmat_geometry_fixtures::exact_circle();
    let GeometryModel::ExactBRep { model } = &document.model else {
        panic!("fixture must be exact")
    };
    let evaluator = PortableExactEvaluator::new(&registry, &topology, model).unwrap();
    let source = boundary(&topology, [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
    let options = ExactFaceChartOptions::default();
    let charts =
        build_exact_face_charts(&source, &topology, &evaluator, &Control, options).unwrap();
    let chart = &charts.charts[0];
    assert_eq!(chart.boundary, source);
    assert_eq!(chart.periodicity, [None, None]);

    let mut tampered = charts;
    tampered.charts[0].boundary.outer_loop.segments[0].node_uv[0][0] += 0.25;
    assert_eq!(
        validate_exact_face_charts(&tampered, &source, &topology, &evaluator, &Control, options,)
            .unwrap_err()
            .kind,
        ExactFaceChartErrorKind::InvalidInput
    );

    let invalid_options = ExactFaceChartOptions {
        maximum_charts_per_face: 0,
        ..options
    };
    assert_eq!(
        build_exact_face_charts(&source, &topology, &evaluator, &Control, invalid_options,)
            .unwrap_err()
            .kind,
        ExactFaceChartErrorKind::InvalidOptions
    );
}

fn boundary(
    topology: &runmat_geometry_core::ExactBRepTopology,
    uv: [[f64; 2]; 4],
) -> ExactFaceBoundary {
    let nodes = [node(1), node(2), node(3), node(4)];
    ExactFaceBoundary {
        source_face_id: topology.faces[0].id.clone(),
        outer_loop: ExactFaceBoundaryLoop {
            source_wire_id: topology.wires[0].id.clone(),
            orientation: TopologicalOrientation::Forward,
            segments: (0..4)
                .map(|index| ExactFaceBoundarySegment {
                    source_coedge_id: topology.coedges[0].id.clone(),
                    source_edge_id: topology.edges[0].id.clone(),
                    seam_image: Some(0),
                    node_ids: [nodes[index], nodes[(index + 1) % 4]],
                    edge_parameters: [index as f64, index as f64 + 1.0],
                    node_uv: [uv[index], uv[(index + 1) % 4]],
                })
                .collect(),
        },
        inner_loops: Vec::new(),
    }
}

fn range(start: f64, end: f64) -> ParameterRange {
    ParameterRange { start, end }
}

fn node(value: u8) -> StableDigest {
    StableDigest::from_bytes([value; 32])
}

fn midpoint(endpoints: [[f64; 2]; 2]) -> [f64; 2] {
    [
        endpoints[0][0] * 0.5 + endpoints[1][0] * 0.5,
        endpoints[0][1] * 0.5 + endpoints[1][1] * 0.5,
    ]
}

fn entity(kind: PersistentEntityKind, source_topology_id: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source_topology_id.to_owned(),
        assembly_path: Vec::new(),
    }
}

struct Control;

impl GeometryEvaluationControl for Control {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}
