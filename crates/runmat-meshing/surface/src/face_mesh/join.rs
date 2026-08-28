use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_geometry_core::TopologicalOrientation;
use runmat_meshing_core::{StableDigest, SurfaceQualityTargets};

use crate::{
    validate_exact_face_chart_acceptance, ExactFaceAcceptanceOptions,
    ExactFaceChartAcceptanceReport, ExactFaceChartRefinedMesh, ExactFaceCharts,
    ExactFacePslgSegmentSource, ExactFaceRefinementContext,
};

use super::{
    admission::validate_exact_face_mesh_contract,
    identity::{canonical_triangle, exact_face_triangle_id, rotate},
    lineage::{node_edge_parameters, validate_trim_lineage},
    ExactFaceJoinContext, ExactFaceJoinError, ExactFaceJoinErrorKind, ExactFaceJoinOptions,
    ExactFaceMesh, ExactFaceMeshBoundarySegment, ExactFaceMeshJoinedCut, ExactFaceMeshNode,
    ExactFaceMeshNodeUse, ExactFaceMeshTriangle, EXACT_FACE_MESH_SCHEMA_VERSION,
};

pub fn join_exact_face_charts(
    charts: &ExactFaceCharts,
    refined: &[ExactFaceChartRefinedMesh],
    acceptance: &[ExactFaceChartAcceptanceReport],
    context: ExactFaceJoinContext<'_>,
    join_options: ExactFaceJoinOptions,
) -> Result<ExactFaceMesh, ExactFaceJoinError> {
    validate_options(charts, join_options)?;
    if charts.charts.is_empty()
        || refined.len() != charts.charts.len()
        || acceptance.len() != charts.charts.len()
    {
        return Err(invalid(
            &charts.source_face_id,
            "chart, refined-mesh, and acceptance inventories differ",
        ));
    }
    let face_orientation = context
        .refinement
        .topology
        .faces
        .iter()
        .find(|face| face.id == charts.source_face_id)
        .ok_or_else(|| {
            invalid(
                &charts.source_face_id,
                "joined face is absent from exact topology",
            )
        })?
        .orientation;

    let mut nodes = BTreeMap::<StableDigest, NodeBuilder>::new();
    let mut triangles = Vec::new();
    let mut triangle_keys = BTreeSet::new();
    let mut boundary_segments = Vec::new();
    let mut boundary_keys = BTreeSet::new();
    let mut cut_pieces =
        BTreeMap::<(StableDigest, [StableDigest; 2]), Vec<[StableDigest; 2]>>::new();
    let mut cut_ids = BTreeSet::new();
    let mut maximum_chordal_deviation_m = 0.0_f64;
    let mut maximum_normal_deviation_rad = 0.0_f64;

    for ((chart, mesh), report) in charts.charts.iter().zip(refined).zip(acceptance) {
        validate_chart_inputs(
            charts,
            chart,
            mesh,
            report,
            context.refinement,
            context.quality,
            context.acceptance,
        )?;
        validate_trim_lineage(chart, mesh)?;
        let edge_parameters = node_edge_parameters(mesh)?;
        for (vertex, pslg_vertex) in mesh
            .mesh
            .geometry
            .vertices
            .iter()
            .zip(&mesh.mesh.topology.pslg.vertices)
        {
            if nodes.len() as u64 >= join_options.maximum_nodes
                && !nodes.contains_key(&pslg_vertex.node_id)
            {
                return Err(limit(
                    &charts.source_face_id,
                    "exact face join exceeded its node hard limit",
                ));
            }
            let use_record = ExactFaceMeshNodeUse {
                source_face_id: charts.source_face_id.clone(),
                chart_id: chart.chart_id,
                uv: vertex.evaluation.uv,
                evaluator_uv: vertex.evaluation.evaluator_uv,
                exact_edge_parameters: edge_parameters
                    .get(&vertex.pslg_vertex_index)
                    .cloned()
                    .unwrap_or_default(),
            };
            match nodes.entry(pslg_vertex.node_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(NodeBuilder {
                        point_m: vertex.evaluation.point_m,
                        uses: vec![use_record],
                    });
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if distance(entry.get().point_m, vertex.evaluation.point_m)
                        > join_options.coordinate_tolerance_m
                    {
                        return Err(invalid(
                            &charts.source_face_id,
                            "shared node images disagree in 3D",
                        )
                        .with_chart(chart.chart_id));
                    }
                    if entry.get().uses.iter().any(|existing| {
                        existing.chart_id == use_record.chart_id && existing.uv == use_record.uv
                    }) {
                        return Err(invalid(
                            &charts.source_face_id,
                            "chart contains a duplicate node image",
                        )
                        .with_chart(chart.chart_id));
                    }
                    entry.get_mut().uses.push(use_record);
                }
            }
        }

        for (geometry, accepted) in mesh
            .mesh
            .geometry
            .triangles
            .iter()
            .zip(&report.acceptance.triangles)
        {
            if triangles.len() as u64 >= join_options.maximum_triangles {
                return Err(limit(
                    &charts.source_face_id,
                    "exact face join exceeded its triangle hard limit",
                ));
            }
            let node_ids = geometry
                .triangle
                .vertex_indices
                .map(|index| mesh.mesh.topology.pslg.vertices[index as usize].node_id);
            let mut duplicate_key = node_ids;
            duplicate_key.sort();
            if duplicate_key[0] == duplicate_key[1]
                || duplicate_key[1] == duplicate_key[2]
                || !triangle_keys.insert(duplicate_key)
            {
                return Err(invalid(
                    &charts.source_face_id,
                    "joined face contains a collapsed or duplicate facet",
                )
                .with_chart(chart.chart_id));
            }
            let (node_ids, metric_edge_lengths, unit_normal) = match face_orientation {
                TopologicalOrientation::Forward => {
                    (node_ids, geometry.metric_edge_lengths, geometry.unit_normal)
                }
                TopologicalOrientation::Reversed => (
                    [node_ids[0], node_ids[2], node_ids[1]],
                    [
                        geometry.metric_edge_lengths[2],
                        geometry.metric_edge_lengths[1],
                        geometry.metric_edge_lengths[0],
                    ],
                    geometry.unit_normal.map(|value| -value),
                ),
            };
            let (node_ids, rotation) = canonical_triangle(node_ids);
            triangles.push(ExactFaceMeshTriangle {
                triangle_id: exact_face_triangle_id(chart.chart_id, node_ids),
                chart_id: chart.chart_id,
                source_face_id: charts.source_face_id.clone(),
                node_ids,
                unit_normal,
                physical_area_m2: geometry.physical_area_m2,
                metric_edge_lengths: rotate(metric_edge_lengths, rotation),
                minimum_metric_angle_rad: geometry.minimum_metric_angle_rad,
                physical_aspect_ratio: geometry.physical_aspect_ratio,
                chordal_deviation_m: geometry.chordal_deviation_m,
                normal_deviation_rad: geometry.normal_deviation_rad,
                acceptance_sample_count: accepted.sample_count,
                accepted_chordal_deviation_m: accepted.maximum_chordal_deviation_m,
                accepted_normal_deviation_rad: accepted.maximum_normal_deviation_rad,
            });
        }
        maximum_chordal_deviation_m =
            maximum_chordal_deviation_m.max(report.acceptance.maximum_chordal_deviation_m);
        maximum_normal_deviation_rad =
            maximum_normal_deviation_rad.max(report.acceptance.maximum_normal_deviation_rad);
        collect_segments(
            charts,
            mesh,
            join_options,
            &mut boundary_segments,
            &mut boundary_keys,
            &mut cut_pieces,
            &mut cut_ids,
        )?;
    }
    validate_cut_pieces(&charts.source_face_id, &cut_pieces)?;
    let result = ExactFaceMesh {
        schema_version: EXACT_FACE_MESH_SCHEMA_VERSION,
        source_face_id: charts.source_face_id.clone(),
        nodes: nodes
            .into_iter()
            .map(|(node_id, builder)| ExactFaceMeshNode {
                node_id,
                point_m: builder.point_m,
                uses: builder.uses,
            })
            .collect(),
        triangles,
        boundary_segments,
        joined_chart_cuts: cut_ids
            .into_iter()
            .map(|cut_id| {
                let piece_count = cut_pieces
                    .keys()
                    .filter(|(candidate, _)| *candidate == cut_id)
                    .count()
                    .try_into()
                    .map_err(|_| {
                        limit(
                            &charts.source_face_id,
                            "exact face join cut-piece count exceeds its representation",
                        )
                    })?;
                Ok(ExactFaceMeshJoinedCut {
                    cut_id,
                    piece_count,
                })
            })
            .collect::<Result<Vec<_>, ExactFaceJoinError>>()?,
        maximum_chordal_deviation_m,
        maximum_normal_deviation_rad,
    };
    validate_exact_face_mesh_contract(&result, context.refinement.topology)?;
    Ok(result)
}

pub fn validate_exact_face_mesh(
    result: &ExactFaceMesh,
    charts: &ExactFaceCharts,
    refined: &[ExactFaceChartRefinedMesh],
    acceptance: &[ExactFaceChartAcceptanceReport],
    context: ExactFaceJoinContext<'_>,
    join_options: ExactFaceJoinOptions,
) -> Result<(), ExactFaceJoinError> {
    let expected = join_exact_face_charts(charts, refined, acceptance, context, join_options)?;
    if result != &expected {
        return Err(invalid(
            &charts.source_face_id,
            "exact face mesh differs from canonical chart join",
        ));
    }
    Ok(())
}

#[derive(Clone)]
struct NodeBuilder {
    point_m: [f64; 3],
    uses: Vec<ExactFaceMeshNodeUse>,
}

fn validate_options(
    charts: &ExactFaceCharts,
    options: ExactFaceJoinOptions,
) -> Result<(), ExactFaceJoinError> {
    if !options.coordinate_tolerance_m.is_finite()
        || options.coordinate_tolerance_m <= 0.0
        || options.maximum_nodes == 0
        || options.maximum_triangles == 0
        || options.maximum_boundary_segments == 0
    {
        return Err(ExactFaceJoinError::new(
            ExactFaceJoinErrorKind::InvalidOptions,
            &charts.source_face_id,
            "face join tolerance and hard limits must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_chart_inputs(
    charts: &ExactFaceCharts,
    chart: &crate::ExactFaceChart,
    mesh: &ExactFaceChartRefinedMesh,
    report: &ExactFaceChartAcceptanceReport,
    context: ExactFaceRefinementContext<'_>,
    quality: SurfaceQualityTargets,
    acceptance_options: ExactFaceAcceptanceOptions,
) -> Result<(), ExactFaceJoinError> {
    if chart.source_face_id != charts.source_face_id
        || mesh.chart_id != chart.chart_id
        || report.chart_id != chart.chart_id
        || mesh.mesh.geometry.source_face_id != charts.source_face_id
    {
        return Err(invalid(
            &charts.source_face_id,
            "face join chart identities or source face are inconsistent",
        )
        .with_chart(chart.chart_id));
    }
    validate_exact_face_chart_acceptance(report, chart, mesh, context, quality, acceptance_options)
        .map_err(|error| {
            ExactFaceJoinError::new(
                ExactFaceJoinErrorKind::Acceptance(error.kind),
                &charts.source_face_id,
                error.reason,
            )
            .with_chart(chart.chart_id)
        })
}

fn collect_segments(
    charts: &ExactFaceCharts,
    mesh: &ExactFaceChartRefinedMesh,
    options: ExactFaceJoinOptions,
    boundary_segments: &mut Vec<ExactFaceMeshBoundarySegment>,
    boundary_keys: &mut BTreeSet<(PersistentEntityId, PersistentEntityId, [StableDigest; 2])>,
    cut_pieces: &mut BTreeMap<(StableDigest, [StableDigest; 2]), Vec<[StableDigest; 2]>>,
    cut_ids: &mut BTreeSet<StableDigest>,
) -> Result<(), ExactFaceJoinError> {
    let pslg = &mesh.mesh.topology.pslg;
    for segment in &pslg.segments {
        let node_ids = segment
            .vertex_indices
            .map(|index| pslg.vertices[index as usize].node_id);
        match &segment.source {
            ExactFacePslgSegmentSource::ExactTrim {
                source_coedge_id,
                source_edge_id,
            } => {
                if boundary_segments.len() as u64 >= options.maximum_boundary_segments {
                    return Err(limit(
                        &charts.source_face_id,
                        "exact face join exceeded its boundary-segment hard limit",
                    ));
                }
                let Some(edge_parameters) = segment.edge_parameters else {
                    return Err(invalid(
                        &charts.source_face_id,
                        "joined exact trim is missing edge parameters",
                    ));
                };
                if !boundary_keys.insert((
                    source_coedge_id.clone(),
                    source_edge_id.clone(),
                    node_ids,
                )) {
                    return Err(invalid(
                        &charts.source_face_id,
                        "joined face contains a duplicate exact boundary segment",
                    ));
                }
                boundary_segments.push(ExactFaceMeshBoundarySegment {
                    source_coedge_id: source_coedge_id.clone(),
                    source_edge_id: source_edge_id.clone(),
                    node_ids,
                    edge_parameters,
                });
            }
            ExactFacePslgSegmentSource::ChartCut { cut_id } => {
                if *cut_id == StableDigest::ZERO || segment.edge_parameters.is_some() {
                    return Err(invalid(
                        &charts.source_face_id,
                        "chart cut has invalid identity or exact-edge parameters",
                    ));
                }
                let mut unordered = node_ids;
                unordered.sort();
                cut_ids.insert(*cut_id);
                cut_pieces
                    .entry((*cut_id, unordered))
                    .or_default()
                    .push(node_ids);
            }
        }
    }
    Ok(())
}

fn validate_cut_pieces(
    face_id: &PersistentEntityId,
    pieces: &BTreeMap<(StableDigest, [StableDigest; 2]), Vec<[StableDigest; 2]>>,
) -> Result<(), ExactFaceJoinError> {
    if pieces
        .values()
        .any(|images| images.len() != 2 || images[0] != [images[1][1], images[1][0]])
    {
        return Err(invalid(
            face_id,
            "chart-cut pieces do not have exactly two reversed periodic images",
        ));
    }
    Ok(())
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>()
        .sqrt()
}

fn invalid(face_id: &PersistentEntityId, reason: &str) -> ExactFaceJoinError {
    ExactFaceJoinError::new(ExactFaceJoinErrorKind::InvalidInput, face_id, reason)
}

fn limit(face_id: &PersistentEntityId, reason: &str) -> ExactFaceJoinError {
    ExactFaceJoinError::new(ExactFaceJoinErrorKind::ResourceLimit, face_id, reason)
}
