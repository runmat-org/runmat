use super::chart_cut::split_exact_face_chart_cut_with_control;
use super::insertion::insert_exact_face_refinement_candidates_with_control;
use crate::{
    acceptance::sample_exact_face_mesh_in_parameterization,
    evaluate_exact_face_geometry_in_parameterization,
    validate_exact_face_geometry_in_parameterization, ExactFaceAcceptanceError,
    ExactFaceAcceptanceErrorKind, ExactFaceBoundary, ExactFaceChart,
    ExactFaceChartParameterization, ExactFaceGeometryError,
};
use runmat_geometry_core::GeometryAllocationScope;

use super::{
    derive_exact_face_feature_collars, insert_validated_face_refinement_candidates,
    select_exact_face_refinement_candidates, separate_refinement_candidate, ExactFaceRefinedMesh,
    ExactFaceRefinedTopology, ExactFaceRefinementCandidate, ExactFaceRefinementContext,
    ExactFaceRefinementError, ExactFaceRefinementErrorKind, ExactFaceRefinementOutcome,
    ExactFaceRefinementPolicy, ExactFaceRefinementReason,
};

#[derive(Clone, Copy)]
pub(crate) enum ExactFaceRefinementDomain<'a> {
    ExactBoundary(&'a ExactFaceBoundary),
    Chart(&'a ExactFaceChart),
}

impl ExactFaceRefinementDomain<'_> {
    fn parameterization(&self) -> &ExactFaceChartParameterization {
        match self {
            Self::ExactBoundary(_) => &ExactFaceChartParameterization::EvaluatorParameters,
            Self::Chart(chart) => &chart.parameterization,
        }
    }
}

pub fn refine_exact_face_until_blocked(
    boundary: &ExactFaceBoundary,
    initial: &ExactFaceRefinedTopology,
    context: ExactFaceRefinementContext<'_>,
    policy: ExactFaceRefinementPolicy,
) -> Result<ExactFaceRefinementOutcome, ExactFaceRefinementError> {
    refine_exact_face_domain_until_blocked(
        ExactFaceRefinementDomain::ExactBoundary(boundary),
        initial,
        context,
        policy,
        0,
    )
    .map(|(outcome, _)| outcome)
}

pub(crate) fn refine_exact_face_domain_until_blocked(
    domain: ExactFaceRefinementDomain<'_>,
    initial: &ExactFaceRefinedTopology,
    context: ExactFaceRefinementContext<'_>,
    policy: ExactFaceRefinementPolicy,
    maximum_chart_cut_splits: u32,
) -> Result<(ExactFaceRefinementOutcome, u32), ExactFaceRefinementError> {
    policy.quality.validate().map_err(|error| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidQuality,
            &initial.pslg.source_face_id,
            error.to_string(),
        )
    })?;
    if policy.refinement.maximum_interior_insertions == 0 || policy.delaunay.validate().is_err() {
        return Err(ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidOptions,
            &initial.pslg.source_face_id,
            "refinement and Delaunay operational limits must be nonzero",
        ));
    }
    if matches!(domain, ExactFaceRefinementDomain::Chart(_)) && maximum_chart_cut_splits == 0 {
        return Err(ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::InvalidOptions,
            &initial.pslg.source_face_id,
            "chart-cut split hard limit must be nonzero",
        ));
    }
    let field = context
        .geometry_context()
        .resolve_metric()
        .map_err(|failure| {
            ExactFaceRefinementError::new(
                ExactFaceRefinementErrorKind::Metric(failure.kind),
                &initial.pslg.source_face_id,
                failure.reason,
            )
        })?;
    let context = ExactFaceRefinementContext {
        metric_field: Some(&field),
        ..context
    };
    let mut current = initial.clone();
    let mut insertion_count = 0u32;
    let mut chart_cut_split_count = 0u32;
    // Encroached boundary segments found while refining the rest of the face. Interior
    // insertion leaves PSLG segments unchanged, so every split remains valid until the face
    // stops; one surface pass then reports them together instead of one round per pass.
    let mut pending_splits: Vec<super::ExactProtectedSegmentSplit> = Vec::new();
    let curve_splits = |pending: Vec<super::ExactProtectedSegmentSplit>, inserted: u32| {
        ExactFaceRefinementOutcome::RequiresCurveSplits {
            splits: pending,
            completed_interior_insertions: inserted,
        }
    };
    loop {
        let iteration_storage = GeometryAllocationScope::new(context.geometry_control);
        let context = ExactFaceRefinementContext {
            geometry_control: &iteration_storage,
            ..context
        };
        // A chart edge between two images of one node (opposite sides of a periodic cut or
        // seam) has zero physical length; its triangle cannot be evaluated. Split such edges
        // at their chart midpoint, which is a distinct surface point, before evaluation.
        let periodic = periodic_image_candidates(&current);
        if !periodic.is_empty() {
            if insertion_count >= policy.refinement.maximum_interior_insertions {
                return Err(ExactFaceRefinementError::new(
                    ExactFaceRefinementErrorKind::ResourceLimit,
                    &current.pslg.source_face_id,
                    "exact face refinement exceeded its interior insertion hard limit",
                ));
            }
            let mut periodic = periodic;
            periodic.truncate(
                (policy.refinement.maximum_interior_insertions - insertion_count) as usize,
            );
            let (refined, inserted) = match domain {
                ExactFaceRefinementDomain::ExactBoundary(boundary) => {
                    insert_exact_face_refinement_candidates_with_control(
                        boundary,
                        &current,
                        &periodic,
                        context.cancellation,
                        policy.delaunay,
                        Some(context.geometry_control),
                    )?
                }
                ExactFaceRefinementDomain::Chart(_) => insert_validated_face_refinement_candidates(
                    &current,
                    &periodic,
                    context.cancellation,
                    policy.delaunay,
                    Some(context.geometry_control),
                )?,
            };
            if inserted == 0 {
                return Err(ExactFaceRefinementError::new(
                    ExactFaceRefinementErrorKind::InvalidGeometry,
                    &current.pslg.source_face_id,
                    "periodic image edge split inserted no vertex",
                ));
            }
            current = refined;
            insertion_count += inserted;
            continue;
        }
        let geometry = evaluate_exact_face_geometry_in_parameterization(
            &current.trimmed,
            &current.pslg,
            domain.parameterization(),
            context.geometry_context(),
        )
        .map_err(map_geometry)?;
        super::control::checkpoint(Some(context.geometry_control), &geometry.source_face_id)?;
        // Interior refinement cannot repair a fixed curve point outside the admitted
        // surface envelope. Reject before spending the insertion budget on it.
        for vertex in &geometry.vertices {
            let displacement = std::array::from_fn::<_, 3, _>(|axis| {
                vertex.point_m[axis] - vertex.evaluation.point_m[axis]
            });
            let deviation = displacement[0]
                .hypot(displacement[1])
                .hypot(displacement[2]);
            if deviation > policy.quality.maximum_chordal_deviation_m.canonical_value() {
                return Err(ExactFaceRefinementError::new(
                    ExactFaceRefinementErrorKind::InvalidQuality,
                    &geometry.source_face_id,
                    "fixed boundary coordinate exceeds the requested surface chordal envelope",
                ));
            }
        }

        let collars = derive_exact_face_feature_collars(&current.pslg, &geometry, policy.quality)?;
        let mut candidates = select_exact_face_refinement_candidates(
            &geometry,
            &current.pslg,
            &collars,
            policy.quality,
            Some(context.geometry_control),
        )?;
        candidates.extend(super::boundary_support::candidates(
            &field,
            &geometry,
            &current.pslg,
            context.geometry_control,
        )?);
        if candidates.is_empty() {
            let provisional = crate::acceptance::ExactFaceRefinedMeshView {
                topology: &current,
                geometry: &geometry,
            };
            let report = sample_exact_face_mesh_in_parameterization(
                provisional,
                context,
                policy.quality,
                policy.acceptance,
                domain.parameterization(),
            )
            .map_err(map_acceptance)?;
            for (triangle_index, triangle) in report.triangles.iter().enumerate() {
                let (reason, witness_uv) = if triangle.maximum_chordal_deviation_m.canonical_value()
                    > policy.quality.maximum_chordal_deviation_m.canonical_value()
                {
                    (
                        ExactFaceRefinementReason::ChordalDeviation,
                        triangle.maximum_chordal_deviation_uv,
                    )
                } else if triangle.maximum_normal_deviation_rad
                    > policy.quality.maximum_normal_deviation_degrees.to_radians()
                {
                    (
                        ExactFaceRefinementReason::NormalDeviation,
                        triangle.maximum_normal_deviation_uv,
                    )
                } else {
                    continue;
                };
                let uv = separate_refinement_candidate(
                    witness_uv,
                    triangle_index,
                    &geometry,
                    &current.pslg,
                    policy.quality.minimum_metric_edge_length,
                    Some(context.geometry_control),
                )?
                .unwrap_or(witness_uv);
                candidates.push(ExactFaceRefinementCandidate {
                    source_face_id: current.pslg.source_face_id.clone(),
                    triangle_index: triangle_index as u32,
                    triangle: triangle.triangle,
                    reason,
                    uv,
                });
            }
            if candidates.is_empty() && !pending_splits.is_empty() {
                return Ok((
                    curve_splits(pending_splits, insertion_count),
                    chart_cut_split_count,
                ));
            }
            if candidates.is_empty() {
                return finish_refinement(
                    domain,
                    current,
                    geometry,
                    collars,
                    context,
                    (insertion_count, chart_cut_split_count),
                    &iteration_storage,
                );
            }
        }

        tracing::info!(event_code = "meshing.surface_refinement.inventory", face = ?current.pslg.source_face_id,
            vertices = current.pslg.vertices.len(), triangles = current.trimmed.triangles.len(),
            candidates = candidates.len(), insertion_count, chart_cut_split_count,
            "surface refinement iteration");
        let classifications = super::encroachment::classify_with_context(
            &candidates,
            &current.pslg,
            domain.parameterization(),
            context.geometry_context(),
        )?;
        for split in classifications.protected_splits {
            if !pending_splits
                .iter()
                .any(|pending| pending.pslg_segment_index == split.pslg_segment_index)
            {
                pending_splits.push(split);
            }
        }
        // A chart-cut split renumbers PSLG segments; report pending curve splits first. With
        // nothing left to insert, the face is blocked on its boundary.
        if !pending_splits.is_empty()
            && (!classifications.chart_cut_splits.is_empty()
                || classifications.insertions.is_empty()
                || insertion_count >= policy.refinement.maximum_interior_insertions)
        {
            return Ok((
                curve_splits(pending_splits, insertion_count),
                chart_cut_split_count,
            ));
        }
        if let Some(split) = classifications.chart_cut_splits.into_iter().next() {
            match domain {
                ExactFaceRefinementDomain::ExactBoundary(_) => {
                    return Ok((
                        ExactFaceRefinementOutcome::RequiresChartCutSplit {
                            split: Box::new(split),
                            completed_interior_insertions: insertion_count,
                        },
                        chart_cut_split_count,
                    ));
                }
                ExactFaceRefinementDomain::Chart(_) => {
                    if chart_cut_split_count >= maximum_chart_cut_splits {
                        return Err(ExactFaceRefinementError::new(
                            ExactFaceRefinementErrorKind::ResourceLimit,
                            &current.pslg.source_face_id,
                            "exact face refinement exceeded its chart-cut split hard limit",
                        ));
                    }
                    current = split_exact_face_chart_cut_with_control(
                        &current,
                        &split,
                        context.cancellation,
                        policy.delaunay,
                        Some(context.geometry_control),
                    )?;
                    chart_cut_split_count += 1;
                }
            }
        } else if !classifications.insertions.is_empty() {
            if insertion_count >= policy.refinement.maximum_interior_insertions {
                return Err(ExactFaceRefinementError::new(
                    ExactFaceRefinementErrorKind::ResourceLimit,
                    &current.pslg.source_face_id,
                    "exact face refinement exceeded its interior insertion hard limit",
                ));
            }
            let remaining = policy.refinement.maximum_interior_insertions - insertion_count;
            let metric_at = |candidate: &ExactFaceRefinementCandidate, uv: [f64; 2]| {
                field
                    .evaluate_parameterized(
                        &candidate.source_face_id,
                        uv,
                        domain.parameterization(),
                        context.evaluator,
                        context.geometry_control,
                    )
                    .map(|evaluation| evaluation.sizing_metric)
                    .map_err(|error| {
                        ExactFaceRefinementError::new(
                            ExactFaceRefinementErrorKind::Metric(error.kind),
                            &candidate.source_face_id,
                            error.reason,
                        )
                    })
            };
            let mut candidates = super::select::independent_candidate_batch_in_metric(
                classifications.insertions,
                &geometry,
                &current.pslg,
                policy.quality.minimum_metric_edge_length,
                Some(context.geometry_control),
                &metric_at,
            )?;
            candidates.truncate(remaining as usize);
            if candidates.is_empty() && !pending_splits.is_empty() {
                return Ok((
                    curve_splits(pending_splits, insertion_count),
                    chart_cut_split_count,
                ));
            }
            if candidates.is_empty() {
                // Only shape candidates may be suppressed by the spacing collar. Their insertion
                // would create an edge below the hard minimum, so the current constrained mesh is
                // the terminal result. Geometry-deviation and metric-size candidates fail above
                // instead of being silently accepted.
                return finish_refinement(
                    domain,
                    current,
                    geometry,
                    collars,
                    context,
                    (insertion_count, chart_cut_split_count),
                    &iteration_storage,
                );
            }
            let (refined, inserted) = match domain {
                ExactFaceRefinementDomain::ExactBoundary(boundary) => {
                    insert_exact_face_refinement_candidates_with_control(
                        boundary,
                        &current,
                        &candidates,
                        context.cancellation,
                        policy.delaunay,
                        Some(context.geometry_control),
                    )?
                }
                ExactFaceRefinementDomain::Chart(_) => insert_validated_face_refinement_candidates(
                    &current,
                    &candidates,
                    context.cancellation,
                    policy.delaunay,
                    Some(context.geometry_control),
                )?,
            };
            current = refined;
            insertion_count += inserted;
        } else if classifications.withdrawn > 0 {
            // Every remaining candidate is a shape point withdrawn against a shared curve. As
            // with collar-suppressed shape points, the current constrained mesh is the result;
            // independent acceptance still checks it.
            return finish_refinement(
                domain,
                current,
                geometry,
                collars,
                context,
                (insertion_count, chart_cut_split_count),
                &iteration_storage,
            );
        } else {
            return Err(ExactFaceRefinementError::new(
                ExactFaceRefinementErrorKind::InvalidGeometry,
                &current.pslg.source_face_id,
                "refinement candidates produced no actionable disposition",
            ));
        }
    }
}

fn finish_refinement(
    domain: ExactFaceRefinementDomain<'_>,
    topology: ExactFaceRefinedTopology,
    geometry: crate::ExactFaceGeometry,
    feature_collars: super::ExactFaceFeatureCollars,
    context: ExactFaceRefinementContext<'_>,
    counts: (u32, u32),
    storage: &GeometryAllocationScope<'_>,
) -> Result<(ExactFaceRefinementOutcome, u32), ExactFaceRefinementError> {
    let (interior_insertion_count, chart_cut_split_count) = counts;
    // Independent validation belongs at the artifact boundary. Re-evaluating every node and
    // deviation sample after every insertion turns refinement into a quadratic prefix audit
    // without protecting additional externally observable state.
    validate_exact_face_geometry_in_parameterization(
        &geometry,
        &topology.trimmed,
        &topology.pslg,
        domain.parameterization(),
        context.geometry_context(),
    )
    .map_err(map_geometry)?;
    // Returned geometry buffers survive this iteration. The partition owner keeps
    // shared field provenance alive; standalone evaluation retains its one face record.
    let mut retained = crate::face_geometry::storage::buffer_bytes(
        geometry.vertices.capacity(),
        geometry.triangles.capacity(),
    )
    .ok_or_else(|| {
        ExactFaceRefinementError::new(
            ExactFaceRefinementErrorKind::ResourceLimit,
            &geometry.source_face_id,
            "geometry buffer inventory overflow",
        )
    })?;
    if context.metric_field.is_none() {
        if let Some(vertex) = geometry.vertices.first() {
            let provenance = &vertex.evaluation.provenance;
            retained = retained
                .checked_add(
                    crate::ExactFaceMetricProvenance::storage_bytes(
                        &provenance.source_face_id,
                        &provenance.active_sources,
                    )
                    .ok_or_else(|| {
                        ExactFaceRefinementError::new(
                            ExactFaceRefinementErrorKind::ResourceLimit,
                            &geometry.source_face_id,
                            "provenance inventory overflow",
                        )
                    })?,
                )
                .ok_or_else(|| {
                    ExactFaceRefinementError::new(
                        ExactFaceRefinementErrorKind::ResourceLimit,
                        &geometry.source_face_id,
                        "geometry inventory overflow",
                    )
                })?;
        }
    }
    storage.retain_bytes(retained);
    Ok((
        ExactFaceRefinementOutcome::Converged(Box::new(ExactFaceRefinedMesh {
            topology,
            geometry,
            feature_collars,
            interior_insertion_count,
        })),
        chart_cut_split_count,
    ))
}

fn map_geometry(error: ExactFaceGeometryError) -> ExactFaceRefinementError {
    ExactFaceRefinementError::new(
        ExactFaceRefinementErrorKind::Geometry(error.kind),
        &error.source_face_id,
        error.reason,
    )
}

fn map_acceptance(error: ExactFaceAcceptanceError) -> ExactFaceRefinementError {
    let kind = match error.kind {
        ExactFaceAcceptanceErrorKind::ResourceLimit => ExactFaceRefinementErrorKind::ResourceLimit,
        ExactFaceAcceptanceErrorKind::UnsatisfiedQuality => {
            ExactFaceRefinementErrorKind::UnsatisfiedQuality
        }
        ExactFaceAcceptanceErrorKind::Metric(kind) => ExactFaceRefinementErrorKind::Metric(kind),
        ExactFaceAcceptanceErrorKind::Geometry(kind) => {
            ExactFaceRefinementErrorKind::Geometry(kind)
        }
        ExactFaceAcceptanceErrorKind::InvalidOptions => {
            ExactFaceRefinementErrorKind::InvalidOptions
        }
        ExactFaceAcceptanceErrorKind::InvalidInput => ExactFaceRefinementErrorKind::InvalidGeometry,
    };
    ExactFaceRefinementError::new(kind, &error.source_face_id, error.reason)
}

/// Midpoints of interior chart edges whose endpoints are images of one boundary node.
fn periodic_image_candidates(
    current: &ExactFaceRefinedTopology,
) -> Vec<ExactFaceRefinementCandidate> {
    let vertices = &current.pslg.vertices;
    let segments = current
        .pslg
        .segments
        .iter()
        .map(|segment| {
            let mut pair = segment.vertex_indices;
            pair.sort_unstable();
            pair
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut seen = std::collections::BTreeSet::new();
    let mut candidates = Vec::new();
    for (index, triangle) in current.trimmed.triangles.iter().enumerate() {
        let corners = triangle.vertex_indices;
        for (first, second) in [(0, 1), (1, 2), (2, 0)] {
            let mut pair = [corners[first], corners[second]];
            pair.sort_unstable();
            let [left, right] = pair.map(|vertex| &vertices[vertex as usize]);
            if pair[0] == pair[1] || left.node_id != right.node_id || segments.contains(&pair) {
                continue;
            }
            if !seen.insert(pair) {
                continue;
            }
            candidates.push(ExactFaceRefinementCandidate {
                source_face_id: current.pslg.source_face_id.clone(),
                triangle_index: index as u32,
                triangle: *triangle,
                reason: ExactFaceRefinementReason::MetricEdgeLength,
                uv: [0, 1].map(|axis| 0.5 * (left.uv[axis] + right.uv[axis])),
            });
        }
    }
    candidates
}

#[cfg(test)]
mod periodic_image_tests {
    use super::*;
    use crate::{
        ExactFaceConstrainedDelaunay, ExactFaceDelaunayTriangle, ExactFacePslg,
        ExactFacePslgSegment, ExactFacePslgSegmentSource, ExactFacePslgVertex,
        ExactFaceTrimmedDelaunay,
    };
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
    use runmat_meshing_core::StableDigest;

    #[test]
    fn interior_edges_between_node_images_are_split_at_their_chart_midpoint() {
        let face = PersistentEntityId::new(
            PersistentEntityKind::Face,
            "face".to_string(),
            vec!["root".to_string()],
        );
        let vertex = |node: u8, uv: [f64; 2]| ExactFacePslgVertex {
            node_id: StableDigest::from_bytes([node; 32]),
            seam_image: None,
            uv,
        };
        // Two images of node 1 on opposite cut sides, and a cut segment between images of
        // node 2 that must never be split as an interior edge.
        let vertices = vec![
            vertex(1, [0.0, 0.0]),
            vertex(1, [1.0, 0.0]),
            vertex(3, [0.5, 1.0]),
            vertex(2, [0.0, -1.0]),
            vertex(2, [1.0, -1.0]),
        ];
        let segment = |vertex_indices| ExactFacePslgSegment {
            source: ExactFacePslgSegmentSource::ChartCut {
                cut_id: StableDigest::from_bytes([9; 32]),
            },
            vertex_indices,
            edge_parameters: None,
        };
        let triangles = vec![
            ExactFaceDelaunayTriangle {
                vertex_indices: [0, 1, 2],
            },
            ExactFaceDelaunayTriangle {
                vertex_indices: [0, 3, 1],
            },
            ExactFaceDelaunayTriangle {
                vertex_indices: [1, 3, 4],
            },
        ];
        let topology = ExactFaceRefinedTopology {
            pslg: ExactFacePslg {
                source_face_id: face.clone(),
                vertices,
                segments: vec![segment([3, 4])],
                loops: Vec::new(),
            },
            constrained: ExactFaceConstrainedDelaunay {
                source_face_id: face.clone(),
                triangles: triangles.clone(),
                protected_segments: Vec::new(),
                recovery_edge_flip_count: 0,
                cavity_retriangulation_count: 0,
                delaunay_restoration_flip_count: 0,
            },
            trimmed: ExactFaceTrimmedDelaunay {
                source_face_id: face,
                triangles,
                boundary_segments: Vec::new(),
                removed_exterior_triangle_count: 0,
                removed_hole_triangle_count: 0,
            },
        };
        let candidates = periodic_image_candidates(&topology);
        assert_eq!(candidates.len(), 1, "a shared image edge is split once");
        assert_eq!(candidates[0].uv, [0.5, 0.0]);
        assert_eq!(candidates[0].triangle_index, 0);
    }
}
