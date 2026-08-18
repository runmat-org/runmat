use std::collections::BTreeSet;

use runmat_meshing_core::{
    predicate::{incircle2d_symbolic, orient2d},
    MeshingCancellationSignal, PredicateSign,
};

use crate::{
    validate_exact_face_pslg, ExactFaceBoundary, ExactFaceDelaunayError,
    ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions, ExactFacePslg,
};

use super::{planarity::validate_planar_edges, ExactFaceConstrainedDelaunay};
use crate::exact_cdt::topology::{edge_uses, sorted_edge};

pub fn validate_exact_face_constrained_delaunay(
    constrained: &ExactFaceConstrainedDelaunay,
    pslg: &ExactFacePslg,
    boundary: &ExactFaceBoundary,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<(), ExactFaceDelaunayError> {
    validate_options(pslg, options)?;
    validate_exact_face_pslg(pslg, boundary).map_err(|error| {
        invalid(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidPslg,
            error.to_string(),
        )
    })?;
    checkpoint(cancellation, pslg)?;
    if constrained.source_face_id != pslg.source_face_id
        || constrained.triangles.is_empty()
        || constrained.triangles.len() > options.maximum_triangles
        || constrained.protected_segments.len() != pslg.segments.len()
    {
        return Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidTopology,
            "constrained Delaunay inventory is inconsistent",
        ));
    }
    if constrained
        .recovery_edge_flip_count
        .saturating_add(constrained.delaunay_restoration_flip_count)
        > options.maximum_edge_flips
    {
        return Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::ResourceLimit,
            "reported constrained edge flips exceed the hard limit",
        ));
    }
    if constrained.cavity_retriangulation_count > options.maximum_cavity_retriangulations {
        return Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::ResourceLimit,
            "reported cavity retriangulations exceed the hard limit",
        ));
    }

    let mut referenced = BTreeSet::new();
    let mut prior = None;
    let mut predicate_count = 0u64;
    for triangle in &constrained.triangles {
        checkpoint(cancellation, pslg)?;
        if prior.is_some_and(|prior| prior >= *triangle) {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                "constrained triangles are duplicate or not canonical",
            ));
        }
        prior = Some(*triangle);
        let indices = triangle.vertex_indices;
        if indices[0] > indices[1].min(indices[2])
            || indices
                .iter()
                .any(|index| *index as usize >= pslg.vertices.len())
            || BTreeSet::from(indices).len() != 3
        {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                "constrained triangle indices are invalid",
            ));
        }
        consume(&mut predicate_count, options, pslg, 1)?;
        if orient2d(indices.map(|index| pslg.vertices[index as usize].uv))
            .map_err(|error| predicate_error(pslg, error))?
            != PredicateSign::Positive
        {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                "constrained triangle is not strictly counterclockwise",
            ));
        }
        referenced.extend(indices);
    }
    if referenced.len() != pslg.vertices.len() {
        return Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidTopology,
            "constrained topology does not reference every PSLG vertex",
        ));
    }

    let edges = edge_uses(&constrained.triangles);
    if edges.values().any(|uses| uses.len() > 2) {
        return Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidTopology,
            "constrained topology contains a nonmanifold edge",
        ));
    }
    predicate_count = predicate_count.saturating_add(validate_planar_edges(
        &edges,
        pslg,
        cancellation,
        ExactFaceDelaunayOptions {
            maximum_predicate_evaluations: options
                .maximum_predicate_evaluations
                .saturating_sub(predicate_count),
            ..options
        },
    )?);
    let mut protected = BTreeSet::new();
    for (index, (actual, expected)) in constrained
        .protected_segments
        .iter()
        .zip(&pslg.segments)
        .enumerate()
    {
        if actual.pslg_segment_index != index as u32
            || actual.source_coedge_id != expected.source_coedge_id
            || actual.source_edge_id != expected.source_edge_id
            || actual.vertex_indices != expected.vertex_indices
        {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                "protected segment provenance differs from the PSLG",
            ));
        }
        let edge = sorted_edge(actual.vertex_indices);
        if !protected.insert(edge) || !edges.contains_key(&edge) {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::UnsatisfiedConstraint,
                "protected PSLG segment is duplicate or absent",
            ));
        }
    }

    for (edge, uses) in &edges {
        checkpoint(cancellation, pslg)?;
        if protected.contains(edge) || uses.len() != 2 {
            continue;
        }
        consume(&mut predicate_count, options, pslg, 1)?;
        let triangle = constrained.triangles[uses[0].triangle_index].vertex_indices;
        let query = [
            triangle[0],
            triangle[1],
            triangle[2],
            uses[1].opposite_vertex,
        ]
        .map(|index| {
            super::super::triangulation::predicate_point(pslg.vertices[index as usize], index)
        });
        if incircle2d_symbolic(query).map_err(|error| predicate_error(pslg, error))?
            == PredicateSign::Positive
        {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                format!("unprotected edge {edge:?} is not symbolically Delaunay"),
            ));
        }
    }
    Ok(())
}

fn validate_options(
    pslg: &ExactFacePslg,
    options: ExactFaceDelaunayOptions,
) -> Result<(), ExactFaceDelaunayError> {
    if options.maximum_triangles == 0
        || options.maximum_predicate_evaluations == 0
        || options.maximum_edge_flips == 0
        || options.maximum_cavity_retriangulations == 0
        || options.cancellation_check_interval == 0
    {
        Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidOptions,
            "constrained Delaunay limits must be non-zero",
        ))
    } else {
        Ok(())
    }
}

fn consume(
    count: &mut u64,
    options: ExactFaceDelaunayOptions,
    pslg: &ExactFacePslg,
    amount: u64,
) -> Result<(), ExactFaceDelaunayError> {
    *count = count.saturating_add(amount);
    if *count > options.maximum_predicate_evaluations {
        Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::ResourceLimit,
            "constrained validation predicate hard limit exceeded",
        ))
    } else {
        Ok(())
    }
}

fn checkpoint(
    cancellation: &dyn MeshingCancellationSignal,
    pslg: &ExactFacePslg,
) -> Result<(), ExactFaceDelaunayError> {
    if cancellation.is_cancelled() {
        Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::Cancelled,
            "constrained Delaunay validation cancelled",
        ))
    } else {
        Ok(())
    }
}

fn predicate_error(
    pslg: &ExactFacePslg,
    error: runmat_meshing_core::PlanarPredicateError,
) -> ExactFaceDelaunayError {
    invalid(
        pslg,
        ExactFaceDelaunayErrorKind::InvalidTopology,
        format!("invalid constrained predicate input: {error:?}"),
    )
}

fn invalid(
    pslg: &ExactFacePslg,
    kind: ExactFaceDelaunayErrorKind,
    reason: impl Into<String>,
) -> ExactFaceDelaunayError {
    ExactFaceDelaunayError::new(kind, &pslg.source_face_id, reason)
}
