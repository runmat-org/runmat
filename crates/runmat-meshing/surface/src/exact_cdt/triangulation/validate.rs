use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    predicate::{incircle2d_symbolic, orient2d},
    MeshingCancellationSignal, PredicateSign,
};

use crate::{validate_exact_face_pslg, ExactFaceBoundary, ExactFacePslg};

use super::{
    predicate_point, ExactFaceDelaunay, ExactFaceDelaunayError, ExactFaceDelaunayErrorKind,
    ExactFaceDelaunayOptions,
};

pub fn validate_exact_face_delaunay(
    triangulation: &ExactFaceDelaunay,
    pslg: &ExactFacePslg,
    boundary: &ExactFaceBoundary,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<(), ExactFaceDelaunayError> {
    if options.maximum_triangles == 0
        || options.maximum_predicate_evaluations == 0
        || options.maximum_edge_flips == 0
        || options.maximum_cavity_retriangulations == 0
        || options.cancellation_check_interval == 0
    {
        return Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidOptions,
            "Delaunay limits and cancellation interval must be non-zero",
        ));
    }
    check_cancelled(cancellation, pslg, "surface Delaunay validation cancelled")?;
    validate_exact_face_pslg(pslg, boundary).map_err(|error| {
        invalid(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidPslg,
            error.to_string(),
        )
    })?;
    if triangulation.source_face_id != pslg.source_face_id || triangulation.triangles.is_empty() {
        return Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidTopology,
            "Delaunay face identity or triangle inventory is invalid",
        ));
    }
    if triangulation.triangles.len() > options.maximum_triangles {
        return Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::ResourceLimit,
            "Delaunay triangle inventory exceeds its hard limit",
        ));
    }
    let mut predicate_count = 0u64;
    let mut referenced = BTreeSet::new();
    let mut prior = None;
    let mut edges = BTreeMap::<[u32; 2], Vec<(usize, u32)>>::new();
    for (triangle_index, triangle) in triangulation.triangles.iter().enumerate() {
        check_cancelled(cancellation, pslg, "surface Delaunay validation cancelled")?;
        if prior.is_some_and(|prior| prior >= *triangle) {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                "Delaunay triangles are duplicate or not canonical",
            ));
        }
        prior = Some(*triangle);
        let indices = triangle.vertex_indices;
        if indices[0] > indices[1].min(indices[2])
            || indices
                .iter()
                .any(|index| *index as usize >= pslg.vertices.len())
            || indices[0] == indices[1]
            || indices[1] == indices[2]
            || indices[2] == indices[0]
        {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                "Delaunay triangle indices are invalid",
            ));
        }
        predicate_count = consume_predicate(pslg, predicate_count, options)?;
        if orient2d(indices.map(|index| pslg.vertices[index as usize].uv))
            .map_err(|error| invalid_predicate(pslg, error))?
            != PredicateSign::Positive
        {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                "Delaunay triangle is not strictly counterclockwise",
            ));
        }
        referenced.extend(indices);
        for edge_position in 0..3 {
            let mut edge = [indices[edge_position], indices[(edge_position + 1) % 3]];
            let opposite = indices[(edge_position + 2) % 3];
            edge.sort_unstable();
            edges
                .entry(edge)
                .or_default()
                .push((triangle_index, opposite));
        }
    }
    if referenced.len() != pslg.vertices.len() {
        return Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::InvalidTopology,
            "Delaunay topology does not reference every PSLG vertex",
        ));
    }
    for (edge, uses) in edges {
        check_cancelled(cancellation, pslg, "surface Delaunay validation cancelled")?;
        if uses.len() > 2 {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                "Delaunay edge is nonmanifold",
            ));
        }
        if uses.len() != 2 {
            continue;
        }
        predicate_count = consume_predicate(pslg, predicate_count, options)?;
        let first = triangulation.triangles[uses[0].0].vertex_indices;
        let opposite = uses[1].1;
        let query = [first[0], first[1], first[2], opposite]
            .map(|index| predicate_point(pslg.vertices[index as usize], index));
        if incircle2d_symbolic(query).map_err(|error| invalid_predicate(pslg, error))?
            == PredicateSign::Positive
        {
            return Err(invalid(
                pslg,
                ExactFaceDelaunayErrorKind::InvalidTopology,
                format!("interior edge {edge:?} is not symbolically Delaunay"),
            ));
        }
    }
    check_cancelled(cancellation, pslg, "surface Delaunay validation cancelled")
}

fn consume_predicate(
    pslg: &ExactFacePslg,
    count: u64,
    options: ExactFaceDelaunayOptions,
) -> Result<u64, ExactFaceDelaunayError> {
    let count = count.saturating_add(1);
    if count > options.maximum_predicate_evaluations {
        Err(invalid(
            pslg,
            ExactFaceDelaunayErrorKind::ResourceLimit,
            "validation predicate evaluation hard limit exceeded",
        ))
    } else {
        Ok(count)
    }
}

fn check_cancelled(
    cancellation: &dyn MeshingCancellationSignal,
    pslg: &ExactFacePslg,
    reason: &str,
) -> Result<(), ExactFaceDelaunayError> {
    if cancellation.is_cancelled() {
        Err(invalid(pslg, ExactFaceDelaunayErrorKind::Cancelled, reason))
    } else {
        Ok(())
    }
}

fn invalid_predicate(
    pslg: &ExactFacePslg,
    error: runmat_meshing_core::PlanarPredicateError,
) -> ExactFaceDelaunayError {
    invalid(
        pslg,
        ExactFaceDelaunayErrorKind::InvalidTopology,
        format!("invalid planar predicate input: {error:?}"),
    )
}

fn invalid(
    pslg: &ExactFacePslg,
    kind: ExactFaceDelaunayErrorKind,
    reason: impl Into<String>,
) -> ExactFaceDelaunayError {
    ExactFaceDelaunayError::new(kind, &pslg.source_face_id, reason)
}
