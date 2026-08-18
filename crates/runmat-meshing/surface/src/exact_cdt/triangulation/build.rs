use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_core::{
    predicate::{incircle2d_symbolic, orient2d_symbolic},
    MeshingCancellationSignal, PlanarPredicatePoint, PredicateSign, StableDigest,
};

use crate::{validate_exact_face_pslg, ExactFaceBoundary, ExactFacePslg};

use super::{
    predicate_point, ExactFaceDelaunay, ExactFaceDelaunayError, ExactFaceDelaunayErrorKind,
    ExactFaceDelaunayOptions, ExactFaceDelaunayTriangle,
};

pub fn triangulate_exact_face_pslg(
    pslg: &ExactFacePslg,
    boundary: &ExactFaceBoundary,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceDelaunay, ExactFaceDelaunayError> {
    validate_exact_face_pslg(pslg, boundary).map_err(|error| {
        ExactFaceDelaunayError::new(
            ExactFaceDelaunayErrorKind::InvalidPslg,
            &pslg.source_face_id,
            error.to_string(),
        )
    })?;
    triangulate_validated_face_pslg(pslg, cancellation, options)
}

pub(crate) fn triangulate_validated_face_pslg(
    pslg: &ExactFacePslg,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceDelaunay, ExactFaceDelaunayError> {
    validate_options(pslg, options)?;
    let mut budget = PredicateBudget::new(pslg, cancellation, options);
    budget.checkpoint()?;

    let mut points = pslg
        .vertices
        .iter()
        .copied()
        .enumerate()
        .map(|(index, vertex)| predicate_point(vertex, index as u32))
        .collect::<Vec<_>>();
    let super_points = super_triangle(&points, &pslg.source_face_id)?;
    let source_count = points.len();
    points.extend(super_points);
    let mut triangles = vec![[
        source_count as u32,
        source_count as u32 + 1,
        source_count as u32 + 2,
    ]];

    for point_index in 0..source_count as u32 {
        budget.checkpoint()?;
        let mut bad = BTreeSet::new();
        for (triangle_index, triangle) in triangles.iter().enumerate() {
            budget.consume(1)?;
            if incircle2d_symbolic([
                point(&points, triangle[0]),
                point(&points, triangle[1]),
                point(&points, triangle[2]),
                point(&points, point_index),
            ])
            .map_err(|error| predicate_error(pslg, error))?
                == PredicateSign::Positive
            {
                bad.insert(triangle_index);
            }
        }
        if bad.is_empty() {
            return Err(ExactFaceDelaunayError::new(
                ExactFaceDelaunayErrorKind::InvalidTopology,
                &pslg.source_face_id,
                "point insertion found no containing Delaunay cavity",
            ));
        }
        let mut edge_counts = BTreeMap::<[u32; 2], usize>::new();
        for triangle_index in &bad {
            for edge in triangle_edges(triangles[*triangle_index]) {
                *edge_counts.entry(sorted_edge(edge)).or_default() += 1;
            }
        }
        triangles = triangles
            .into_iter()
            .enumerate()
            .filter_map(|(index, triangle)| (!bad.contains(&index)).then_some(triangle))
            .collect();
        for (edge, count) in edge_counts {
            if count != 1 {
                continue;
            }
            budget.consume(1)?;
            let mut triangle = [edge[0], edge[1], point_index];
            if orient2d_symbolic(triangle.map(|index| point(&points, index)))
                .map_err(|error| predicate_error(pslg, error))?
                == PredicateSign::Negative
            {
                triangle.swap(0, 1);
            }
            triangles.push(triangle);
            if triangles.len() > options.maximum_triangles {
                return Err(resource_error(pslg, "triangle hard limit exceeded"));
            }
        }
    }

    let mut triangles = triangles
        .into_iter()
        .filter(|triangle| triangle.iter().all(|index| *index < source_count as u32))
        .map(|triangle| ExactFaceDelaunayTriangle {
            vertex_indices: canonical_triangle(triangle),
        })
        .collect::<Vec<_>>();
    triangles.sort();
    let result = ExactFaceDelaunay {
        source_face_id: pslg.source_face_id.clone(),
        triangles,
    };
    super::validate::validate_face_delaunay_topology(&result, pslg, cancellation, options)?;
    Ok(result)
}

fn validate_options(
    pslg: &ExactFacePslg,
    options: ExactFaceDelaunayOptions,
) -> Result<(), ExactFaceDelaunayError> {
    if let Err(reason) = options.validate() {
        return Err(ExactFaceDelaunayError::new(
            ExactFaceDelaunayErrorKind::InvalidOptions,
            &pslg.source_face_id,
            reason,
        ));
    }
    if pslg.vertices.len() > u32::MAX as usize - 3 {
        return Err(resource_error(
            pslg,
            "vertex inventory exceeds index capacity",
        ));
    }
    Ok(())
}

fn super_triangle(
    points: &[PlanarPredicatePoint],
    source_face_id: &runmat_geometry_core::PersistentEntityId,
) -> Result<[PlanarPredicatePoint; 3], ExactFaceDelaunayError> {
    let mut minimum = points[0].coordinates;
    let mut maximum = minimum;
    for point in &points[1..] {
        for axis in 0..2 {
            minimum[axis] = minimum[axis].min(point.coordinates[axis]);
            maximum[axis] = maximum[axis].max(point.coordinates[axis]);
        }
    }
    let center = [
        minimum[0] * 0.5 + maximum[0] * 0.5,
        minimum[1] * 0.5 + maximum[1] * 0.5,
    ];
    let span = (maximum[0] - minimum[0]).max(maximum[1] - minimum[1]);
    let radius = span * 64.0;
    if !center.into_iter().all(f64::is_finite) || !radius.is_finite() || radius <= 0.0 {
        return Err(ExactFaceDelaunayError::new(
            ExactFaceDelaunayErrorKind::InvalidTopology,
            source_face_id,
            "face UV bounds cannot form a finite two-dimensional super triangle",
        ));
    }
    let coordinates = [
        [center[0] - radius, center[1] - radius],
        [center[0] + radius, center[1] - radius],
        [center[0], center[1] + radius],
    ];
    let mut next = 0u64;
    Ok(coordinates.map(|coordinates| {
        let mut bytes = [0x53; 32];
        bytes[..8].copy_from_slice(&next.to_be_bytes());
        next += 1;
        PlanarPredicatePoint {
            identity: StableDigest::from_bytes(bytes),
            coordinates,
        }
    }))
}

fn point(points: &[PlanarPredicatePoint], index: u32) -> PlanarPredicatePoint {
    points[index as usize]
}

fn triangle_edges(triangle: [u32; 3]) -> [[u32; 2]; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
}

fn sorted_edge(mut edge: [u32; 2]) -> [u32; 2] {
    edge.sort_unstable();
    edge
}

fn canonical_triangle(triangle: [u32; 3]) -> [u32; 3] {
    let minimum = triangle
        .iter()
        .enumerate()
        .min_by_key(|(_, index)| *index)
        .map(|(position, _)| position)
        .expect("triangle is non-empty");
    [
        triangle[minimum],
        triangle[(minimum + 1) % 3],
        triangle[(minimum + 2) % 3],
    ]
}

fn predicate_error(
    pslg: &ExactFacePslg,
    error: runmat_meshing_core::PlanarPredicateError,
) -> ExactFaceDelaunayError {
    ExactFaceDelaunayError::new(
        ExactFaceDelaunayErrorKind::InvalidTopology,
        &pslg.source_face_id,
        format!("invalid planar predicate input: {error:?}"),
    )
}

fn resource_error(pslg: &ExactFacePslg, reason: &str) -> ExactFaceDelaunayError {
    ExactFaceDelaunayError::new(
        ExactFaceDelaunayErrorKind::ResourceLimit,
        &pslg.source_face_id,
        reason,
    )
}

struct PredicateBudget<'a> {
    pslg: &'a ExactFacePslg,
    cancellation: &'a dyn MeshingCancellationSignal,
    remaining: u64,
    check_interval: u64,
    since_check: u64,
}

impl<'a> PredicateBudget<'a> {
    fn new(
        pslg: &'a ExactFacePslg,
        cancellation: &'a dyn MeshingCancellationSignal,
        options: ExactFaceDelaunayOptions,
    ) -> Self {
        Self {
            pslg,
            cancellation,
            remaining: options.maximum_predicate_evaluations,
            check_interval: options.cancellation_check_interval,
            since_check: 0,
        }
    }

    fn consume(&mut self, count: u64) -> Result<(), ExactFaceDelaunayError> {
        self.remaining = self
            .remaining
            .checked_sub(count)
            .ok_or_else(|| resource_error(self.pslg, "predicate evaluation hard limit exceeded"))?;
        self.since_check += count;
        if self.since_check >= self.check_interval {
            self.checkpoint()?;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.since_check = 0;
        if self.cancellation.is_cancelled() {
            Err(ExactFaceDelaunayError::new(
                ExactFaceDelaunayErrorKind::Cancelled,
                &self.pslg.source_face_id,
                "surface Delaunay construction cancelled",
            ))
        } else {
            Ok(())
        }
    }
}
