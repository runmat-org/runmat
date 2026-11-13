//! Stable triangle slots and connected Bowyer–Watson cavities. No global scan per point.
use super::*;
use runmat_meshing_core::predicate::orient2d;

mod insertion;
#[cfg(test)]
mod tests;

const NONE: usize = usize::MAX;
struct Triangle {
    vertices: [u32; 3],
    neighbors: [usize; 3],
    stamp: u32,
    conflict: bool,
    active: bool,
}
impl Triangle {
    fn new(vertices: [u32; 3]) -> Self {
        Self {
            vertices,
            neighbors: [NONE; 3],
            stamp: 0,
            conflict: false,
            active: true,
        }
    }
}
struct Mesh {
    triangles: Vec<Triangle>,
    free: Vec<usize>,
    hint: usize,
    active: usize,
}
impl Mesh {
    fn invalid(budget: &PredicateBudget<'_>, reason: &str) -> ExactFaceDelaunayError {
        resource_error(
            budget.pslg,
            ExactFaceDelaunayErrorKind::InvalidTopology,
            reason,
        )
    }
    fn locate(
        &self,
        points: &[PlanarPredicatePoint],
        p: u32,
        budget: &mut PredicateBudget<'_>,
    ) -> Result<usize, ExactFaceDelaunayError> {
        let mut current = self.hint;
        for _ in 0..=self.active {
            let triangle = &self.triangles[current];
            let mut next = NONE;
            for (edge, [a, b]) in triangle_edges(triangle.vertices).into_iter().enumerate() {
                budget.consume(1)?;
                if orient2d([
                    points[a as usize].coordinates,
                    points[b as usize].coordinates,
                    points[p as usize].coordinates,
                ])
                .map_err(|e| predicate_error(budget.pslg, e))?
                    == PredicateSign::Negative
                {
                    next = triangle.neighbors[edge];
                    if next == NONE {
                        return Err(Self::invalid(
                            budget,
                            "point walk escaped the super triangle",
                        ));
                    }
                    break;
                }
            }
            if next == NONE {
                return Ok(current);
            }
            current = next;
        }
        Err(Self::invalid(
            budget,
            "Delaunay point-location walk did not terminate",
        ))
    }
    fn conflicts(
        &mut self,
        points: &[PlanarPredicatePoint],
        p: u32,
        budget: &mut PredicateBudget<'_>,
    ) -> Result<Vec<usize>, ExactFaceDelaunayError> {
        let start = self.locate(points, p, budget)?;
        let mut pending = vec![start];
        let mut bad = Vec::new();
        while let Some(index) = pending.pop() {
            let t = &mut self.triangles[index];
            if t.stamp == p + 1 {
                continue;
            }
            t.stamp = p + 1;
            budget.consume(1)?;
            t.conflict = matches!(
                incircle2d([
                    points[t.vertices[0] as usize].coordinates,
                    points[t.vertices[1] as usize].coordinates,
                    points[t.vertices[2] as usize].coordinates,
                    points[p as usize].coordinates
                ])
                .map_err(|e| predicate_error(budget.pslg, e))?,
                PredicateSign::Positive | PredicateSign::Zero
            );
            if t.conflict {
                bad.push(index);
                pending.extend(t.neighbors.into_iter().filter(|n| *n != NONE));
            }
        }
        if bad.is_empty() {
            return Err(Self::invalid(
                budget,
                "point insertion found no containing Delaunay cavity",
            ));
        }
        Ok(bad)
    }
}

pub(super) fn triangulate(
    points: &[PlanarPredicatePoint],
    source_count: usize,
    budget: &mut PredicateBudget<'_>,
    options: ExactFaceDelaunayOptions,
) -> Result<Vec<[u32; 3]>, ExactFaceDelaunayError> {
    let n = source_count as u32;
    let mut mesh = Mesh {
        triangles: vec![Triangle::new([n, n + 1, n + 2])],
        free: Vec::new(),
        hint: 0,
        active: 1,
    };
    for p in 0..n {
        budget.checkpoint()?;
        let bad = mesh.conflicts(points, p, budget)?;
        mesh.replace(points, p, &bad, budget, options.maximum_triangles)?;
    }
    Ok(mesh
        .triangles
        .into_iter()
        .filter(|t| t.active)
        .map(|t| t.vertices)
        .collect())
}
