use super::*;

struct Boundary {
    vertices: [u32; 2],
    outside: usize,
    outside_edge: usize,
}
impl Mesh {
    pub(super) fn replace(
        &mut self,
        points: &[PlanarPredicatePoint],
        p: u32,
        bad: &[usize],
        budget: &mut PredicateBudget<'_>,
        maximum: usize,
    ) -> Result<(), ExactFaceDelaunayError> {
        let mut boundary = Vec::new();
        for &index in bad {
            let t = &self.triangles[index];
            for (edge, vertices) in triangle_edges(t.vertices).into_iter().enumerate() {
                budget.consume(1)?;
                let outside = t.neighbors[edge];
                if outside != NONE
                    && self.triangles[outside].stamp == p + 1
                    && self.triangles[outside].conflict
                {
                    continue;
                }
                let outside_edge = if outside == NONE {
                    NONE
                } else {
                    self.triangles[outside]
                        .neighbors
                        .iter()
                        .position(|n| *n == index)
                        .ok_or_else(|| Self::invalid(budget, "nonreciprocal cavity adjacency"))?
                };
                boundary.push(Boundary {
                    vertices: sorted_edge(vertices),
                    outside,
                    outside_edge,
                });
            }
        }
        boundary.sort_unstable_by_key(|b| b.vertices);
        let active = self.active - bad.len();
        if boundary.len() > maximum.saturating_sub(active) {
            return Err(resource_error(
                budget.pslg,
                ExactFaceDelaunayErrorKind::ElementLimit,
                "triangle hard limit exceeded",
            ));
        }
        for &index in bad {
            self.triangles[index].active = false;
            self.free.push(index);
        }
        let mut star = Vec::with_capacity(boundary.len() * 2);
        let mut first = NONE;
        for b in boundary {
            budget.consume(1)?;
            let mut vertices = [b.vertices[0], b.vertices[1], p];
            if orient2d_symbolic(vertices.map(|i| points[i as usize]))
                .map_err(|e| predicate_error(budget.pslg, e))?
                == PredicateSign::Negative
            {
                vertices.swap(0, 1);
            }
            let index = if let Some(index) = self.free.pop() {
                self.triangles[index] = Triangle::new(vertices);
                index
            } else {
                self.triangles.push(Triangle::new(vertices));
                self.triangles.len() - 1
            };
            if first == NONE {
                first = index;
            }
            self.triangles[index].neighbors[0] = b.outside;
            if b.outside != NONE {
                self.triangles[b.outside].neighbors[b.outside_edge] = index;
            }
            star.push((vertices[1], index, 1));
            star.push((vertices[0], index, 2));
        }
        star.sort_unstable();
        for pair in star.chunks(2) {
            budget.consume(1)?;
            if pair.len() != 2 || pair[0].0 != pair[1].0 {
                return Err(Self::invalid(
                    budget,
                    "cavity fan has an unpaired radial edge",
                ));
            }
            let (_, a, ae) = pair[0];
            let (_, b, be) = pair[1];
            self.triangles[a].neighbors[ae] = b;
            self.triangles[b].neighbors[be] = a;
        }
        if first == NONE {
            return Err(Self::invalid(budget, "empty cavity boundary"));
        }
        self.active = active + star.len() / 2;
        self.hint = first;
        Ok(())
    }
}
