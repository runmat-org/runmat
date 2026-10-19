//! Mixed implicit projection derivative; shares the local metric and shape authority.
use super::*;
use crate::algebra::NormalizedVectorDifferential;
#[derive(Clone, Copy, Debug)]
pub struct BoundaryProjectionDirection<'a> {
    pub query_m: [f64; 3],
    pub nodes_m: &'a [[f64; 3]],
}
#[derive(Clone, Copy, Debug)]
pub struct BoundaryProjectionMixedVariation {
    pub parameter_mixed: [f64; 2],
    pub surface_tangent_mixed_m: [[f64; 3]; 2],
    pub unit_normal_mixed: [f64; 3],
    pub signed_gap_mixed_m: f64,
}
impl BoundaryProjectionMixedVariation {
    pub(in crate::topology::surface_search) fn orient(mut self, side: f64) -> Self {
        self.unit_normal_mixed = self.unit_normal_mixed.map(|x| side * x);
        self.signed_gap_mixed_m *= side;
        self
    }
}
impl BoundarySurface {
    pub fn stationary_projection_mixed_variation(
        &self,
        query: [f64; 3],
        barycentric: [f64; 3],
        directions: [BoundaryProjectionDirection<'_>; 2],
        policy: BoundaryProjectionDifferentialPolicy,
    ) -> Result<BoundaryProjectionMixedVariation, BoundarySearchError> {
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled);
        }
        let g = geometry::ProjectionGeometry::derive(self, query, barycentric, policy)?;
        let [a, b] = directions;
        let va =
            self.stationary_projection_variation(query, barycentric, a.query_m, a.nodes_m, policy)?;
        let vb =
            self.stationary_projection_variation(query, barycentric, b.query_m, b.nodes_m, policy)?;
        let fixed = |d: BoundaryProjectionDirection<'_>| {
            let r = subtract(
                d.query_m,
                quadrature::weighted_position(d.nodes_m, &g.weights[..self.count]),
            );
            let t = [0, 1]
                .map(|i| quadrature::weighted_derivative(d.nodes_m, &g.first[..self.count], i));
            let h: [[[f64; 3]; 2]; 2] = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    std::array::from_fn(|k| {
                        d.nodes_m
                            .iter()
                            .enumerate()
                            .map(|(n, p)| p[k] * g.second[n][i][j])
                            .sum()
                    })
                })
            });
            (r, t, h)
        };
        let (ra, _, ha) = fixed(a);
        let (rb, tb, hb) = fixed(b);
        let da = subtract(a.query_m, va.point_position_increment_m);
        let rb_a = quadrature::weighted_position(b.nodes_m, va.shape_increment()).map(|x| -x);
        let tb_a: [[f64; 3]; 2] = std::array::from_fn(|i| {
            std::array::from_fn(|k| {
                (0..2)
                    .map(|j| hb[i][j][k] * va.parameter_increment[j])
                    .sum()
            })
        });
        let metric_a: [[f64; 2]; 2] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                dot(va.surface_tangent_increment_m[i], g.t[j])
                    + dot(g.t[i], va.surface_tangent_increment_m[j])
                    - dot(da, g.curvature[i][j])
                    - dot(g.d, ha[i][j])
            })
        });
        let rhs = [0, 1].map(|i| {
            dot(va.surface_tangent_increment_m[i], rb)
                + dot(g.t[i], rb_a)
                + dot(da, tb[i])
                + dot(g.d, tb_a[i])
                - (0..2)
                    .map(|j| metric_a[i][j] * vb.parameter_increment[j])
                    .sum::<f64>()
        });
        let parameter_mixed = g.solve(rhs);
        let surface_tangent_mixed_m = std::array::from_fn(|i| {
            std::array::from_fn(|k| {
                (0..2)
                    .map(|j| {
                        ha[i][j][k] * vb.parameter_increment[j]
                            + hb[i][j][k] * va.parameter_increment[j]
                            + g.curvature[i][j][k] * parameter_mixed[j]
                    })
                    .sum()
            })
        });
        let raw = cross(g.t[0], g.t[1]);
        let raw_first = [
            va.surface_tangent_increment_m,
            vb.surface_tangent_increment_m,
        ]
        .map(|t| {
            let x = cross(t[0], g.t[1]);
            let y = cross(g.t[0], t[1]);
            std::array::from_fn(|k| x[k] + y[k])
        });
        let terms = [
            cross(surface_tangent_mixed_m[0], g.t[1]),
            cross(
                va.surface_tangent_increment_m[0],
                vb.surface_tangent_increment_m[1],
            ),
            cross(
                vb.surface_tangent_increment_m[0],
                va.surface_tangent_increment_m[1],
            ),
            cross(g.t[0], surface_tangent_mixed_m[1]),
        ];
        let raw_mixed = std::array::from_fn(|k| terms.iter().map(|v| v[k]).sum());
        let normalized =
            NormalizedVectorDifferential::derive(raw, raw_first[0], raw_first[1], raw_mixed)
                .map_err(|_| BoundarySearchError::Resolution)?;
        let shape_b_a = quadrature::weighted_position(a.nodes_m, vb.shape_increment());
        let result = BoundaryProjectionMixedVariation {
            parameter_mixed,
            surface_tangent_mixed_m,
            unit_normal_mixed: normalized.mixed,
            signed_gap_mixed_m: dot(vb.unit_normal_increment, ra)
                - dot(g.point.unit_normal, shape_b_a),
        };
        if crate::is_fea_cancelled() {
            return Err(BoundarySearchError::Cancelled);
        }
        if result
            .parameter_mixed
            .iter()
            .chain(result.surface_tangent_mixed_m.iter().flatten())
            .chain(result.unit_normal_mixed.iter())
            .any(|x| !x.is_finite())
            || !result.signed_gap_mixed_m.is_finite()
        {
            return Err(BoundarySearchError::Resolution);
        }
        Ok(result)
    }
}
