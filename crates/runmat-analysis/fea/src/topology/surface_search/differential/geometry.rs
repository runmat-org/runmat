use super::*;
pub(in crate::topology::surface_search) struct ProjectionGeometry {
    pub point: BoundarySurfacePoint,
    pub d: [f64; 3],
    pub t: [[f64; 3]; 2],
    pub weights: [f64; 6],
    pub first: [[f64; 2]; 6],
    pub second: [[[f64; 2]; 2]; 6],
    pub curvature: [[[f64; 3]; 2]; 2],
    pub scale: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub determinant: f64,
}
impl BoundaryProjectionDifferentialPolicy {
    pub fn validate(self) -> Result<(), BoundarySearchError> {
        if !self.stationarity_tolerance_m.is_finite()
            || self.stationarity_tolerance_m <= 0.0
            || !self.minimum_barycentric.is_finite()
            || self.minimum_barycentric < 0.0
            || self.minimum_barycentric >= 1.0 / 3.0
            || !self.minimum_metric_ratio.is_finite()
            || self.minimum_metric_ratio <= 0.0
            || self.minimum_metric_ratio >= 1.0
        {
            return Err(BoundarySearchError::InvalidInput);
        }
        Ok(())
    }
}

impl ProjectionGeometry {
    pub fn derive(
        surface: &BoundarySurface,
        query: [f64; 3],
        barycentric: [f64; 3],
        policy: BoundaryProjectionDifferentialPolicy,
    ) -> Result<Self, BoundarySearchError> {
        Self::derive_local(surface, query, barycentric, policy, true)
    }
    pub(in crate::topology::surface_search) fn derive_local(
        surface: &BoundarySurface,
        query: [f64; 3],
        barycentric: [f64; 3],
        policy: BoundaryProjectionDifferentialPolicy,
        require_stationarity: bool,
    ) -> Result<Self, BoundarySearchError> {
        policy.validate()?;
        if query.iter().any(|x| !x.is_finite()) {
            return Err(BoundarySearchError::InvalidInput);
        }
        let point = surface.evaluate(barycentric)?;
        if barycentric.iter().any(|b| *b <= policy.minimum_barycentric) {
            return Err(BoundarySearchError::NonSmoothProjection);
        }
        let d = subtract(query, point.position_m);
        let t = point.tangents_m;
        if d.iter().any(|x| !x.is_finite())
            || t.iter().any(|axis| {
                !norm(*axis).is_finite() || norm(*axis) <= 0.0 || !dot(d, *axis).is_finite()
            })
        {
            return Err(BoundarySearchError::Resolution);
        }
        if require_stationarity
            && t.iter()
                .any(|axis| dot(d, *axis).abs() / norm(*axis) > policy.stationarity_tolerance_m)
        {
            return Err(BoundarySearchError::NonStationaryProjection);
        }
        let (weights, first) = quadrature::boundary_shape(surface.order, barycentric);
        let second = quadrature::boundary_shape_second_derivatives(surface.order);
        let curvature: [[[f64; 3]; 2]; 2] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                std::array::from_fn(|k| {
                    (0..surface.count)
                        .map(|n| surface.nodes[n][k] * second[n][i][j])
                        .sum()
                })
            })
        });
        let metric: [[f64; 2]; 2] = std::array::from_fn(|i| {
            std::array::from_fn(|j| dot(t[i], t[j]) - dot(d, curvature[i][j]))
        });
        if metric.iter().flatten().any(|x| !x.is_finite()) {
            return Err(BoundarySearchError::Resolution);
        }
        let scale = metric
            .iter()
            .flatten()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        if !scale.is_finite() || scale == 0.0 {
            return Err(BoundarySearchError::DegenerateProjection);
        }
        let [a, b, c] = [
            metric[0][0] / scale,
            metric[0][1] / scale,
            metric[1][1] / scale,
        ];
        let largest = 0.5 * (a + c + (a - c).hypot(2.0 * b));
        let determinant = a * c - b * b;
        if largest <= 0.0
            || determinant <= policy.minimum_metric_ratio * largest * largest
            || !determinant.is_finite()
        {
            return Err(BoundarySearchError::DegenerateProjection);
        }

        Ok(Self {
            point,
            d,
            t,
            weights,
            first,
            second,
            curvature,
            scale,
            a,
            b,
            c,
            determinant,
        })
    }
    pub fn solve(&self, rhs: [f64; 2]) -> [f64; 2] {
        let rhs = rhs.map(|x| x / self.scale);
        [
            (self.c * rhs[0] - self.b * rhs[1]) / self.determinant,
            (self.a * rhs[1] - self.b * rhs[0]) / self.determinant,
        ]
    }
}
