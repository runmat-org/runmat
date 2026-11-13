//! Mode-specific numerical spaces share only the quadratic trust-ball kernel.
//! Every returned proposal owns its actual rounded point and displacement.
use super::*;
use runmat_builtins::OptimSubproblemAlgorithm;
use trust_quadratic::{cg, factorization, Operator};
pub(super) enum Prepared {
    Cg(cg::Preconditioner),
    Direct(factorization::Factorization),
}
impl Prepared {
    pub(super) async fn new(
        operator: &mut impl Operator,
        n: usize,
        controls: &Controls,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        match controls.subproblem {
            OptimSubproblemAlgorithm::Cg => {
                cg::Preconditioner::new(operator, n, controls.bandwidth, cooperation)
                    .await
                    .map(|p| p.map(Self::Cg))
            }
            OptimSubproblemAlgorithm::Factorization => {
                factorization::Factorization::new(operator, n, cooperation)
                    .await
                    .map(|p| p.map(Self::Direct))
            }
        }
    }
    pub(super) async fn step(
        &self,
        operator: &mut impl Operator,
        gradient: &[f64],
        radius: f64,
        controls: &Controls,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<cg::Step>> {
        match self {
            Self::Cg(preconditioner) => {
                cg::step(
                    operator,
                    preconditioner,
                    gradient,
                    radius,
                    controls.maximum_cg_iterations,
                    controls.cg_tolerance,
                    cooperation,
                )
                .await
            }
            Self::Direct(factorization) => {
                factorization
                    .step(operator, gradient, radius, cooperation)
                    .await
            }
        }
    }
}
pub(super) enum Space<'a, 'matrix, 'model> {
    Bounds(bounds::Scaled<'a, 'model>),
    Equality(equality::Projected<'a, 'matrix, 'model>),
}
impl Operator for Space<'_, '_, '_> {
    fn policy(&self) -> trust_quadratic::Policy {
        model::policy()
    }
    async fn apply(
        &mut self,
        input: &[f64],
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        match self {
            Self::Bounds(space) => Operator::apply(space, input, output, cooperation).await,
            Self::Equality(space) => Operator::apply(space, input, output, cooperation).await,
        }
    }
    async fn project(
        &mut self,
        values: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<()>> {
        match self {
            Self::Bounds(space) => Operator::project(space, values, cooperation).await,
            Self::Equality(space) => Operator::project(space, values, cooperation).await,
        }
    }
    async fn copy_band(
        &self,
        n: usize,
        bandwidth: usize,
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<trust_quadratic::Materialization> {
        match self {
            Self::Bounds(space) => {
                Operator::copy_band(space, n, bandwidth, output, cooperation).await
            }
            Self::Equality(_) => Ok(trust_quadratic::Materialization::Unavailable),
        }
    }
    async fn copy_matrix(
        &mut self,
        n: usize,
        output: &mut Vec<num_complex::Complex64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<trust_quadratic::Materialization> {
        use trust_quadratic::Materialization;
        // This path is selected only by explicit direct mode. Hmultiply is
        // prohibited there; a real explicit/FD symmetric source is required.
        let bound = match self {
            Self::Bounds(space) => {
                let Some(scale) = space.base.magnitude(cooperation).await? else {
                    return Ok(Materialization::Undefined);
                };
                let mut d = 0.0_f64;
                let mut correction = 0.0_f64;
                for (&scale, &diagonal) in space.metric.scale.iter().zip(&space.metric.diagonal) {
                    d = d.max(scale);
                    correction = correction.max(diagonal.abs());
                    cooperation.completed(1).await?;
                }
                scale * n as f64 * d * d + correction
            }
            Self::Equality(space) => {
                let Some(scale) = space.base.magnitude(cooperation).await? else {
                    return Ok(Materialization::Undefined);
                };
                scale * n as f64
            }
        };
        if !bound.is_finite() {
            return Ok(Materialization::Undefined);
        }
        let mut unit = zeros(n, cooperation).await?;
        let mut product = zeros(n, cooperation).await?;
        for column in 0..n {
            unit[column] = 1.0;
            if Operator::apply(self, &unit, &mut product, cooperation)
                .await?
                .is_none()
            {
                return Ok(Materialization::Undefined);
            }
            unit[column] = 0.0;
            for chunk in product.chunks(1024) {
                output.extend(
                    chunk
                        .iter()
                        .map(|&value| num_complex::Complex64::new(value, 0.0)),
                );
                cooperation.completed(chunk.len() as u64).await?;
            }
        }
        let tolerance = 128.0 * f64::EPSILON * n.max(1) as f64;
        for column in 0..n {
            for row in column + 1..n {
                let left = output[column * n + row].re;
                let right = output[row * n + column].re;
                let error = (left - right).abs();
                if !error.is_finite()
                    || if bound == 0.0 {
                        error != 0.0
                    } else {
                        error / bound > tolerance
                    }
                {
                    return Ok(Materialization::Undefined);
                }
                let symmetric = 0.5 * left + 0.5 * right;
                output[column * n + row].re = symmetric;
                output[row * n + column].re = symmetric;
                cooperation.completed(1).await?;
            }
        }
        Ok(Materialization::Complete)
    }
}
pub(super) struct Proposal {
    pub x: Vec<f64>,
    pub displacement: Vec<f64>,
    pub predicted: f64,
    pub boundary: bool,
}
async fn point(
    mut p: Vec<f64>,
    x: &[f64],
    geometry: &Geometry,
    scale: Option<&[f64]>,
    radius: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<(Vec<f64>, Vec<f64>, bool)>> {
    let mut trial = workspace::capacity(x.len(), IDENTITY)?;
    let mut length = 0.0_f64;
    for (index, (p, &x)) in p.iter_mut().zip(x).enumerate() {
        let next = x + *p;
        if !next.is_finite() || next <= geometry.lower[index] || next >= geometry.upper[index] {
            return Ok(None);
        }
        *p = next - x;
        if !p.is_finite() {
            return Ok(None);
        }
        let value = scale.map_or(*p, |scale| *p / scale[index]);
        length = length.hypot(value);
        trial.push(next);
        cooperation.completed(1).await?;
    }
    if !length.is_finite() || length / radius > 1.0 + 128.0 * f64::EPSILON * x.len().max(1) as f64 {
        return Ok(None);
    }
    Ok(Some((trial, p, length / radius >= 0.9)))
}
async fn energy(
    base: &mut model::Products<'_>,
    p: &[f64],
    gradient: &[f64],
    scratch: &mut [f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<f64>> {
    if base
        .apply_numerical(p, scratch, cooperation)
        .await?
        .is_none()
    {
        return Ok(None);
    }
    let predicted = -dot(gradient, p) - 0.5 * dot(p, scratch);
    Ok((predicted > 0.0 && predicted.is_finite()).then_some(predicted))
}
impl Space<'_, '_, '_> {
    pub(super) async fn proposal(
        &mut self,
        step: cg::Step,
        x: &[f64],
        gradient: &[f64],
        geometry: &Geometry,
        tolerance: f64,
        radius: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Proposal>> {
        match self {
            Self::Bounds(space) => {
                let Some(direction) = space
                    .metric
                    .physical(&step.displacement, cooperation)
                    .await?
                else {
                    return Ok(None);
                };
                let mut best = None;
                for reflected in [false, true] {
                    let candidate = if reflected {
                        bounds::reflected(x, &direction, geometry, cooperation).await?
                    } else {
                        bounds::limited(x, &direction, geometry, cooperation).await?
                    };
                    let Some(candidate) = candidate else { continue };
                    let Some((trial, p, boundary)) = point(
                        candidate,
                        x,
                        geometry,
                        Some(&space.metric.scale),
                        radius,
                        cooperation,
                    )
                    .await?
                    else {
                        continue;
                    };
                    let Some(predicted) =
                        energy(space.base, &p, gradient, &mut space.scratch, cooperation).await?
                    else {
                        continue;
                    };
                    if best
                        .as_ref()
                        .is_none_or(|best: &Proposal| predicted > best.predicted)
                    {
                        best = Some(Proposal {
                            x: trial,
                            displacement: p,
                            predicted,
                            boundary,
                        });
                    }
                }
                // The scaled steepest direction is an independent Cauchy
                // safeguard, not a projection of an unconstrained optimizer.
                let length = norm(&space.metric.gradient);
                if length > 0.0 && length.is_finite() {
                    let mut cauchy = workspace::capacity(x.len(), IDENTITY)?;
                    for (&g, &d) in space.metric.gradient.iter().zip(&space.metric.scale) {
                        let p = -(g / length) * radius * d;
                        if !p.is_finite() {
                            return Ok(best);
                        }
                        cauchy.push(p);
                        cooperation.completed(1).await?;
                    }
                    if let Some(candidate) =
                        bounds::limited(x, &cauchy, geometry, cooperation).await?
                    {
                        if let Some((trial, p, boundary)) = point(
                            candidate,
                            x,
                            geometry,
                            Some(&space.metric.scale),
                            radius,
                            cooperation,
                        )
                        .await?
                        {
                            if let Some(predicted) =
                                energy(space.base, &p, gradient, &mut space.scratch, cooperation)
                                    .await?
                            {
                                if best.as_ref().is_none_or(|best| predicted > best.predicted) {
                                    best = Some(Proposal {
                                        x: trial,
                                        displacement: p,
                                        predicted,
                                        boundary,
                                    });
                                }
                            }
                        }
                    }
                }
                Ok(best)
            }
            Self::Equality(space) => {
                let mut p = step.displacement;
                if Operator::project(space, &mut p, cooperation)
                    .await?
                    .is_none()
                {
                    return Ok(None);
                }
                let Some((trial, p, boundary)) =
                    point(p, x, geometry, None, radius, cooperation).await?
                else {
                    return Ok(None);
                };
                if !space
                    .equality
                    .feasible(&trial, tolerance, &mut space.rhs, cooperation)
                    .await?
                {
                    return Ok(None);
                }
                let Some(predicted) =
                    energy(space.base, &p, gradient, &mut space.input, cooperation).await?
                else {
                    return Ok(None);
                };
                Ok(Some(Proposal {
                    x: trial,
                    displacement: p,
                    predicted,
                    boundary,
                }))
            }
        }
    }
}
