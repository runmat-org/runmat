//! Nonuniform physical axes/random frame and original sparse boundary rays.
use super::super::{
    buffer, cone::Cone, evaluation::Point, invalid, linear::Linear, scaling::Scaling,
};
use crate::builtins::math::linalg::ops::sparse_factor::orthogonal_frame::OrthogonalFrame;
use crate::builtins::math::optim::common::{feasible_cone::Basis, ScalarCooperation};
use crate::BuiltinResult;

#[derive(Clone, Copy)]
pub(in crate::builtins::math::optim::patternsearch) enum Orientation<'a> {
    Gps,
    Random(&'a OrthogonalFrame),
}
pub(in crate::builtins::math::optim::patternsearch) struct Frame<'a> {
    orientation: Orientation<'a>,
    positive: bool,
    dimension: usize,
    base_count: usize,
    baseline: Option<Cone>,
    hints: Option<Cone>,
    scaling: &'a Scaling,
}
impl<'a> Frame<'a> {
    pub async fn new(
        base: &Point,
        geometry: &Linear,
        mesh: f64,
        scaling: &'a Scaling,
        orientation: Orientation<'a>,
        positive: bool,
        binding: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let n = base.x.len();
        if n == 0 || scaling.dimension() != n {
            return Err(invalid("NUPS frame dimension mismatch"));
        }
        if let Orientation::Random(frame) = orientation {
            if frame.dimension() != n {
                return Err(invalid("NUPS random frame dimension changed"));
            }
            if geometry.equalities > 0 {
                return Err(invalid(
                    "random NUPS frame does not support linear equalities",
                ));
            }
        }
        let baseline = if geometry.equalities > 0 {
            let Some((cone, _)) = geometry
                .binding_cone(&base.x, mesh, 0.0, scaling, cooperation)
                .await?
            else {
                return Ok(None);
            };
            Some(cone)
        } else {
            None
        };
        let hints =
            if geometry.matrix.m > geometry.equalities && (baseline.is_none() || binding > 0.0) {
                let Some((cone, active)) = geometry
                    .binding_cone(&base.x, mesh, binding, scaling, cooperation)
                    .await?
                else {
                    return Ok(None);
                };
                active.then_some(cone)
            } else {
                None
            };
        let base_count = if baseline.is_some() {
            0
        } else {
            n.checked_add(1)
                .ok_or_else(|| invalid("NUPS base count overflow"))?
        };
        Ok(Some(Self {
            orientation,
            positive,
            dimension: n,
            base_count,
            baseline,
            hints,
            scaling,
        }))
    }
    pub fn spans_pair(&self) -> bool {
        self.base_count == 0
    }
    fn cone_count(cone: &Option<Cone>) -> BuiltinResult<usize> {
        cone.as_ref()
            .map(|cone| cone.len(Basis::Maximal))
            .transpose()
            .map(|count| count.unwrap_or(0))
    }
    pub fn count(&self) -> BuiltinResult<usize> {
        let baseline = Self::cone_count(&self.baseline)?;
        let hints = Self::cone_count(&self.hints)?;
        self.base_count
            .checked_add(baseline)
            .and_then(|count| count.checked_add(hints))
            .ok_or_else(|| invalid("NUPS boundary count overflow"))
    }
    pub async fn direction(
        &self,
        index: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        if index >= self.count()? {
            return Err(invalid("NUPS direction index out of range"));
        }
        let mut ray = if index < self.base_count {
            let mut ray = buffer(self.dimension)?;
            for _ in 0..self.dimension {
                ray.push(if index == self.dimension { 1.0 } else { 0.0 });
                cooperation.completed(1).await?;
            }
            if index < self.dimension {
                ray[index] = 1.0;
            }
            if let Orientation::Random(frame) = self.orientation {
                let (_, budget) = cooperation.runtime_budget();
                frame.transform_into(&mut ray, budget).await?;
            }
            if !self.positive {
                for value in &mut ray {
                    *value = -*value;
                    cooperation.completed(1).await?;
                }
            }
            ray
        } else {
            let index = index - self.base_count;
            let baseline_count = Self::cone_count(&self.baseline)?;
            let (cone, index) = if index < baseline_count {
                (self.baseline.as_ref(), index)
            } else {
                (self.hints.as_ref(), index - baseline_count)
            };
            let Some(cone) = cone else {
                return Err(invalid("NUPS boundary generator absent"));
            };
            let Some(ray) = cone
                .oriented_direction(index, Basis::Maximal, false, cooperation)
                .await?
            else {
                return Ok(None);
            };
            ray
        };
        if !self.scaling.apply(&mut ray, cooperation).await? {
            return Ok(None);
        }
        Ok(Some(ray))
    }
}
