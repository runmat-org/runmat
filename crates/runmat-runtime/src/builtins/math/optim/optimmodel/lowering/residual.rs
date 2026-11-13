//! One original-DAG probe/accepted-point owner for residual solver consumers.
use super::*;
use crate::builtins::math::optim::least_squares::{LeastSquaresEvaluator, ResidualFuture};
use runmat_gc::GcHandle;
mod candidate;
pub(super) use candidate::Candidate;

pub(super) enum Projection<'a> {
    Squares {
        objective: GcHandle,
        squares: &'a objective::LeastSquares,
    },
    Equations(&'a equation::Evidence),
}
pub(super) struct Original<'a> {
    pub source: &'a evaluation::Plan,
    pub layout: &'a layout::Layout,
    pub projection: Projection<'a>,
    pub roots: &'a [GcHandle],
    pub rows: usize,
    pub pending: Option<numerical::Point>,
    pub accepted: Option<numerical::Point>,
    pub policy: Policy,
    pub cooperation: &'a mut ScalarCooperation,
}
impl LeastSquaresEvaluator for Original<'_> {
    fn accept_residual(&mut self) {
        self.accepted = self.pending.take();
    }
    fn cooperation(&mut self) -> Option<&mut ScalarCooperation> {
        Some(&mut *self.cooperation)
    }
    fn residual<'a>(&'a mut self, x: &'a [f64]) -> ResidualFuture<'a> {
        Box::pin(self.probe(x))
    }
}
impl Original<'_> {
    pub(super) async fn probe(&mut self, x: &[f64]) -> BuiltinResult<Vec<f64>> {
        self.pending = None;
        self.cooperation.completed(0).await?;
        let point = numerical::Point::evaluate_layout(
            self.source,
            self.layout,
            None,
            self.roots,
            x,
            self.policy,
            self.cooperation,
        )
        .await?
        .ok_or_else(|| {
            self.policy
                .invalid("residual evaluation is nonreal or outside its original numerical domain")
        })?;
        let values = match &self.projection {
            Projection::Equations(evidence) => {
                evidence
                    .residual(&point, self.policy, self.cooperation)
                    .await?
            }
            Projection::Squares { objective, squares } => {
                if !point.at(*objective, 0, self.policy)?.is_finite() {
                    return Err(self
                        .policy
                        .invalid("original nonlinear least-squares objective must be finite"));
                }
                let mut values = runmat_value::exact_capacity_vec(self.rows)
                    .map_err(|e| self.policy.storage(e))?;
                for term in &squares.terms {
                    let scale = term.weight.proposal.sqrt();
                    let node = self
                        .source
                        .nodes
                        .get(&term.residual)
                        .ok_or_else(|| self.policy.invalid("residual lost original source"))?
                        .node();
                    for row in 0..evaluation::element_count(&node.shape, self.policy)? {
                        let value = scale * point.at(term.residual, row, self.policy)?;
                        if !value.is_finite() {
                            return Err(self
                                .policy
                                .invalid("nonlinear residual evaluation must be finite"));
                        }
                        values.push(value);
                        self.cooperation.completed(1).await?;
                    }
                }
                values
            }
        };
        if values.len() != self.rows {
            return Err(self.policy.invalid("original residual shape changed"));
        }
        self.pending = Some(point);
        Ok(values)
    }
}
