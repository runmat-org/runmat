//! Random-QR and coordinate sampler frames; this is not deterministic MADS.
//! Only the existing Householder factor is retained, never a dense Q mirror.
use super::{invalid, IDENTITY};
use crate::builtins::math::linalg::ops::sparse_factor::orthogonal_frame::{
    OrthogonalFrame, OrthogonalSpan,
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Coordinate,
    RandomOrthogonal,
}
pub(super) struct Frame<'a> {
    dimension: usize,
    intrinsic: usize,
    basis: Option<OrthogonalFrame>,
    span: Option<&'a OrthogonalSpan>,
}
impl<'a> Frame<'a> {
    pub(super) async fn new(
        kind: Kind,
        dimension: usize,
        span: Option<&'a OrthogonalSpan>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let intrinsic = span.map_or(dimension, OrthogonalSpan::rank);
        if dimension == 0
            || intrinsic == 0
            || span.is_some_and(|span| span.dimension() != dimension)
            || intrinsic
                .checked_mul(2)
                .and_then(|n| n.checked_add(2))
                .is_none()
        {
            return Err(invalid("sampler frame dimension overflow"));
        }
        if matches!(kind, Kind::Coordinate) {
            return Ok(Some(Self {
                dimension,
                intrinsic,
                basis: None,
                span,
            }));
        }
        let (runtime, budget) = cooperation.runtime_budget();
        let Some(factor) = OrthogonalFrame::random(intrinsic, runtime, IDENTITY, budget).await?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            dimension,
            intrinsic,
            basis: Some(factor),
            span,
        }))
    }
    pub(super) fn count(&self) -> usize {
        self.intrinsic * 2 + 2
    }
    pub(super) async fn direction_into(
        &self,
        index: usize,
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if index >= self.count() || output.len() != self.dimension {
            return Err(invalid("invalid sampler direction extent"));
        }
        let sign = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
        let sum = index >= self.intrinsic * 2;
        let column = index / 2;
        for (row, value) in output.iter_mut().enumerate() {
            *value = if row < self.intrinsic && (sum || row == column) {
                sign
            } else {
                0.0
            };
            cooperation.completed(1).await?;
        }
        if !sum {
            if let Some(frame) = &self.basis {
                let (_, budget) = cooperation.runtime_budget();
                frame
                    .transform_into(&mut output[..self.intrinsic], budget)
                    .await?;
            }
        }
        if let Some(span) = self.span {
            let (_, budget) = cooperation.runtime_budget();
            span.embed_into(output, budget).await?;
        }
        Ok(())
    }
}

/// One frame, progressively halved steps. Exhaustion is representational,
/// never an unbounded retry or a fabricated feasible/optimality certificate.
pub(super) struct Rings<'a> {
    frame: Frame<'a>,
    next: usize,
    step: f64,
}
impl<'a> Rings<'a> {
    pub(super) fn new(frame: Frame<'a>, scale: f64) -> BuiltinResult<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid("invalid sampler scale"));
        }
        Ok(Self {
            frame,
            next: 0,
            step: scale,
        })
    }
    pub(super) async fn next_into(
        &mut self,
        output: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>> {
        if self.step == 0.0 {
            return Ok(None);
        }
        self.frame
            .direction_into(self.next, output, cooperation)
            .await?;
        let step = self.step;
        self.next += 1;
        if self.next == self.frame.count() {
            self.next = 0;
            self.step *= 0.5;
        }
        Ok(Some(step))
    }
}
