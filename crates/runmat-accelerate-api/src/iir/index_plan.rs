//! One sealed, bounded-work reflection index plan for resident zero-phase IIR.
//! The invocation owner advances the builder between cooperation checkpoints;
//! providers receive only the completed immutable plan.

use runmat_execution::cancellation::CancellationToken;
use runmat_execution::CancellationReason;

const INDEX_CHUNK: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Reflection,
    FirstEndpoint,
    SecondEndpoint,
    Trim,
    Complete,
}

#[derive(Debug)]
pub struct ProviderIirZeroPhaseIndexPlan {
    signal_shape: [usize; 2],
    axis: usize,
    reflection_len: usize,
    coefficient_len: usize,
    edge: Vec<u32>,
    neighbor: Vec<u32>,
    first_endpoint: Vec<u32>,
    second_endpoint: Vec<u32>,
    trim: Vec<u32>,
    extended_shape: [usize; 2],
    state_shape: [usize; 2],
}

impl ProviderIirZeroPhaseIndexPlan {
    pub fn matches(
        &self,
        signal_shape: &[usize],
        coefficient_len: usize,
        axis: usize,
        reflection_len: usize,
    ) -> bool {
        signal_shape == self.signal_shape.as_slice()
            && coefficient_len == self.coefficient_len
            && axis == self.axis
            && reflection_len == self.reflection_len
    }

    pub fn edge(&self) -> &[u32] {
        &self.edge
    }
    pub fn neighbor(&self) -> &[u32] {
        &self.neighbor
    }
    pub fn first_endpoint(&self) -> &[u32] {
        &self.first_endpoint
    }
    pub fn second_endpoint(&self) -> &[u32] {
        &self.second_endpoint
    }
    pub fn trim(&self) -> &[u32] {
        &self.trim
    }
    pub fn extended_shape(&self) -> &[usize; 2] {
        &self.extended_shape
    }
    pub fn state_shape(&self) -> &[usize; 2] {
        &self.state_shape
    }

    /// Direct provider callers have no runtime cooperation service. They
    /// still check the same invocation token before allocation and after each
    /// bounded construction chunk.
    pub fn prepare_direct(
        signal_shape: [usize; 2],
        coefficient_len: usize,
        axis: usize,
        reflection_len: usize,
        cancellation: &CancellationToken,
    ) -> anyhow::Result<Self> {
        check_cancel(cancellation)?;
        let mut builder = ProviderIirZeroPhaseIndexBuilder::new(
            signal_shape,
            coefficient_len,
            axis,
            reflection_len,
        )?;
        while builder.advance_chunk().is_some() {
            check_cancel(cancellation)?;
        }
        builder.finish()
    }
}

#[derive(Debug)]
pub struct ProviderIirZeroPhaseIndexBuilder {
    plan: ProviderIirZeroPhaseIndexPlan,
    phase: Phase,
    position: usize,
    signal_len: usize,
    extended_len: usize,
}

impl ProviderIirZeroPhaseIndexBuilder {
    pub fn new(
        signal_shape: [usize; 2],
        coefficient_len: usize,
        axis: usize,
        reflection_len: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(axis <= 1, "iir_zero_phase: invalid filtered axis");
        let signal_len = signal_shape[axis];
        anyhow::ensure!(
            signal_len != 0 && signal_shape[1 - axis] == 1 && coefficient_len != 0,
            "iir_zero_phase: invalid vector or coefficient geometry"
        );
        let state_len = coefficient_len - 1;
        anyhow::ensure!(
            reflection_len
                == state_len.checked_mul(3).ok_or_else(|| {
                    anyhow::anyhow!("iir_zero_phase: reflection length overflow")
                })?
                && signal_len > reflection_len,
            "iir_zero_phase: reflection exceeds signal length"
        );
        let extended_len = reflection_len
            .checked_mul(2)
            .and_then(|padding| signal_len.checked_add(padding))
            .ok_or_else(|| anyhow::anyhow!("iir_zero_phase: extended signal overflow"))?;
        u32::try_from(extended_len - 1)?;
        u32::try_from(state_len)?;

        let mut edge = Vec::new();
        let mut neighbor = Vec::new();
        let mut first_endpoint = Vec::new();
        let mut second_endpoint = Vec::new();
        let mut trim = Vec::new();
        edge.try_reserve_exact(extended_len)?;
        neighbor.try_reserve_exact(extended_len)?;
        first_endpoint.try_reserve_exact(state_len)?;
        second_endpoint.try_reserve_exact(state_len)?;
        trim.try_reserve_exact(signal_len)?;
        let extended_shape = if axis == 0 {
            [extended_len, 1]
        } else {
            [1, extended_len]
        };
        // Causal IIR states are order-first even when samples run along the
        // second axis. A vector has one channel on either axis.
        let state_shape = [state_len, 1];
        Ok(Self {
            plan: ProviderIirZeroPhaseIndexPlan {
                signal_shape,
                axis,
                reflection_len,
                coefficient_len,
                edge,
                neighbor,
                first_endpoint,
                second_endpoint,
                trim,
                extended_shape,
                state_shape,
            },
            phase: Phase::Reflection,
            position: 0,
            signal_len,
            extended_len,
        })
    }

    /// Performs at most `INDEX_CHUNK` index writes and returns the exact work
    /// to charge to the caller's existing invocation budget. `None` means the
    /// plan is complete and ready to consume.
    pub fn advance_chunk(&mut self) -> Option<usize> {
        if self.phase == Phase::Complete {
            return None;
        }
        let mut work = 0;
        while work < INDEX_CHUNK && self.phase != Phase::Complete {
            if self.phase == Phase::Reflection && work + 2 > INDEX_CHUNK {
                break;
            }
            let limit = match self.phase {
                Phase::Reflection => self.extended_len,
                Phase::FirstEndpoint | Phase::SecondEndpoint => self.coefficient_len() - 1,
                Phase::Trim => self.signal_len,
                Phase::Complete => break,
            };
            if self.position == limit {
                self.phase = match self.phase {
                    Phase::Reflection => Phase::FirstEndpoint,
                    Phase::FirstEndpoint => Phase::SecondEndpoint,
                    Phase::SecondEndpoint => Phase::Trim,
                    Phase::Trim => Phase::Complete,
                    Phase::Complete => Phase::Complete,
                };
                self.position = 0;
                continue;
            }
            match self.phase {
                Phase::Reflection => {
                    let position = self.position;
                    let padding = self.plan.reflection_len;
                    let (edge, neighbor) = if position < padding {
                        (0, padding - position)
                    } else if position < padding + self.signal_len {
                        let source = position - padding;
                        (source, source)
                    } else {
                        let offset = position - padding - self.signal_len;
                        (self.signal_len - 1, self.signal_len - 2 - offset)
                    };
                    self.plan.edge.push(edge as u32);
                    self.plan.neighbor.push(neighbor as u32);
                    work += 2;
                }
                Phase::FirstEndpoint => {
                    self.plan.first_endpoint.push(0);
                    work += 1;
                }
                Phase::SecondEndpoint => {
                    self.plan
                        .second_endpoint
                        .push((self.extended_len - 1) as u32);
                    work += 1;
                }
                Phase::Trim => {
                    self.plan
                        .trim
                        .push((self.plan.reflection_len + self.position) as u32);
                    work += 1;
                }
                Phase::Complete => break,
            }
            self.position += 1;
        }
        (work != 0).then_some(work)
    }

    fn coefficient_len(&self) -> usize {
        self.plan.coefficient_len
    }

    pub fn finish(self) -> anyhow::Result<ProviderIirZeroPhaseIndexPlan> {
        anyhow::ensure!(
            self.phase == Phase::Complete,
            "iir_zero_phase: incomplete index preparation"
        );
        Ok(self.plan)
    }
}

#[derive(Debug)]
pub struct ProviderIirIndexCancelled {
    pub reason: Option<CancellationReason>,
}

impl std::fmt::Display for ProviderIirIndexCancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "iir_zero_phase: index preparation cancelled ({:?})",
            self.reason
        )
    }
}

impl std::error::Error for ProviderIirIndexCancelled {}

fn check_cancel(cancellation: &CancellationToken) -> anyhow::Result<()> {
    if cancellation.is_cancelled() {
        return Err(ProviderIirIndexCancelled {
            reason: cancellation.reason(),
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::cancellation::CancellationSource;

    #[test]
    fn row_column_reflection_and_trim_share_one_sealed_builder() {
        for (shape, axis) in [([8, 1], 0), ([1, 8], 1)] {
            let mut builder = ProviderIirZeroPhaseIndexBuilder::new(shape, 2, axis, 3).unwrap();
            let mut charged = 0;
            while let Some(work) = builder.advance_chunk() {
                assert!(work <= INDEX_CHUNK);
                charged += work;
            }
            let plan = builder.finish().unwrap();
            assert_eq!(plan.edge(), &[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7]);
            assert_eq!(plan.neighbor(), &[3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4]);
            assert_eq!(plan.trim(), &[3, 4, 5, 6, 7, 8, 9, 10]);
            assert_eq!(charged, 2 * 14 + 2 + 8);
            assert_eq!(plan.state_shape(), &[1, 1]);
            assert!(plan.matches(&shape, 2, axis, 3));
            assert!(!plan.matches(&shape, 2, 1 - axis, 3));
        }
    }

    #[test]
    fn row_vector_multistate_geometry_keeps_order_as_leading_axis() {
        let source = CancellationSource::new();
        let plan = ProviderIirZeroPhaseIndexPlan::prepare_direct([1, 12], 3, 1, 6, &source.token())
            .unwrap();
        assert_eq!(plan.extended_shape(), &[1, 24]);
        assert_eq!(plan.state_shape(), &[2, 1]);
    }

    #[test]
    fn direct_preparation_checks_cancel_before_allocation_and_between_chunks() {
        let source = CancellationSource::new();
        source.cancel(CancellationReason::User);
        let error = ProviderIirZeroPhaseIndexPlan::prepare_direct([8, 1], 2, 0, 3, &source.token())
            .unwrap_err();
        assert!(error.is::<ProviderIirIndexCancelled>());
    }
}
