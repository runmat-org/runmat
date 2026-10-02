//! Checked logical index plans for zero-phase vector reflection and trim.
use runmat_accelerate_api::ProviderIirZeroPhaseRequest;

pub(super) struct IndexPlan {
    pub edge: Vec<u32>,
    pub neighbor: Vec<u32>,
    pub first_endpoint: Vec<u32>,
    pub second_endpoint: Vec<u32>,
    pub trim: Vec<u32>,
    pub extended_shape: [usize; 2],
    pub state_shape: [usize; 2],
}

impl IndexPlan {
    pub fn new(request: &ProviderIirZeroPhaseRequest<'_>) -> anyhow::Result<Self> {
        let signal_len = request
            .input
            .shape()
            .iter()
            .try_fold(1usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| anyhow::anyhow!("iir_zero_phase: signal extent overflow"))?;
        let padding = request.reflection_len;
        let extended_len = padding
            .checked_mul(2)
            .and_then(|count| signal_len.checked_add(count))
            .ok_or_else(|| anyhow::anyhow!("iir_zero_phase: reflection extent overflow"))?;
        let state_len = request.initial_state.len();
        let mut edge = Vec::new();
        let mut neighbor = Vec::new();
        let mut first_endpoint = Vec::new();
        let mut second_endpoint = Vec::new();
        let mut trim = Vec::new();
        for (target, capacity) in [
            (&mut edge, extended_len),
            (&mut neighbor, extended_len),
            (&mut first_endpoint, state_len),
            (&mut second_endpoint, state_len),
            (&mut trim, signal_len),
        ] {
            target.try_reserve_exact(capacity)?;
        }
        for offset in 0..padding {
            edge.push(0);
            neighbor.push(u32::try_from(padding - offset)?);
        }
        for index in 0..signal_len {
            let index = u32::try_from(index)?;
            edge.push(index);
            neighbor.push(index);
        }
        for offset in 0..padding {
            edge.push(u32::try_from(signal_len - 1)?);
            neighbor.push(u32::try_from(signal_len - 2 - offset)?);
        }
        first_endpoint.resize(state_len, 0);
        second_endpoint.resize(state_len, u32::try_from(extended_len - 1)?);
        for index in padding..padding + signal_len {
            trim.push(u32::try_from(index)?);
        }
        let extended_shape = if request.axis == 0 {
            [extended_len, 1]
        } else {
            [1, extended_len]
        };
        let state_shape = if request.axis == 0 {
            [state_len, 1]
        } else {
            [1, state_len]
        };
        Ok(Self {
            edge,
            neighbor,
            first_endpoint,
            second_endpoint,
            trim,
            extended_shape,
            state_shape,
        })
    }
}
