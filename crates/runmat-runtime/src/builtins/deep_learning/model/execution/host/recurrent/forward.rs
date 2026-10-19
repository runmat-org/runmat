use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::gates::{gate, gate_row, state, Gate, Real};
use super::geometry::{Geometry, OutputMode, RecurrentError, RecurrentOptions};
use super::required_slice;

const GATE_COUNT: usize = 4;
const TRACE_SLOTS: usize = 9; // Four preactivations, four gate values, one cell value.

pub(in crate::builtins::deep_learning) struct RecurrentOutput {
    pub tensor: Tensor,
    pub axes: AxisPlan,
    pub hidden_state: Tensor,
    pub cell_state: Tensor,
    pub cache: Option<RecurrentBackwardCache>,
}

pub(in crate::builtins::deep_learning) struct RecurrentBackwardCache {
    pub(super) source_axes: AxisPlan,
    pub(super) output_axes: AxisPlan,
    pub(super) trace: RecurrentTrace,
}

pub(super) enum RecurrentTrace {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

pub(super) fn trace_index(
    g: &Geometry,
    slot: usize,
    direction: usize,
    batch: usize,
    time: usize,
    hidden: usize,
) -> usize {
    (((slot * g.directions + direction) * g.batches + batch) * g.steps + time)
        * g.options.hidden_size
        + hidden
}

pub(in crate::builtins::deep_learning) fn forward(
    input: &Tensor,
    input_weights: &Tensor,
    recurrent_weights: &Tensor,
    bias: &Tensor,
    initial_hidden: Option<&Tensor>,
    initial_cell: Option<&Tensor>,
    axes: &AxisPlan,
    options: RecurrentOptions,
    capture_trace: bool,
) -> Result<RecurrentOutput, RecurrentError> {
    let g = Geometry::new(
        input,
        input_weights,
        recurrent_weights,
        bias,
        initial_hidden,
        initial_cell,
        axes,
        options,
    )?;
    let (output, hidden, cell, trace) = if let (Some(x), Some(w), Some(r), Some(b)) = (
        input.as_f32_slice(),
        input_weights.as_f32_slice(),
        recurrent_weights.as_f32_slice(),
        bias.as_f32_slice(),
    ) {
        let h0 = required_slice(initial_hidden.map(Tensor::as_f32_slice))?;
        let c0 = required_slice(initial_cell.map(Tensor::as_f32_slice))?;
        let (y, h, c, trace) = forward_values(x, w, r, b, h0, c0, &g, capture_trace);
        (
            NumericStorage::F32(y),
            NumericStorage::F32(h),
            NumericStorage::F32(c),
            trace.map(RecurrentTrace::F32),
        )
    } else if let (Some(x), Some(w), Some(r), Some(b)) = (
        input.as_f64_slice(),
        input_weights.as_f64_slice(),
        recurrent_weights.as_f64_slice(),
        bias.as_f64_slice(),
    ) {
        let h0 = required_slice(initial_hidden.map(Tensor::as_f64_slice))?;
        let c0 = required_slice(initial_cell.map(Tensor::as_f64_slice))?;
        let (y, h, c, trace) = forward_values(x, w, r, b, h0, c0, &g, capture_trace);
        (
            NumericStorage::F64(y),
            NumericStorage::F64(h),
            NumericStorage::F64(c),
            trace.map(RecurrentTrace::F64),
        )
    } else {
        return Err(RecurrentError::UnsupportedPrecision);
    };
    let state_shape = vec![g.directions * options.hidden_size, g.batches];
    let cache = trace.map(|trace| RecurrentBackwardCache {
        source_axes: axes.clone(),
        output_axes: g.output_axes.clone(),
        trace,
    });
    Ok(RecurrentOutput {
        tensor: Tensor::from_numeric_storage(output, g.output_shape.clone())
            .map_err(RecurrentError::Storage)?,
        axes: g.output_axes,
        hidden_state: Tensor::from_numeric_storage(hidden, state_shape.clone())
            .map_err(RecurrentError::Storage)?,
        cell_state: Tensor::from_numeric_storage(cell, state_shape)
            .map_err(RecurrentError::Storage)?,
        cache,
    })
}

pub(super) fn forward_values<T: Real>(
    x: &[T],
    w: &[T],
    r: &[T],
    b: &[T],
    h0: Option<&[T]>,
    c0: Option<&[T]>,
    g: &Geometry,
    capture_trace: bool,
) -> (Vec<T>, Vec<T>, Vec<T>, Option<Vec<T>>) {
    let hidden_size = g.options.hidden_size;
    let output_len = g.output_axes.element_count();
    let mut output = vec![T::zero(); output_len];
    let state_len = g.directions * hidden_size * g.batches;
    let mut final_hidden = vec![T::zero(); state_len];
    let mut final_cell = vec![T::zero(); state_len];
    let mut trace = capture_trace.then(|| vec![T::zero(); TRACE_SLOTS * state_len * g.steps]);
    let rows = GATE_COUNT * hidden_size * g.directions;
    for direction in 0..g.directions {
        for batch in 0..g.batches {
            let mut h_prev = vec![T::zero(); hidden_size];
            let mut c_prev = vec![T::zero(); hidden_size];
            if let (Some(h0), Some(c0)) = (h0, c0) {
                for hidden in 0..hidden_size {
                    let index = g.state_offset(direction, hidden, batch);
                    h_prev[hidden] = h0[index];
                    c_prev[hidden] = c0[index];
                }
            }
            let mut pre = vec![T::zero(); GATE_COUNT * hidden_size];
            let mut h_next = vec![T::zero(); hidden_size];
            let mut c_next = vec![T::zero(); hidden_size];
            let mut input_offsets = vec![0usize; g.input_channels];
            for step in 0..g.steps {
                let time = if direction == 0 {
                    step
                } else {
                    g.steps - 1 - step
                };
                for (channel, offset) in input_offsets.iter_mut().enumerate() {
                    *offset = g.input_offset(channel, batch, time);
                }
                for (gate_number, gate_kind) in
                    [Gate::Input, Gate::Forget, Gate::Candidate, Gate::Output]
                        .iter()
                        .enumerate()
                {
                    for hidden in 0..hidden_size {
                        let row = gate_row(direction, *gate_kind, hidden, hidden_size);
                        let mut sum = b[row];
                        for (channel, &input_index) in input_offsets.iter().enumerate() {
                            sum += w[row + channel * rows] * x[input_index];
                        }
                        for previous in 0..hidden_size {
                            sum += r[row + previous * rows] * h_prev[previous];
                        }
                        pre[gate_number * hidden_size + hidden] = sum;
                        if let Some(trace) = &mut trace {
                            trace[trace_index(g, gate_number, direction, batch, time, hidden)] =
                                sum;
                        }
                    }
                }
                for hidden in 0..hidden_size {
                    let i = gate(g.options.gate_activation, pre[hidden]);
                    let f = gate(g.options.gate_activation, pre[hidden_size + hidden]);
                    let candidate =
                        state(g.options.state_activation, pre[2 * hidden_size + hidden]);
                    let o = gate(g.options.gate_activation, pre[3 * hidden_size + hidden]);
                    c_next[hidden] = f * c_prev[hidden] + i * candidate;
                    h_next[hidden] = o * state(g.options.state_activation, c_next[hidden]);
                    if let Some(trace) = &mut trace {
                        for (slot, value) in
                            [i, f, candidate, o, c_next[hidden]].into_iter().enumerate()
                        {
                            trace[trace_index(
                                g,
                                GATE_COUNT + slot,
                                direction,
                                batch,
                                time,
                                hidden,
                            )] = value;
                        }
                    }
                    if g.options.output_mode == OutputMode::Sequence || step + 1 == g.steps {
                        // In `Last`, the terminal state of each direction is the
                        // sequence summary. The caller owns public layer policy.
                        output[g.output_offset(direction * hidden_size + hidden, batch, time)] =
                            h_next[hidden];
                    }
                }
                std::mem::swap(&mut h_prev, &mut h_next);
                std::mem::swap(&mut c_prev, &mut c_next);
            }
            for hidden in 0..hidden_size {
                let index = direction * hidden_size + hidden + batch * hidden_size * g.directions;
                final_hidden[index] = h_prev[hidden];
                final_cell[index] = c_prev[hidden];
            }
        }
    }
    (output, final_hidden, final_cell, trace)
}
