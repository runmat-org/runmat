use runmat_value::{NumericStorage, Tensor};

use crate::builtins::deep_learning::model::plan::axes::AxisPlan;

use super::forward::{trace_index, RecurrentBackwardCache, RecurrentTrace};
use super::gates::{gate_derivative, gate_row, state, state_derivative, Gate, Real};
use super::geometry::{Geometry, OutputMode, RecurrentError, RecurrentOptions};
use super::required_slice;

pub(in crate::builtins::deep_learning) struct RecurrentGradients {
    pub input: Tensor,
    pub input_weights: Tensor,
    pub recurrent_weights: Tensor,
    pub bias: Tensor,
    pub initial_hidden: Tensor,
    pub initial_cell: Tensor,
}

type GradientValues<T> = (Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>);

pub(in crate::builtins::deep_learning) fn backward(
    input: &Tensor,
    input_weights: &Tensor,
    recurrent_weights: &Tensor,
    bias: &Tensor,
    initial_hidden: Option<&Tensor>,
    initial_cell: Option<&Tensor>,
    axes: &AxisPlan,
    options: RecurrentOptions,
    cache: &RecurrentBackwardCache,
    output_gradient: &Tensor,
    terminal_hidden_gradient: Option<&Tensor>,
    terminal_cell_gradient: Option<&Tensor>,
) -> Result<RecurrentGradients, RecurrentError> {
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
    if output_gradient.shape() != g.output_shape
        || &cache.source_axes != axes
        || cache.output_axes != g.output_axes
        || output_gradient.numeric_dtype() != input.numeric_dtype()
    {
        return Err(RecurrentError::Invalid(
            "forward output and gradient must match admitted recurrent shape, axes, and class",
        ));
    }
    let state_shape = vec![g.directions * options.hidden_size, g.batches];
    for gradient in [terminal_hidden_gradient, terminal_cell_gradient]
        .into_iter()
        .flatten()
    {
        if gradient.shape() != state_shape || gradient.numeric_dtype() != input.numeric_dtype() {
            return Err(RecurrentError::Invalid(
                "terminal hidden/cell gradients must match output state shape and precision",
            ));
        }
    }
    let expected_trace_len = 9 * g.directions * g.options.hidden_size * g.batches * g.steps;
    let trace_len = match &cache.trace {
        RecurrentTrace::F32(values) => values.len(),
        RecurrentTrace::F64(values) => values.len(),
    };
    if trace_len != expected_trace_len {
        return Err(RecurrentError::Invalid(
            "captured recurrent trace has incompatible geometry",
        ));
    }
    let gradients = match (
        input.as_f32_slice(),
        input_weights.as_f32_slice(),
        recurrent_weights.as_f32_slice(),
        output_gradient.as_f32_slice(),
        &cache.trace,
    ) {
        (Some(x), Some(w), Some(r), Some(dy), RecurrentTrace::F32(trace)) => {
            let h0 = required_slice(initial_hidden.map(Tensor::as_f32_slice))?;
            let c0 = required_slice(initial_cell.map(Tensor::as_f32_slice))?;
            let dh_terminal = required_slice(terminal_hidden_gradient.map(Tensor::as_f32_slice))?;
            let dc_terminal = required_slice(terminal_cell_gradient.map(Tensor::as_f32_slice))?;
            let (dx, dw, dr, db, dh0, dc0) =
                backward_values(x, w, r, h0, c0, dy, dh_terminal, dc_terminal, trace, &g);
            (
                NumericStorage::F32(dx),
                NumericStorage::F32(dw),
                NumericStorage::F32(dr),
                NumericStorage::F32(db),
                NumericStorage::F32(dh0),
                NumericStorage::F32(dc0),
            )
        }
        _ => match (
            input.as_f64_slice(),
            input_weights.as_f64_slice(),
            recurrent_weights.as_f64_slice(),
            output_gradient.as_f64_slice(),
            &cache.trace,
        ) {
            (Some(x), Some(w), Some(r), Some(dy), RecurrentTrace::F64(trace)) => {
                let h0 = required_slice(initial_hidden.map(Tensor::as_f64_slice))?;
                let c0 = required_slice(initial_cell.map(Tensor::as_f64_slice))?;
                let dh_terminal =
                    required_slice(terminal_hidden_gradient.map(Tensor::as_f64_slice))?;
                let dc_terminal = required_slice(terminal_cell_gradient.map(Tensor::as_f64_slice))?;
                let (dx, dw, dr, db, dh0, dc0) =
                    backward_values(x, w, r, h0, c0, dy, dh_terminal, dc_terminal, trace, &g);
                (
                    NumericStorage::F64(dx),
                    NumericStorage::F64(dw),
                    NumericStorage::F64(dr),
                    NumericStorage::F64(db),
                    NumericStorage::F64(dh0),
                    NumericStorage::F64(dc0),
                )
            }
            _ => {
                return Err(RecurrentError::Invalid(
                    "backward requires a matching captured forward trace",
                ))
            }
        },
    };
    let (dx, dw, dr, db, dh0, dc0) = gradients;
    let default_state_shape = vec![g.directions * options.hidden_size, g.state_batch];
    Ok(RecurrentGradients {
        input: Tensor::from_numeric_storage(dx, input.shape().to_vec())
            .map_err(RecurrentError::Storage)?,
        input_weights: Tensor::from_numeric_storage(dw, input_weights.shape().to_vec())
            .map_err(RecurrentError::Storage)?,
        recurrent_weights: Tensor::from_numeric_storage(dr, recurrent_weights.shape().to_vec())
            .map_err(RecurrentError::Storage)?,
        bias: Tensor::from_numeric_storage(db, bias.shape().to_vec())
            .map_err(RecurrentError::Storage)?,
        initial_hidden: Tensor::from_numeric_storage(
            dh0,
            initial_hidden.map_or_else(
                || default_state_shape.clone(),
                |state| state.shape().to_vec(),
            ),
        )
        .map_err(RecurrentError::Storage)?,
        initial_cell: Tensor::from_numeric_storage(
            dc0,
            initial_cell.map_or(default_state_shape, |state| state.shape().to_vec()),
        )
        .map_err(RecurrentError::Storage)?,
    })
}

fn traced<T: Real>(
    trace: &[T],
    g: &Geometry,
    slot: usize,
    direction: usize,
    batch: usize,
    time: usize,
    hidden: usize,
) -> T {
    trace[trace_index(g, slot, direction, batch, time, hidden)]
}

fn backward_values<T: Real>(
    x: &[T],
    w: &[T],
    r: &[T],
    h0: Option<&[T]>,
    c0: Option<&[T]>,
    dy: &[T],
    dh_terminal: Option<&[T]>,
    dc_terminal: Option<&[T]>,
    trace: &[T],
    g: &Geometry,
) -> GradientValues<T> {
    let hidden_size = g.options.hidden_size;
    let rows = 4 * hidden_size * g.directions;
    let mut dx = vec![T::zero(); x.len()];
    let mut dw = vec![T::zero(); w.len()];
    let mut dr = vec![T::zero(); r.len()];
    let mut db = vec![T::zero(); rows];
    let state_len = g.directions * hidden_size * g.state_batch;
    let mut dh0 = vec![T::zero(); state_len];
    let mut dc0 = vec![T::zero(); state_len];
    for direction in 0..g.directions {
        for batch in 0..g.batches {
            let mut dh_carry = vec![T::zero(); hidden_size];
            let mut dc_carry = vec![T::zero(); hidden_size];
            for hidden in 0..hidden_size {
                let terminal_index =
                    direction * hidden_size + hidden + batch * hidden_size * g.directions;
                dh_carry[hidden] =
                    dh_terminal.map_or(T::zero(), |gradient| gradient[terminal_index]);
                dc_carry[hidden] =
                    dc_terminal.map_or(T::zero(), |gradient| gradient[terminal_index]);
            }
            let mut dh_previous = vec![T::zero(); hidden_size];
            let mut dc_previous = vec![T::zero(); hidden_size];
            let mut prior_hidden = vec![T::zero(); hidden_size];
            let mut prior_cell = vec![T::zero(); hidden_size];
            let mut input_offsets = vec![0usize; g.input_channels];
            for reverse_step in 0..g.steps {
                let time = if direction == 0 {
                    g.steps - 1 - reverse_step
                } else {
                    reverse_step
                };
                let first_step = if direction == 0 {
                    time == 0
                } else {
                    time + 1 == g.steps
                };
                let prior_time = if direction == 0 {
                    time.saturating_sub(1)
                } else {
                    (time + 1).min(g.steps - 1)
                };
                for previous in 0..hidden_size {
                    let state_index = g.state_offset(direction, previous, batch);
                    prior_cell[previous] = if first_step {
                        c0.map_or(T::zero(), |state| state[state_index])
                    } else {
                        traced(trace, g, 8, direction, batch, prior_time, previous)
                    };
                    prior_hidden[previous] = if first_step {
                        h0.map_or(T::zero(), |state| state[state_index])
                    } else {
                        let prior_o = traced(trace, g, 7, direction, batch, prior_time, previous);
                        prior_o * state(g.options.state_activation, prior_cell[previous])
                    };
                }
                for (channel, offset) in input_offsets.iter_mut().enumerate() {
                    *offset = g.input_offset(channel, batch, time);
                }
                for hidden in 0..hidden_size {
                    let previous_cell = prior_cell[hidden];
                    let i = traced(trace, g, 4, direction, batch, time, hidden);
                    let f = traced(trace, g, 5, direction, batch, time, hidden);
                    let candidate = traced(trace, g, 6, direction, batch, time, hidden);
                    let o = traced(trace, g, 7, direction, batch, time, hidden);
                    let cell = traced(trace, g, 8, direction, batch, time, hidden);
                    let activated_cell = state(g.options.state_activation, cell);
                    let terminal = if direction == 0 {
                        time + 1 == g.steps
                    } else {
                        time == 0
                    };
                    let dy_here = if g.options.output_mode == OutputMode::Sequence || terminal {
                        dy[g.output_offset(direction * hidden_size + hidden, batch, time)]
                    } else {
                        T::zero()
                    };
                    let dh = dy_here + dh_carry[hidden];
                    let dc = dc_carry[hidden]
                        + dh * o
                            * state_derivative(g.options.state_activation, cell, activated_cell);
                    let d_input = dc
                        * candidate
                        * gate_derivative(
                            g.options.gate_activation,
                            traced(trace, g, 0, direction, batch, time, hidden),
                            i,
                        );
                    let d_forget = dc
                        * previous_cell
                        * gate_derivative(
                            g.options.gate_activation,
                            traced(trace, g, 1, direction, batch, time, hidden),
                            f,
                        );
                    let d_candidate = dc
                        * i
                        * state_derivative(
                            g.options.state_activation,
                            traced(trace, g, 2, direction, batch, time, hidden),
                            candidate,
                        );
                    let d_output = dh
                        * activated_cell
                        * gate_derivative(
                            g.options.gate_activation,
                            traced(trace, g, 3, direction, batch, time, hidden),
                            o,
                        );
                    dc_previous[hidden] = dc * f;
                    for (gate, gradient) in [
                        (Gate::Input, d_input),
                        (Gate::Forget, d_forget),
                        (Gate::Candidate, d_candidate),
                        (Gate::Output, d_output),
                    ] {
                        let row = gate_row(direction, gate, hidden, hidden_size);
                        db[row] += gradient;
                        for (channel, &input_index) in input_offsets.iter().enumerate() {
                            dw[row + channel * rows] += gradient * x[input_index];
                            dx[input_index] += gradient * w[row + channel * rows];
                        }
                        for previous in 0..hidden_size {
                            dr[row + previous * rows] += gradient * prior_hidden[previous];
                            dh_previous[previous] += gradient * r[row + previous * rows];
                        }
                    }
                }
                if first_step {
                    for hidden in 0..hidden_size {
                        let index = g.state_offset(direction, hidden, batch);
                        dh0[index] += dh_previous[hidden];
                        dc0[index] += dc_previous[hidden];
                    }
                }
                std::mem::swap(&mut dh_carry, &mut dh_previous);
                std::mem::swap(&mut dc_carry, &mut dc_previous);
                dh_previous.fill(T::zero());
                dc_previous.fill(T::zero());
            }
        }
    }
    (dx, dw, dr, db, dh0, dc0)
}
