//! Initial-state admission and deferred materialization.
use super::admission::shapes_compatible;
use super::*;

pub(super) struct InitialState {
    pub(super) provided: bool,
    pub(super) column_major: Vec<Complex<f64>>,
    pub(super) shape: Vec<usize>,
    pub(super) is_complex: bool,
    pub(super) gpu_handle: Option<GpuTensorHandle>,
}

impl InitialState {
    pub(super) fn empty(shape: Vec<usize>) -> Self {
        Self {
            provided: false,
            column_major: Vec::new(),
            shape,
            is_complex: false,
            gpu_handle: None,
        }
    }

    pub(super) async fn materialize(
        &mut self,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<()> {
        let Some(handle) = &self.gpu_handle else {
            return Ok(());
        };
        if !self.column_major.is_empty() || self.shape.contains(&0) {
            return Ok(());
        }
        let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await
            .map_err(|source| {
                let mut error = filter_error_with_source(
                    &FILTER_ERROR_GATHER_FAILED,
                    "initial-state gather failed",
                    source,
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                error
            })?;
        let expected_states = self
            .shape
            .iter()
            .copied()
            .try_fold(1usize, |count, extent| count.checked_mul(extent))
            .ok_or_else(|| filter_error(&FILTER_ERROR_INVALID_INITIAL_STATE))?;
        let decoded = Self::from_value(
            gathered,
            self.shape[0],
            &self.shape,
            expected_states,
            cooperation,
        )
        .await?;
        self.column_major = decoded.column_major;
        self.is_complex = decoded.is_complex;
        Ok(())
    }

    pub(super) async fn from_value(
        value: Value,
        state_len: usize,
        expected_shape: &[usize],
        expected_states: usize,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<Self> {
        let name = BUILTIN_NAME;
        if state_len == 0 {
            match value {
                Value::Tensor(tensor) if tensor::tensor_element_len(&tensor) == 0 => {
                    return Ok(Self::empty(expected_shape.to_vec()))
                }
                Value::GpuTensor(handle) => {
                    if handle.shape().iter().all(|&extent| extent != 0) {
                        return Err(filter_error_with_detail(
                            &FILTER_ERROR_INVALID_INITIAL_STATE,
                            format!(
                                "{name}: initial conditions must be empty when the filter order is zero"
                            ),
                        ));
                    }
                    return Ok(Self {
                        provided: true,
                        column_major: Vec::new(),
                        shape: expected_shape.to_vec(),
                        is_complex: runmat_accelerate_api::handle_storage(&handle)
                            == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                        gpu_handle: Some(handle),
                    });
                }
                Value::LogicalArray(logical) if logical.data.is_empty() => {
                    return Ok(Self::empty(expected_shape.to_vec()))
                }
                Value::ComplexTensor(tensor)
                    if tensor::complex_tensor_element_len(&tensor) == 0 =>
                {
                    return Ok(Self {
                        provided: true,
                        column_major: Vec::new(),
                        shape: expected_shape.to_vec(),
                        is_complex: true,
                        gpu_handle: None,
                    });
                }
                other => {
                    let msg = format!(
                        "{name}: initial conditions must be empty when the filter order is zero"
                    );
                    let detail = match other {
                        Value::Tensor(t)
                            if t.is_empty() && !shapes_compatible(expected_shape, t.shape()) =>
                        {
                            msg.clone()
                        }
                        Value::Tensor(_) | Value::ComplexTensor(_) | Value::GpuTensor(_) => {
                            msg.clone()
                        }
                        _ => format!("{msg}; received {:?}", other),
                    };
                    return Err(filter_error_with_detail(
                        &FILTER_ERROR_INVALID_INITIAL_STATE,
                        detail,
                    ));
                }
            }
        }

        let (column_major, shape, is_complex, gpu_handle) = match value {
            Value::GpuTensor(handle) => {
                let shape = handle.shape().to_vec();
                (
                    Vec::new(),
                    shape,
                    runmat_accelerate_api::handle_storage(&handle)
                        == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                    Some(handle),
                )
            }
            Value::Tensor(tensor) => {
                let shape = tensor.shape().to_vec();
                (
                    preparation::real_tensor(&tensor, cooperation).await?,
                    shape,
                    false,
                    None,
                )
            }
            Value::ComplexTensor(tensor) => (
                preparation::complex_tensor(&tensor, cooperation).await?,
                tensor.shape().to_vec(),
                true,
                None,
            ),
            Value::LogicalArray(logical) => {
                let shape = logical.shape.to_vec();
                (
                    preparation::logical_array(&logical, cooperation).await?,
                    shape,
                    false,
                    None,
                )
            }
            Value::Num(n) => (vec![Complex::new(n, 0.0)], vec![1, 1], false, None),
            Value::Int(i) => (vec![Complex::new(i.to_f64(), 0.0)], vec![1, 1], false, None),
            Value::Bool(b) => (
                vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)],
                vec![1, 1],
                false,
                None,
            ),
            Value::Complex(re, im) => (vec![Complex::new(re, im)], vec![1, 1], true, None),
            other => {
                return Err(filter_error_with_detail(
                    &FILTER_ERROR_INVALID_INITIAL_STATE,
                    format!("{name}: unsupported initial condition type {other:?}"),
                ))
            }
        };

        let actual_states = if gpu_handle.is_some() {
            shape
                .iter()
                .copied()
                .try_fold(1usize, |count, extent| count.checked_mul(extent))
                .ok_or_else(|| filter_error(&FILTER_ERROR_INVALID_INITIAL_STATE))?
        } else {
            column_major.len()
        };
        if actual_states != expected_states {
            return Err(filter_error_with_detail(
                &FILTER_ERROR_INVALID_INITIAL_STATE,
                format!(
                    "{name}: initial conditions have {} elements but {} were expected",
                    actual_states, expected_states
                ),
            ));
        }

        if !shapes_compatible(expected_shape, &shape) {
            return Err(filter_error_with_detail(
                &FILTER_ERROR_INVALID_INITIAL_STATE,
                format!(
                    "{name}: initial conditions must have shape {:?}, received {:?}",
                    expected_shape, shape
                ),
            ));
        }

        Ok(Self {
            provided: true,
            column_major,
            shape: expected_shape.to_vec(),
            is_complex,
            gpu_handle,
        })
    }
}
