//! Coefficient and signal input ownership and materialization.
use super::admission::ensure_vector_shape;
use super::*;

pub(super) struct CoeffInput {
    pub(super) data: Vec<Complex<f64>>,
    pub(super) len: usize,
    pub(super) is_complex: bool,
    pub(super) original_handle: Option<GpuTensorHandle>,
}

#[derive(Clone, Copy)]
pub(super) enum CoefficientRole {
    Numerator,
    Denominator,
}

impl CoefficientRole {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Numerator => "numerator",
            Self::Denominator => "denominator",
        }
    }

    pub(super) fn empty_error(self) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Numerator => &FILTER_ERROR_EMPTY_NUMERATOR,
            Self::Denominator => &FILTER_ERROR_EMPTY_DENOMINATOR,
        }
    }
}

impl CoeffInput {
    pub(super) async fn from_value(
        role: CoefficientRole,
        value: Value,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => {
                let original_handle = handle.clone();
                let tensor = gpu_helpers::gather_tensor_async(&handle)
                    .await
                    .map_err(|flow| {
                        filter_error_with_source(
                            &FILTER_ERROR_GATHER_FAILED,
                            "coefficient gather failed",
                            map_control_flow_with_builtin(flow, BUILTIN_NAME),
                        )
                    })?;
                let mut decoded = Self::from_tensor(role, tensor, cooperation).await?;
                decoded.original_handle = Some(original_handle);
                Ok(decoded)
            }
            Value::Tensor(tensor) => Self::from_tensor(role, tensor, cooperation).await,
            Value::ComplexTensor(tensor) => {
                let len = tensor.len();
                if len == 0 {
                    return Err(filter_error_with_detail(
                        role.empty_error(),
                        format!(
                            "{BUILTIN_NAME}: {} coefficients cannot be empty",
                            role.label()
                        ),
                    ));
                }
                ensure_vector_shape(role, tensor.shape())?;
                let data = preparation::complex_tensor(&tensor, cooperation).await?;
                Ok(Self {
                    data,
                    len,
                    is_complex: true,
                    original_handle: None,
                })
            }
            Value::LogicalArray(logical) => {
                ensure_vector_shape(role, &logical.shape)?;
                if logical.data.is_empty() {
                    return Err(filter_error(role.empty_error()));
                }
                Ok(Self {
                    len: logical.data.len(),
                    data: preparation::logical_array(&logical, cooperation).await?,
                    is_complex: false,
                    original_handle: None,
                })
            }
            Value::Num(n) => Ok(Self {
                data: vec![Complex::new(n, 0.0)],
                len: 1,
                is_complex: false,
                original_handle: None,
            }),
            Value::Int(i) => Ok(Self {
                data: vec![Complex::new(i.to_f64(), 0.0)],
                len: 1,
                is_complex: false,
                original_handle: None,
            }),
            Value::Bool(b) => Ok(Self {
                data: vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)],
                len: 1,
                is_complex: false,
                original_handle: None,
            }),
            Value::Complex(re, im) => Ok(Self {
                data: vec![Complex::new(re, im)],
                len: 1,
                is_complex: true,
                original_handle: None,
            }),
            other => Err(filter_error_with_detail(
                &FILTER_ERROR_INVALID_COEFFICIENTS,
                format!("{BUILTIN_NAME}: received {other:?}"),
            )),
        }
    }

    async fn from_tensor(
        role: CoefficientRole,
        tensor: Tensor,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<Self> {
        ensure_vector_shape(role, tensor.shape())?;
        let len = tensor.len();
        if len == 0 {
            return Err(filter_error_with_detail(
                role.empty_error(),
                format!(
                    "{BUILTIN_NAME}: {} coefficients cannot be empty",
                    role.label()
                ),
            ));
        }
        let data = preparation::real_tensor(&tensor, cooperation).await?;
        Ok(Self {
            data,
            len,
            is_complex: false,
            original_handle: None,
        })
    }
}

pub(super) struct SignalInput {
    payload: SignalPayload,
    pub(super) shape: Vec<usize>,
    pub(super) is_complex: bool,
    pub(super) output_single: bool,
}

enum SignalPayload {
    Host(Vec<Complex<f64>>),
    Resident(GpuTensorHandle),
}

impl SignalInput {
    pub(super) async fn from_value(
        value: Value,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<Self> {
        match value {
            Value::GpuTensor(handle) => {
                let output_single = runmat_accelerate_api::handle_integer_type(&handle).is_none()
                    && !runmat_accelerate_api::handle_is_logical(&handle)
                    && runmat_accelerate_api::handle_precision(&handle)
                        == Some(runmat_accelerate_api::ProviderPrecision::F32);
                let shape = handle.shape().to_vec();
                let is_complex = runmat_accelerate_api::handle_storage(&handle)
                    == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved;
                Ok(Self {
                    payload: SignalPayload::Resident(handle),
                    shape,
                    is_complex,
                    output_single,
                })
            }
            Value::Tensor(tensor) => {
                let output_single = tensor.numeric_dtype() == NumericDType::F32;
                let shape = tensor.shape().to_vec();
                let data = preparation::real_tensor(&tensor, cooperation).await?;
                Ok(Self {
                    payload: SignalPayload::Host(data),
                    shape,
                    is_complex: false,
                    output_single,
                })
            }
            Value::ComplexTensor(tensor) => {
                let output_single = tensor.numeric_dtype() == NumericDType::F32;
                let shape = tensor.shape().to_vec();
                let data = preparation::complex_tensor(&tensor, cooperation).await?;
                Ok(Self {
                    payload: SignalPayload::Host(data),
                    shape,
                    is_complex: true,
                    output_single,
                })
            }
            Value::LogicalArray(logical) => {
                let shape = logical.shape.to_vec();
                let data = preparation::logical_array(&logical, cooperation).await?;
                Ok(Self {
                    payload: SignalPayload::Host(data),
                    shape,
                    is_complex: false,
                    output_single: false,
                })
            }
            Value::Num(n) => Ok(Self {
                payload: SignalPayload::Host(vec![Complex::new(n, 0.0)]),
                shape: vec![1, 1],
                is_complex: false,
                output_single: false,
            }),
            Value::Int(i) => Ok(Self {
                payload: SignalPayload::Host(vec![Complex::new(i.to_f64(), 0.0)]),
                shape: vec![1, 1],
                is_complex: false,
                output_single: false,
            }),
            Value::Bool(b) => Ok(Self {
                payload: SignalPayload::Host(vec![Complex::new(if b { 1.0 } else { 0.0 }, 0.0)]),
                shape: vec![1, 1],
                is_complex: false,
                output_single: false,
            }),
            Value::Complex(re, im) => Ok(Self {
                payload: SignalPayload::Host(vec![Complex::new(re, im)]),
                shape: vec![1, 1],
                is_complex: true,
                output_single: false,
            }),
            other => Err(filter_error_with_detail(
                &FILTER_ERROR_INVALID_SIGNAL,
                format!("received {other:?}"),
            )),
        }
    }

    pub(super) fn resident(&self) -> Option<&GpuTensorHandle> {
        match &self.payload {
            SignalPayload::Resident(handle) => Some(handle),
            SignalPayload::Host(_) => None,
        }
    }

    pub(super) fn host_data(&self) -> BuiltinResult<&[Complex<f64>]> {
        match &self.payload {
            SignalPayload::Host(data) => Ok(data),
            SignalPayload::Resident(_) => Err(filter_error(&FILTER_ERROR_INTERNAL)),
        }
    }

    pub(super) async fn materialize(
        &mut self,
        cooperation: &mut SignalCooperation,
    ) -> BuiltinResult<()> {
        let SignalPayload::Resident(handle) = &self.payload else {
            return Ok(());
        };
        let value = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
            .await
            .map_err(|flow| {
                filter_error_with_source(
                    &FILTER_ERROR_GATHER_FAILED,
                    "signal gather failed",
                    map_control_flow_with_builtin(flow, BUILTIN_NAME),
                )
            })?;
        let materialized = Self::from_value(value, cooperation).await?;
        if materialized.shape != self.shape
            || materialized.is_complex != self.is_complex
            || !matches!(&materialized.payload, SignalPayload::Host(_))
        {
            return Err(filter_error(&FILTER_ERROR_INVALID_SIGNAL));
        }
        self.payload = materialized.payload;
        Ok(())
    }
}
