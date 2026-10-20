//! Host numeric admission and scalar-control decoding.
use super::*;

pub(super) struct AdamUpdateArgs {
    pub(super) parameters: NumericPayload,
    pub(super) gradient: NumericPayload,
    pub(super) average_grad: NumericPayload,
    pub(super) average_sq_grad: NumericPayload,
    pub(super) iteration: usize,
    pub(super) learn_rate: f64,
    pub(super) gradient_decay_factor: f64,
    pub(super) squared_gradient_decay_factor: f64,
    pub(super) epsilon: f64,
}

impl AdamUpdateArgs {
    pub(super) fn parse(args: crate::call::BuiltinArguments) -> BuiltinResult<Self> {
        if !(5..=9).contains(&args.len()) {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: expected 5 to 9 positional arguments",
            ));
        }
        let parameters = NumericPayload::parse(&args[0], "parameters", true)?;
        if parameters.data.is_empty() {
            return Err(deep_learning_error(
                "adamupdate",
                "adamupdate: parameters must not be empty",
            ));
        }
        let gradient = NumericPayload::parse(&args[1], "grad", true)?;
        let average_grad = NumericPayload::parse(&args[2], "averageGrad", false)?
            .with_default_shape(&parameters)?;
        let average_sq_grad = NumericPayload::parse(&args[3], "averageSqGrad", false)?
            .with_default_shape(&parameters)?;
        require_same_shape(&parameters, &gradient, "grad")?;
        require_same_shape(&parameters, &average_grad, "averageGrad")?;
        require_same_shape(&parameters, &average_sq_grad, "averageSqGrad")?;
        let iteration = positive_iteration(&args[4])?;
        Ok(Self {
            parameters,
            gradient,
            average_grad,
            average_sq_grad,
            iteration,
            learn_rate: optional_positive_scalar(args.view(), 5, 0.001, "learnRate")?,
            gradient_decay_factor: optional_unit_scalar(args.view(), 6, 0.9, "gradDecay")?,
            squared_gradient_decay_factor: optional_unit_scalar(
                args.view(),
                7,
                0.999,
                "sqGradDecay",
            )?,
            epsilon: optional_positive_scalar(args.view(), 8, 1.0e-8, "epsilon")?,
        })
    }
}

#[derive(Clone)]
pub(super) struct NumericPayload {
    pub(super) data: Vec<f64>,
    pub(super) shape: Vec<usize>,
    repr: NumericRepr,
}

impl NumericPayload {
    pub(super) fn parse(
        value: &Value,
        label: &'static str,
        reject_empty: bool,
    ) -> BuiltinResult<Self> {
        let payload = match value {
            Value::Num(n) if n.is_finite() => Self {
                data: vec![*n],
                shape: vec![1, 1],
                repr: NumericRepr::Scalar,
            },
            Value::Int(i) => Self {
                data: vec![i.to_f64()],
                shape: vec![1, 1],
                repr: NumericRepr::Scalar,
            },
            Value::Tensor(tensor) => {
                let dtype = tensor.numeric_dtype();
                if !matches!(dtype, NumericDType::F64 | NumericDType::F32) {
                    return Err(deep_learning_error(
                        "adamupdate",
                        format!(
                            "adamupdate: {label} tensor must be double or single, got {}",
                            dtype.class_name()
                        ),
                    ));
                }
                let data = tensor
                    .clone()
                    .into_numeric_storage()
                    .map_err(|err| deep_learning_error("adamupdate", err))?
                    .materialize_f64();
                ensure_finite(&data, label)?;
                Self {
                    data,
                    shape: tensor.shape().to_vec(),
                    repr: NumericRepr::Dense {
                        shape: tensor.shape().to_vec(),
                        dtype,
                    },
                }
            }
            Value::Object(object) if object.class_name().is(super::super::super::DLARRAY_CLASS) => {
                let data = object.properties().get("Data").ok_or_else(|| {
                    deep_learning_error("adamupdate", "adamupdate: dlarray is missing Data")
                })?;
                let inner = Self::parse(data, label, reject_empty)?;
                Self {
                    data: inner.data,
                    shape: inner.shape,
                    repr: NumericRepr::Dlarray {
                        inner: Box::new(inner.repr),
                        format: Box::new(
                            object
                                .properties()
                                .get("Format")
                                .cloned()
                                .unwrap_or_else(|| Value::string(String::new())),
                        ),
                        labels: Box::new(
                            object
                                .properties()
                                .get("Labels")
                                .cloned()
                                .unwrap_or_else(|| Value::string(String::new())),
                        ),
                    },
                }
            }
            Value::GpuTensor(_) => {
                return Err(deep_learning_error(
                    "adamupdate",
                    "adamupdate: gpuArray optimizer updates require provider kernels and are not implemented in this runtime slice",
                ));
            }
            other => {
                return Err(deep_learning_error(
                    "adamupdate",
                    format!("adamupdate: {label} must be a finite numeric array or dlarray, got {other:?}"),
                ));
            }
        };
        if reject_empty && payload.data.is_empty() {
            return Err(deep_learning_error(
                "adamupdate",
                format!("adamupdate: {label} must not be empty"),
            ));
        }
        Ok(payload)
    }

    fn with_default_shape(mut self, fallback: &NumericPayload) -> BuiltinResult<Self> {
        if !self.data.is_empty() {
            return Ok(self);
        }
        self.data = vec![0.0; fallback.data.len()];
        self.shape = fallback.shape.clone();
        self.repr = fallback.repr.clone();
        Ok(self)
    }

    pub(super) fn materialize(&self, data: Vec<f64>) -> BuiltinResult<Value> {
        self.repr.materialize(data)
    }
}

#[derive(Clone)]
enum NumericRepr {
    Scalar,
    Dense {
        shape: Vec<usize>,
        dtype: NumericDType,
    },
    Dlarray {
        inner: Box<NumericRepr>,
        format: Box<Value>,
        labels: Box<Value>,
    },
}

impl NumericRepr {
    fn materialize(&self, data: Vec<f64>) -> BuiltinResult<Value> {
        match self {
            Self::Scalar => match data.as_slice() {
                [value] => Ok(Value::Num(*value)),
                _ => {
                    let len = data.len();
                    Tensor::new(data, vec![1, len])
                        .map(Value::Tensor)
                        .map_err(|err| deep_learning_error("adamupdate", err))
                }
            },
            Self::Dense { shape, dtype } => Tensor::new_with_dtype(data, shape.clone(), *dtype)
                .map(Value::Tensor)
                .map_err(|err| deep_learning_error("adamupdate", err)),
            Self::Dlarray {
                inner,
                format,
                labels,
            } => Ok(object(
                "dlarray",
                vec![
                    ("Data", inner.materialize(data)?),
                    ("Format", format.as_ref().clone()),
                    ("Labels", labels.as_ref().clone()),
                ],
            )?),
        }
    }
}

fn ensure_finite(values: &[f64], label: &'static str) -> BuiltinResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must contain finite values"),
        ));
    }
    Ok(())
}

fn require_same_shape(
    expected: &NumericPayload,
    actual: &NumericPayload,
    label: &'static str,
) -> BuiltinResult<()> {
    if expected.shape != actual.shape || expected.data.len() != actual.data.len() {
        return Err(deep_learning_error(
            "adamupdate",
            format!(
                "adamupdate: {label} must match parameter shape {:?}, got {:?}",
                expected.shape, actual.shape
            ),
        ));
    }
    Ok(())
}

pub(super) fn positive_iteration(value: &Value) -> BuiltinResult<usize> {
    positive_usize(value, "adamupdate", "iteration").map_err(|_| {
        deep_learning_error(
            "adamupdate",
            "adamupdate: iteration must be a positive integer",
        )
    })
}

fn optional_finite_scalar(
    args: crate::call::FunctionArgumentView<'_>,
    index: usize,
    default: f64,
    label: &'static str,
) -> BuiltinResult<f64> {
    let Some(value) = args.get(index) else {
        return Ok(default);
    };
    match value {
        Value::Num(n) if n.is_finite() => Ok(*n),
        Value::Int(i) => Ok(i.to_f64()),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => {
            let n = crate::builtins::common::tensor::tensor_value_f64(tensor, 0);
            if n.is_finite() {
                Ok(n)
            } else {
                Err(deep_learning_error(
                    "adamupdate",
                    format!("adamupdate: {label} must be a finite numeric scalar, got {value:?}"),
                ))
            }
        }
        other => Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be a finite numeric scalar, got {other:?}"),
        )),
    }
}

pub(super) fn optional_positive_scalar(
    args: crate::call::FunctionArgumentView<'_>,
    index: usize,
    default: f64,
    label: &'static str,
) -> BuiltinResult<f64> {
    let value = optional_finite_scalar(args, index, default, label)?;
    if value <= 0.0 {
        return Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be positive"),
        ));
    }
    Ok(value)
}

pub(super) fn optional_unit_scalar(
    args: crate::call::FunctionArgumentView<'_>,
    index: usize,
    default: f64,
    label: &'static str,
) -> BuiltinResult<f64> {
    let value = optional_finite_scalar(args, index, default, label)?;
    if !(0.0..1.0).contains(&value) {
        return Err(deep_learning_error(
            "adamupdate",
            format!("adamupdate: {label} must be greater than or equal to 0 and less than 1"),
        ));
    }
    Ok(value)
}
