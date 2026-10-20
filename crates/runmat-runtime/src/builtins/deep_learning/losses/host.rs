use super::*;

pub(super) fn evaluate_crossentropy_terms(
    predictions: &[f64],
    targets: &[f64],
    mode: &ClassificationMode,
) -> BuiltinResult<Vec<f64>> {
    let eps = 1.0e-12;
    let mut out = Vec::with_capacity(predictions.len());
    for (prediction, target) in predictions.iter().zip(targets.iter()) {
        if !(0.0..=1.0).contains(target) {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: targets must be probabilities in the range [0, 1]",
            ));
        }
        let clipped = prediction.clamp(eps, 1.0 - eps);
        let loss = match mode {
            ClassificationMode::SingleLabel => -target * clipped.ln(),
            ClassificationMode::MultiLabel => {
                -target * clipped.ln() - (1.0 - target) * (1.0 - clipped).ln()
            }
        };
        if !loss.is_finite() {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: loss produced a non-finite value",
            ));
        }
        out.push(loss);
    }
    Ok(out)
}

pub(super) fn is_crossentropy_option_name(value: &Value) -> bool {
    scalar_text(value, "crossentropy")
        .map(|name| OptionKey::from_name(&name).is_some())
        .unwrap_or(false)
}

pub(super) fn require_loss_payload_compatible(
    predictions: &LossPayload,
    targets: &LossPayload,
    data_format: Option<&Layout>,
) -> BuiltinResult<()> {
    if predictions.shape != targets.shape {
        return Err(deep_learning_error(
            "crossentropy",
            format!(
                "crossentropy: targets must match prediction shape {:?}, got {:?}",
                predictions.shape, targets.shape
            ),
        ));
    }
    let effective_format = predictions.format_or_default(data_format);
    if let (Some(pred_format), Some(target_format)) = (effective_format, targets.format.as_ref()) {
        if pred_format != target_format {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: target dlarray format must match prediction format",
            ));
        }
    }
    Ok(())
}

#[derive(Clone)]
pub(super) struct LossPayload {
    pub(super) data: Vec<f64>,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: NumericDType,
    pub(super) format: Option<Layout>,
    pub(super) dlarray: bool,
}

impl LossPayload {
    pub(super) fn parse(value: &Value, label: &'static str) -> BuiltinResult<Self> {
        match value {
            Value::Num(n) if n.is_finite() => Ok(Self {
                data: vec![*n],
                shape: vec![1, 1],
                dtype: NumericDType::F64,
                format: None,
                dlarray: false,
            }),
            Value::Int(i) => Ok(Self {
                data: vec![i.to_f64()],
                shape: vec![1, 1],
                dtype: NumericDType::F64,
                format: None,
                dlarray: false,
            }),
            Value::Tensor(tensor) => {
                let output_dtype = match tensor.numeric_dtype() {
                    NumericDType::F64 => NumericDType::F64,
                    NumericDType::F32 => NumericDType::F32,
                    NumericDType::I8
                    | NumericDType::I16
                    | NumericDType::I32
                    | NumericDType::I64
                    | NumericDType::U8
                    | NumericDType::U16
                    | NumericDType::U32
                    | NumericDType::U64 => NumericDType::F64,
                };
                let data = tensor::tensor_values_f64(tensor);
                if data.iter().any(|value| !value.is_finite()) {
                    return Err(deep_learning_error(
                        "crossentropy",
                        format!("crossentropy: {label} must contain finite numeric values"),
                    ));
                }
                Ok(Self {
                    data,
                    shape: tensor.shape().to_vec(),
                    dtype: output_dtype,
                    format: None,
                    dlarray: false,
                })
            }
            Value::Object(object) if object.class_name().is(runmat_types::standard::DL_ARRAY) => {
                let data = object.properties().get("Data").ok_or_else(|| {
                    deep_learning_error("crossentropy", "crossentropy: dlarray is missing Data")
                })?;
                let mut payload = Self::parse(data, label)?;
                let raw_format = object
                    .properties()
                    .get("Format")
                    .and_then(|value| scalar_text(value, "crossentropy").ok())
                    .filter(|format| !format.is_empty());
                payload.format = parse_loss_format(raw_format.as_deref(), &payload.shape, "DataFormat")?;
                payload.dlarray = true;
                Ok(payload)
            }
            Value::GpuTensor(_) => Err(deep_learning_error(
                "crossentropy",
                "crossentropy: gpuArray inputs require provider kernels and are tracked by the GPU fast-path audit",
            )),
            other => Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: {label} must be a finite numeric array or dlarray, got {other:?}"),
            )),
        }
    }

    pub(super) fn materialize_scalar(&self, loss: f64) -> BuiltinResult<Value> {
        if self.dlarray {
            Ok(object(
                "dlarray",
                vec![
                    ("Data", Value::Num(loss)),
                    ("Format", Value::string(String::new())),
                    ("Labels", Value::string(String::new())),
                ],
            )?)
        } else {
            Ok(Value::Num(loss))
        }
    }

    pub(super) fn materialize_array(
        &self,
        data: Vec<f64>,
        keep_format: bool,
    ) -> BuiltinResult<Value> {
        let tensor = Tensor::new_with_dtype(data, self.shape.clone(), self.dtype)
            .map(Value::Tensor)
            .map_err(|err| deep_learning_error("crossentropy", err))?;
        if self.dlarray {
            let format = if keep_format {
                self.format.as_ref().map(Layout::labels).unwrap_or_default()
            } else {
                String::new()
            };
            Ok(object(
                "dlarray",
                vec![
                    ("Data", tensor),
                    ("Format", Value::string(format.clone())),
                    ("Labels", Value::string(format)),
                ],
            )?)
        } else {
            Ok(tensor)
        }
    }

    pub(super) fn format_or_default<'a>(
        &'a self,
        override_format: Option<&'a Layout>,
    ) -> Option<&'a Layout> {
        override_format.or(self.format.as_ref())
    }
}
