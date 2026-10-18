use super::*;

pub(super) struct LossMask {
    pub(super) data: Vec<f64>,
}

impl LossMask {
    pub(super) fn parse(value: &Value, predictions: &LossPayload) -> BuiltinResult<Self> {
        let (data, shape) = match value {
            Value::Bool(flag) => (vec![if *flag { 1.0 } else { 0.0 }], &[1, 1][..]),
            Value::LogicalArray(array) => (
                array.data.iter().map(|value| f64::from(*value)).collect(),
                array.shape.as_slice(),
            ),
            Value::Tensor(tensor) => (tensor::tensor_values_f64(tensor), tensor.shape()),
            other => {
                return Err(deep_learning_error(
                    "crossentropy",
                    format!("crossentropy: Mask must be a logical or binary numeric array, got {other:?}"),
                ));
            }
        };
        if shape != predictions.shape.as_slice() || data.len() != predictions.data.len() {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: Mask must have the same size as predictions",
            ));
        }
        if data
            .iter()
            .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
        {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: Mask must contain binary 0 or 1 values",
            ));
        }
        Ok(Self { data })
    }
}

pub(super) fn mask_values_for_layout(
    value: &Value,
    shape: &[usize],
    len: usize,
) -> BuiltinResult<Vec<f64>> {
    let (data, mask_shape) = match value {
        Value::Bool(flag) => (vec![if *flag { 1.0 } else { 0.0 }], &[1, 1][..]),
        Value::LogicalArray(array) => (
            array.data.iter().map(|value| f64::from(*value)).collect(),
            array.shape.as_slice(),
        ),
        Value::Tensor(tensor) => (tensor::tensor_values_f64(tensor), tensor.shape()),
        other => {
            return Err(deep_learning_error(
                "crossentropy",
                format!(
                    "crossentropy: Mask must be a logical or binary numeric array, got {other:?}"
                ),
            ));
        }
    };
    if mask_shape != shape || data.len() != len {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: Mask must have the same size as predictions",
        ));
    }
    if data
        .iter()
        .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
    {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: Mask must contain binary 0 or 1 values",
        ));
    }
    Ok(data)
}
