use runmat_builtins::{NumericDType, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::BuiltinResult;

use super::{
    any_type, deep_learning_error, gather_args, numeric_scalar, object, parse_name_values,
    scalar_text,
};

#[runtime_builtin(
    name = "crossentropy",
    category = "deep_learning",
    summary = "Compute cross-entropy loss for numeric prediction and target arrays.",
    keywords = "crossentropy,deep learning,loss",
    type_resolver(any_type),
    descriptor(crate::builtins::deep_learning::ARRAY_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::losses"
)]
pub(super) async fn crossentropy_builtin(
    predictions: Value,
    targets: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let predictions = LossPayload::parse(&predictions, "predictions")?;
    let targets = LossPayload::parse(&targets, "targets")?;
    if predictions.data.len() != targets.data.len() {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: predictions and targets must have the same number of elements",
        ));
    }
    if predictions.data.is_empty() {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: predictions and targets must not be empty",
        ));
    }
    let mut args = gather_args(rest).await?;
    let weights = if args
        .first()
        .is_some_and(|value| !is_crossentropy_option_name(value))
    {
        Some(LossWeights::parse(&args.remove(0), &predictions, None)?)
    } else {
        None
    };
    let mut options = CrossentropyOptions::from_args(args, &predictions)?;
    if let Some(weights) = weights {
        options.weights = Some(weights);
    }
    require_loss_payload_compatible(&predictions, &targets, options.data_format.as_deref())?;

    let mask = options.mask_values(&predictions)?;
    let weights = match options.weights {
        Some(ref weights) => weights.materialize(
            &predictions,
            options.data_format.as_deref(),
            options.weights_format.as_deref(),
        )?,
        None => vec![1.0; predictions.data.len()],
    };

    let losses = evaluate_crossentropy_terms(&predictions.data, &targets.data, &options.mode)?;
    let mut weighted_losses = Vec::with_capacity(losses.len());
    for ((loss, weight), mask) in losses.iter().zip(weights.iter()).zip(mask.iter()) {
        weighted_losses.push(loss * weight * mask);
    }

    match options.reduction {
        CrossentropyReduction::Sum => {
            let normalized = options.reduce_sum(&predictions, &weighted_losses, &mask)?;
            if !normalized.is_finite() {
                return Err(deep_learning_error(
                    "crossentropy",
                    "crossentropy: reduction produced a non-finite loss",
                ));
            }
            predictions.materialize_scalar(normalized)
        }
        CrossentropyReduction::None => predictions.materialize_array(weighted_losses, false),
    }
}

fn evaluate_crossentropy_terms(
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

fn is_crossentropy_option_name(value: &Value) -> bool {
    scalar_text(value, "crossentropy")
        .map(|name| {
            matches!(
                name.to_ascii_lowercase().as_str(),
                "classificationmode"
                    | "targetcategories"
                    | "mask"
                    | "reduction"
                    | "normalizationfactor"
                    | "dataformat"
                    | "weights"
                    | "weightsformat"
            )
        })
        .unwrap_or(false)
}

fn require_loss_payload_compatible(
    predictions: &LossPayload,
    targets: &LossPayload,
    data_format: Option<&str>,
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
    if let (Some(pred_format), Some(target_format)) = (effective_format, targets.format.clone()) {
        if !pred_format.eq_ignore_ascii_case(&target_format) {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: target dlarray format must match prediction format",
            ));
        }
    }
    Ok(())
}

#[derive(Clone)]
struct LossPayload {
    data: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
    format: Option<String>,
    dlarray: bool,
}

impl LossPayload {
    fn parse(value: &Value, label: &'static str) -> BuiltinResult<Self> {
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
                if !matches!(tensor.dtype, NumericDType::F64 | NumericDType::F32) {
                    return Err(deep_learning_error(
                        "crossentropy",
                        format!(
                            "crossentropy: {label} tensor must be double or single, got {}",
                            tensor.dtype.class_name()
                        ),
                    ));
                }
                if tensor.data.iter().any(|value| !value.is_finite()) {
                    return Err(deep_learning_error(
                        "crossentropy",
                        format!("crossentropy: {label} must contain finite numeric values"),
                    ));
                }
                Ok(Self {
                    data: tensor.data.clone(),
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                    format: None,
                    dlarray: false,
                })
            }
            Value::Object(object) if object.class_name == "dlarray" => {
                let data = object.properties.get("Data").ok_or_else(|| {
                    deep_learning_error("crossentropy", "crossentropy: dlarray is missing Data")
                })?;
                let mut payload = Self::parse(data, label)?;
                payload.format = object
                    .properties
                    .get("Format")
                    .and_then(|value| scalar_text(value, "crossentropy").ok())
                    .filter(|format| !format.is_empty());
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

    fn materialize_scalar(&self, loss: f64) -> BuiltinResult<Value> {
        if self.dlarray {
            Ok(object(
                "dlarray",
                vec![
                    ("Data", Value::Num(loss)),
                    ("Format", Value::String(String::new())),
                    ("Labels", Value::String(String::new())),
                ],
            ))
        } else {
            Ok(Value::Num(loss))
        }
    }

    fn materialize_array(&self, data: Vec<f64>, keep_format: bool) -> BuiltinResult<Value> {
        let tensor = Tensor::new_with_dtype(data, self.shape.clone(), self.dtype)
            .map(Value::Tensor)
            .map_err(|err| deep_learning_error("crossentropy", err))?;
        if self.dlarray {
            let format = if keep_format {
                self.format.clone().unwrap_or_default()
            } else {
                String::new()
            };
            Ok(object(
                "dlarray",
                vec![
                    ("Data", tensor),
                    ("Format", Value::String(format.clone())),
                    ("Labels", Value::String(format)),
                ],
            ))
        } else {
            Ok(tensor)
        }
    }

    fn format_or_default(&self, override_format: Option<&str>) -> Option<String> {
        override_format
            .map(str::to_string)
            .or_else(|| self.format.clone())
            .filter(|format| !format.is_empty())
    }
}

enum ClassificationMode {
    SingleLabel,
    MultiLabel,
}

impl ClassificationMode {
    fn parse(value: &Value, option: &str) -> BuiltinResult<Self> {
        let text = scalar_text(value, "crossentropy")?
            .to_ascii_lowercase()
            .replace(['_', ' '], "-");
        match text.as_str() {
            "single-label" | "singlelabel" | "exclusive" => Ok(Self::SingleLabel),
            "multi-label" | "multilabel" | "independent" => Ok(Self::MultiLabel),
            other => Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: unsupported {option} '{other}'"),
            )),
        }
    }
}

enum CrossentropyReduction {
    Sum,
    None,
}

impl CrossentropyReduction {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        match scalar_text(value, "crossentropy")?
            .to_ascii_lowercase()
            .as_str()
        {
            "sum" => Ok(Self::Sum),
            "none" => Ok(Self::None),
            other => Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: unsupported Reduction '{other}'"),
            )),
        }
    }
}

enum NormalizationFactor {
    AllElements,
    BatchSize,
    MaskIncluded,
    None,
    Scalar(f64),
}

impl NormalizationFactor {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        match value {
            Value::String(_) | Value::CharArray(_) | Value::StringArray(_) => {
                match scalar_text(value, "crossentropy")?
                    .to_ascii_lowercase()
                    .replace(['_', ' '], "-")
                    .as_str()
                {
                    "all-elements" | "all" => Ok(Self::AllElements),
                    "batch-size" | "batch" => Ok(Self::BatchSize),
                    "mask-included" | "mask" => Ok(Self::MaskIncluded),
                    "none" => Ok(Self::None),
                    other => Err(deep_learning_error(
                        "crossentropy",
                        format!("crossentropy: unsupported NormalizationFactor '{other}'"),
                    )),
                }
            }
            _ => {
                let scalar = numeric_scalar(value, "crossentropy", "NormalizationFactor")?;
                if scalar <= 0.0 {
                    return Err(deep_learning_error(
                        "crossentropy",
                        "crossentropy: NormalizationFactor must be positive",
                    ));
                }
                Ok(Self::Scalar(scalar))
            }
        }
    }
}

struct CrossentropyOptions {
    mode: ClassificationMode,
    reduction: CrossentropyReduction,
    normalization: NormalizationFactor,
    data_format: Option<String>,
    weights_format: Option<String>,
    weights: Option<LossWeights>,
    mask: Option<LossMask>,
}

impl CrossentropyOptions {
    fn from_args(args: Vec<Value>, predictions: &LossPayload) -> BuiltinResult<Self> {
        let mut parsed = parse_name_values(args, "crossentropy")?;
        let mode = parsed
            .remove("classificationmode")
            .or_else(|| parsed.remove("targetcategories"))
            .map(|value| ClassificationMode::parse(&value, "ClassificationMode"))
            .transpose()?
            .unwrap_or(ClassificationMode::SingleLabel);
        let reduction = parsed
            .remove("reduction")
            .map(|value| CrossentropyReduction::parse(&value))
            .transpose()?
            .unwrap_or(CrossentropyReduction::Sum);
        let data_format = parsed
            .remove("dataformat")
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?
            .filter(|format| !format.is_empty());
        let normalization = parsed
            .remove("normalizationfactor")
            .map(|value| NormalizationFactor::parse(&value))
            .transpose()?
            .unwrap_or_else(|| {
                if predictions.format.is_some() || data_format.is_some() {
                    NormalizationFactor::BatchSize
                } else {
                    NormalizationFactor::AllElements
                }
            });
        let weights_format = parsed
            .remove("weightsformat")
            .map(|value| scalar_text(&value, "crossentropy"))
            .transpose()?
            .filter(|format| !format.is_empty());
        let weights = parsed
            .remove("weights")
            .map(|value| LossWeights::parse(&value, predictions, weights_format.as_deref()))
            .transpose()?;
        let mask = parsed
            .remove("mask")
            .map(|value| LossMask::parse(&value, predictions))
            .transpose()?;
        if !parsed.is_empty() {
            let first = parsed.keys().next().cloned().unwrap_or_default();
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: unsupported option '{first}'"),
            ));
        }
        validate_format_for_shape(
            predictions
                .format_or_default(data_format.as_deref())
                .as_deref(),
            &predictions.shape,
            "DataFormat",
        )?;
        if let Some(weights_format) = weights_format.as_deref() {
            if let Some(weights) = &weights {
                validate_format_for_shape(Some(weights_format), &weights.shape, "WeightsFormat")?;
            }
        }
        Ok(Self {
            mode,
            reduction,
            normalization,
            data_format,
            weights_format,
            weights,
            mask,
        })
    }

    fn mask_values(&self, predictions: &LossPayload) -> BuiltinResult<Vec<f64>> {
        match &self.mask {
            Some(mask) => Ok(mask.data.clone()),
            None => Ok(vec![1.0; predictions.data.len()]),
        }
    }

    fn reduce_sum(
        &self,
        predictions: &LossPayload,
        weighted_losses: &[f64],
        mask: &[f64],
    ) -> BuiltinResult<f64> {
        match self.normalization {
            NormalizationFactor::AllElements => {
                Ok(weighted_losses.iter().sum::<f64>() / predictions.data.len() as f64)
            }
            NormalizationFactor::BatchSize => Ok(weighted_losses.iter().sum::<f64>()
                / batch_size(predictions, self.data_format.as_deref())?),
            NormalizationFactor::MaskIncluded => reduce_mask_included(
                predictions,
                weighted_losses,
                mask,
                self.data_format.as_deref(),
            ),
            NormalizationFactor::None => Ok(weighted_losses.iter().sum::<f64>()),
            NormalizationFactor::Scalar(value) => Ok(weighted_losses.iter().sum::<f64>() / value),
        }
    }
}

struct LossWeights {
    data: Vec<f64>,
    shape: Vec<usize>,
}

impl LossWeights {
    fn parse(
        value: &Value,
        _predictions: &LossPayload,
        _format: Option<&str>,
    ) -> BuiltinResult<Self> {
        match value {
            Value::Bool(flag) => Ok(Self {
                data: vec![if *flag { 1.0 } else { 0.0 }],
                shape: vec![1, 1],
            }),
            Value::LogicalArray(array) => Ok(Self {
                data: array.data.iter().map(|value| f64::from(*value)).collect(),
                shape: array.shape.clone(),
            }),
            _ => {
                let payload = LossPayload::parse(value, "weights")?;
                Ok(Self {
                    data: payload.data,
                    shape: payload.shape,
                })
            }
        }
    }

    fn materialize(
        &self,
        predictions: &LossPayload,
        data_format: Option<&str>,
        weights_format: Option<&str>,
    ) -> BuiltinResult<Vec<f64>> {
        if self
            .data
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: weights and masks must contain finite nonnegative values",
            ));
        }
        if self.data.len() == 1 {
            return Ok(vec![self.data[0]; predictions.data.len()]);
        }
        if let Some(weights_format) = weights_format {
            return self.materialize_by_format(predictions, data_format, weights_format);
        }
        if self.data.len() == predictions.data.len() {
            return Ok(self.data.clone());
        }
        if let Some(channel) = label_index(predictions, data_format, 'C') {
            return self.materialize_along_dimension(predictions, channel);
        }
        if let Some(batch) = label_index(predictions, data_format, 'B') {
            return self.materialize_along_dimension(predictions, batch);
        }
        Err(deep_learning_error(
            "crossentropy",
            "crossentropy: weights or mask size is not compatible with predictions",
        ))
    }

    fn materialize_along_dimension(
        &self,
        predictions: &LossPayload,
        dimension: usize,
    ) -> BuiltinResult<Vec<f64>> {
        if self.data.len() != predictions.shape[dimension] {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: weights size is not compatible with predictions",
            ));
        }
        let pred_strides = column_major_strides(&predictions.shape);
        let mut out = Vec::with_capacity(predictions.data.len());
        for idx in 0..predictions.data.len() {
            let coord = (idx / pred_strides[dimension]) % predictions.shape[dimension];
            out.push(self.data[coord]);
        }
        Ok(out)
    }

    fn materialize_by_format(
        &self,
        predictions: &LossPayload,
        data_format: Option<&str>,
        weights_format: &str,
    ) -> BuiltinResult<Vec<f64>> {
        validate_format_for_shape(Some(weights_format), &self.shape, "WeightsFormat")?;
        let data_format = predictions.format_or_default(data_format).ok_or_else(|| {
            deep_learning_error(
                "crossentropy",
                "crossentropy: DataFormat is required when WeightsFormat is supplied",
            )
        })?;
        let pred_labels = data_format.chars().collect::<Vec<_>>();
        let weight_labels = weights_format.chars().collect::<Vec<_>>();
        let pred_strides = column_major_strides(&predictions.shape);
        let weight_strides = column_major_strides(&self.shape);
        let mut out = Vec::with_capacity(predictions.data.len());
        for pred_idx in 0..predictions.data.len() {
            let mut weight_idx = 0usize;
            for (weight_dim, label) in weight_labels.iter().enumerate() {
                if label.eq_ignore_ascii_case(&'U') {
                    continue;
                }
                let pred_dim = pred_labels
                    .iter()
                    .position(|pred_label| pred_label.eq_ignore_ascii_case(label))
                    .ok_or_else(|| {
                        deep_learning_error(
                            "crossentropy",
                            format!(
                                "crossentropy: WeightsFormat label '{label}' is not present in DataFormat"
                            ),
                        )
                    })?;
                let pred_extent = predictions.shape[pred_dim];
                let weight_extent = self.shape[weight_dim];
                if weight_extent != 1 && weight_extent != pred_extent {
                    return Err(deep_learning_error(
                        "crossentropy",
                        "crossentropy: WeightsFormat dimensions are not compatible with DataFormat",
                    ));
                }
                if weight_extent > 1 {
                    let coord = (pred_idx / pred_strides[pred_dim]) % pred_extent;
                    weight_idx += coord * weight_strides[weight_dim];
                }
            }
            out.push(self.data[weight_idx]);
        }
        Ok(out)
    }
}

struct LossMask {
    data: Vec<f64>,
}

impl LossMask {
    fn parse(value: &Value, predictions: &LossPayload) -> BuiltinResult<Self> {
        let (data, shape) = match value {
            Value::Bool(flag) => (vec![if *flag { 1.0 } else { 0.0 }], vec![1, 1]),
            Value::LogicalArray(array) => (
                array.data.iter().map(|value| f64::from(*value)).collect(),
                array.shape.clone(),
            ),
            Value::Tensor(tensor) => (tensor.data.clone(), tensor.shape.clone()),
            other => {
                return Err(deep_learning_error(
                    "crossentropy",
                    format!("crossentropy: Mask must be a logical or binary numeric array, got {other:?}"),
                ));
            }
        };
        if shape != predictions.shape || data.len() != predictions.data.len() {
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

fn batch_size(predictions: &LossPayload, format: Option<&str>) -> BuiltinResult<f64> {
    let Some(format) = predictions.format_or_default(format) else {
        return Ok(predictions.data.len() as f64);
    };
    let Some(batch_dim) = format
        .chars()
        .position(|label| label.eq_ignore_ascii_case(&'B'))
    else {
        return Ok(1.0);
    };
    Ok(predictions.shape.get(batch_dim).copied().unwrap_or(1) as f64)
}

fn reduce_mask_included(
    predictions: &LossPayload,
    weighted_losses: &[f64],
    mask: &[f64],
    format: Option<&str>,
) -> BuiltinResult<f64> {
    let batch_dim = label_index(predictions, format, 'B');
    let batch_count = batch_dim
        .map(|dim| predictions.shape[dim])
        .unwrap_or(1)
        .max(1);
    let pred_strides = column_major_strides(&predictions.shape);
    let mut included = vec![0usize; batch_count];
    let mut totals = vec![0.0; batch_count];
    for idx in 0..weighted_losses.len() {
        if mask[idx] == 0.0 {
            continue;
        }
        let batch = batch_dim
            .map(|dim| (idx / pred_strides[dim]) % predictions.shape[dim])
            .unwrap_or(0);
        included[batch] += 1;
        totals[batch] += weighted_losses[idx];
    }
    if included.iter().all(|count| *count == 0) {
        return Err(deep_learning_error(
            "crossentropy",
            "crossentropy: Mask must include at least one element",
        ));
    }
    let mut normalized = 0.0;
    for (total, count) in totals.iter().zip(included.iter()) {
        if *count > 0 {
            normalized += total / *count as f64;
        }
    }
    Ok(normalized / batch_count as f64)
}

fn label_index(predictions: &LossPayload, format: Option<&str>, label: char) -> Option<usize> {
    predictions.format_or_default(format).and_then(|format| {
        format
            .chars()
            .position(|candidate| candidate.eq_ignore_ascii_case(&label))
    })
}

fn validate_format_for_shape(
    format: Option<&str>,
    shape: &[usize],
    label: &'static str,
) -> BuiltinResult<()> {
    let Some(format) = format else {
        return Ok(());
    };
    if format.chars().count() != shape.len() {
        return Err(deep_learning_error(
            "crossentropy",
            format!("crossentropy: {label} length must match array rank"),
        ));
    }
    for required_unique in ['C', 'B', 'T'] {
        if format
            .chars()
            .filter(|candidate| candidate.eq_ignore_ascii_case(&required_unique))
            .count()
            > 1
        {
            return Err(deep_learning_error(
                "crossentropy",
                format!("crossentropy: {label} cannot repeat '{required_unique}' labels"),
            ));
        }
    }
    if format.chars().any(|candidate| {
        !"SCBTU"
            .chars()
            .any(|allowed| allowed.eq_ignore_ascii_case(&candidate))
    }) {
        return Err(deep_learning_error(
            "crossentropy",
            format!("crossentropy: {label} contains unsupported dimension labels"),
        ));
    }
    Ok(())
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(*dim);
    }
    strides
}
