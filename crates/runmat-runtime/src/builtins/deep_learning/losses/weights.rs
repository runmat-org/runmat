use super::*;

pub(super) struct LossWeights {
    pub(super) data: Vec<f64>,
    pub(super) shape: Vec<usize>,
}

impl LossWeights {
    pub(super) fn parse(value: &Value, label: &'static str) -> BuiltinResult<Self> {
        match value {
            Value::Bool(flag) => Ok(Self {
                data: vec![if *flag { 1.0 } else { 0.0 }],
                shape: vec![1, 1],
            }),
            Value::LogicalArray(array) => Ok(Self {
                data: array.data.iter().map(|value| f64::from(*value)).collect(),
                shape: array.shape.to_vec(),
            }),
            _ => {
                let payload = LossPayload::parse(value, label)?;
                Ok(Self {
                    data: payload.data,
                    shape: payload.shape,
                })
            }
        }
    }

    pub(super) fn materialize(
        &self,
        predictions: &LossPayload,
        data_format: Option<&Layout>,
        weights_format: Option<&Layout>,
    ) -> BuiltinResult<Vec<f64>> {
        let format = predictions.format_or_default(data_format);
        self.materialize_for_layout(
            &predictions.shape,
            predictions.data.len(),
            format,
            weights_format,
        )
    }

    pub(super) fn materialize_for_layout(
        &self,
        prediction_shape: &[usize],
        prediction_len: usize,
        data_format: Option<&Layout>,
        weights_format: Option<&Layout>,
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
            return Ok(vec![self.data[0]; prediction_len]);
        }
        if let Some(weights_format) = weights_format {
            return self.materialize_by_format(
                prediction_shape,
                prediction_len,
                data_format,
                weights_format,
            );
        }
        if self.data.len() == prediction_len {
            return Ok(self.data.clone());
        }
        if let Some(channel) = data_format.and_then(|layout| layout.index(Axis::Channel)) {
            return self.materialize_along_dimension(prediction_shape, prediction_len, channel);
        }
        if let Some(batch) = data_format.and_then(|layout| layout.index(Axis::Batch)) {
            return self.materialize_along_dimension(prediction_shape, prediction_len, batch);
        }
        Err(deep_learning_error(
            "crossentropy",
            "crossentropy: weights or mask size is not compatible with predictions",
        ))
    }

    fn materialize_along_dimension(
        &self,
        prediction_shape: &[usize],
        prediction_len: usize,
        dimension: usize,
    ) -> BuiltinResult<Vec<f64>> {
        if self.data.len() != prediction_shape[dimension] {
            return Err(deep_learning_error(
                "crossentropy",
                "crossentropy: weights size is not compatible with predictions",
            ));
        }
        let pred_strides = column_major_strides(prediction_shape);
        let mut out = Vec::with_capacity(prediction_len);
        for idx in 0..prediction_len {
            let coord = (idx / pred_strides[dimension]) % prediction_shape[dimension];
            out.push(self.data[coord]);
        }
        Ok(out)
    }

    fn materialize_by_format(
        &self,
        prediction_shape: &[usize],
        prediction_len: usize,
        data_format: Option<&Layout>,
        weights_format: &Layout,
    ) -> BuiltinResult<Vec<f64>> {
        let data_format = data_format.ok_or_else(|| {
            deep_learning_error(
                "crossentropy",
                "crossentropy: DataFormat is required when WeightsFormat is supplied",
            )
        })?;
        let pred_labels = data_format.axes();
        let weight_labels = weights_format.axes();
        let mut varying_axes = Vec::with_capacity(weight_labels.len());
        for (weight_dim, label) in weight_labels.iter().enumerate() {
            if *label == Axis::Unspecified {
                continue;
            }
            let occurrence = weight_labels[..weight_dim]
                .iter()
                .filter(|candidate| *candidate == label)
                .count();
            let pred_dim = pred_labels
                .iter()
                .enumerate()
                .filter(|(_, pred_label)| *pred_label == label)
                .nth(occurrence)
                .map(|(index, _)| index)
                .ok_or_else(|| {
                    deep_learning_error(
                        "crossentropy",
                        format!(
                            "crossentropy: WeightsFormat label '{}' is not present in DataFormat",
                            label.label()
                        ),
                    )
                })?;
            let weight_extent = self.shape[weight_dim];
            if weight_extent != 1 && weight_extent != prediction_shape[pred_dim] {
                return Err(deep_learning_error(
                    "crossentropy",
                    "crossentropy: WeightsFormat dimensions are not compatible with DataFormat",
                ));
            }
            if weight_extent > 1 {
                varying_axes.push((pred_dim, weight_dim));
            }
        }
        let pred_strides = column_major_strides(prediction_shape);
        let weight_strides = column_major_strides(&self.shape);
        let mut out = Vec::with_capacity(prediction_len);
        for pred_idx in 0..prediction_len {
            let mut weight_idx = 0usize;
            for &(pred_dim, weight_dim) in &varying_axes {
                let coord = (pred_idx / pred_strides[pred_dim]) % prediction_shape[pred_dim];
                weight_idx += coord * weight_strides[weight_dim];
            }
            out.push(self.data[weight_idx]);
        }
        Ok(out)
    }
}
