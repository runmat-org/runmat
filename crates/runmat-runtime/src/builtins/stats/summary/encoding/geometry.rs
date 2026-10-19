//! Dense encoding geometry and checked encoded input materialization.
use super::*;

pub(super) fn fill_feature_vector(
    values: &mut [f64],
    output_shape: &[usize],
    output_strides: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
    feature_axis: usize,
    input_idx: usize,
    class_count: usize,
    value: f64,
) {
    let mut coords = coords_for_linear(input_idx, input_shape, input_strides);
    coords.resize(output_shape.len(), 0);
    for class_idx in 0..class_count {
        coords[feature_axis] = class_idx;
        let out_idx = linear_for_coords(&coords, output_strides);
        values[out_idx] = value;
    }
}

pub(super) struct EncodedArray {
    pub(super) data: Vec<f64>,
    pub(super) shape: Vec<usize>,
}

pub(super) fn encoded_values(value: Value) -> BuiltinResult<EncodedArray> {
    match value {
        Value::Tensor(tensor) => {
            let shape = normalized_shape(tensor.shape());
            Ok(EncodedArray {
                data: tensor::tensor_into_values_f64(tensor),
                shape,
            })
        }
        Value::LogicalArray(array) => Ok(EncodedArray {
            data: array.data.into_iter().map(|value| value as f64).collect(),
            shape: normalized_shape(&array.shape),
        }),
        Value::Num(value) => Ok(EncodedArray {
            data: vec![value],
            shape: vec![1, 1],
        }),
        Value::Int(value) => {
            let encoded = match value.try_to_u64() {
                Some(0) => 0.0,
                Some(1) => 1.0,
                _ => {
                    return Err(invalid(
                        EncodingBuiltin::OnehotDecode,
                        "onehotdecode: encoded integer values must be zero or one",
                    ))
                }
            };
            Ok(EncodedArray {
                data: vec![encoded],
                shape: vec![1, 1],
            })
        }
        Value::Bool(value) => Ok(EncodedArray {
            data: vec![if value { 1.0 } else { 0.0 }],
            shape: vec![1, 1],
        }),
        other => Err(invalid(
            EncodingBuiltin::OnehotDecode,
            format!("onehotdecode: encoded data must be numeric or logical, got {other:?}"),
        )),
    }
}

pub(super) fn validate_encoded_range(values: &[f64]) -> BuiltinResult<()> {
    if values
        .iter()
        .any(|value| !value.is_nan() && !(0.0..=1.0).contains(value))
    {
        return Err(invalid(
            EncodingBuiltin::OnehotDecode,
            "onehotdecode: encoded values must be probabilities between 0 and 1",
        ));
    }
    Ok(())
}

pub(super) fn onehot_encoded_shape(
    name: EncodingBuiltin,
    input_shape: &[usize],
    feature_dim: usize,
    classes: usize,
) -> BuiltinResult<Vec<usize>> {
    let mut shape = normalized_shape(input_shape);
    if feature_dim == 0 {
        return Err(invalid(
            name,
            format!("{name}: featureDim must be positive"),
        ));
    }
    let feature_axis = feature_dim - 1;
    if feature_axis >= MAX_ENCODING_RANK || shape.len() > MAX_ENCODING_RANK {
        return Err(invalid(name, format!("{name}: featureDim is too large")));
    }
    if feature_axis >= shape.len() {
        shape.resize(feature_axis + 1, 1);
    }
    if shape[feature_axis] != 1 {
        return Err(invalid(
            name,
            format!("{name}: input dimension featureDim must be singleton"),
        ));
    }
    shape[feature_axis] = classes;
    Ok(shape)
}

pub(super) fn normalized_shape(shape: &[usize]) -> Vec<usize> {
    match shape.len() {
        0 => vec![1, 1],
        1 => vec![shape[0], 1],
        _ => shape.to_vec(),
    }
}

pub(super) fn element_count(owner: EncodingBuiltin, shape: &[usize]) -> BuiltinResult<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| invalid(owner, format!("{owner}: array is too large")))
    })
}

pub(super) fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut acc = 1usize;
    for dim in shape {
        strides.push(acc);
        acc = acc.saturating_mul(*dim);
    }
    strides
}

pub(super) fn coords_for_linear(mut idx: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let stride = strides[axis];
        coords[axis] = if stride == 0 { 0 } else { idx / stride };
        idx = if stride == 0 { idx } else { idx % stride };
    }
    coords
}

pub(super) fn linear_for_coords(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides)
        .map(|(coord, stride)| coord * stride)
        .sum()
}
