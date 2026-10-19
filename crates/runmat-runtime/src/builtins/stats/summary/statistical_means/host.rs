//! Column-major N-D host reductions. Admission, class selection, and numeric
//! kernels are separate from geometry so each public identity uses one route.
use super::data::HostData;
use super::data::NonnegativeWeights;
use super::numerics;
use super::numerics::ScaledMagnitude;
use super::options::{invalid, Axes, MissingMode, Options, ReductionKind};
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::BuiltinResult;
use runmat_value::{NumericDType, Tensor, Value};

#[derive(Clone)]
struct Geometry {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    input_strides: Vec<usize>,
    output_strides: Vec<usize>,
    reduced: Vec<bool>,
    axis: Option<usize>,
    output_len: usize,
}

impl Geometry {
    fn new(kind: ReductionKind, shape: &[usize], axes: &Axes) -> BuiltinResult<Self> {
        let mut input_shape = shape.to_vec();
        if input_shape.len() < 2 {
            input_shape.resize(2, 1);
        }
        let selected = match axes {
            Axes::Default => vec![input_shape
                .iter()
                .position(|extent| *extent != 1)
                .unwrap_or(0)],
            Axes::All => (0..input_shape.len()).collect(),
            Axes::Dims(dimensions) => dimensions
                .iter()
                .filter_map(|axis| {
                    let zero_based = axis - 1;
                    (zero_based < input_shape.len()).then_some(zero_based)
                })
                .collect(),
        };
        let rank = input_shape.len();
        let mut reduced = vec![false; rank];
        for axis in selected {
            reduced[axis] = true;
        }
        let mut output_shape = input_shape.clone();
        for (extent, selected) in output_shape.iter_mut().zip(&reduced) {
            if *selected {
                *extent = 1;
            }
        }
        let output_len = checked_len(&output_shape)
            .ok_or_else(|| invalid(kind, "output shape exceeds platform capacity"))?;
        let input_strides = strides(kind, &input_shape)?;
        let output_strides = strides(kind, &output_shape)?;
        let axis = reduced.iter().position(|selected| *selected);
        Ok(Self {
            input_shape,
            output_shape,
            input_strides,
            output_strides,
            reduced,
            axis,
            output_len,
        })
    }

    fn destination(&self, linear: usize) -> usize {
        (0..self.input_shape.len())
            .filter(|axis| !self.reduced[*axis])
            .map(|axis| {
                let coordinate = (linear / self.input_strides[axis]) % self.input_shape[axis];
                coordinate * self.output_strides[axis]
            })
            .sum()
    }

    fn axis_coordinate(&self, linear: usize) -> usize {
        let axis = self.axis.expect("weighted reduction has one axis");
        (linear / self.input_strides[axis]) % self.input_shape[axis]
    }
}

fn checked_len(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
}

fn strides(kind: ReductionKind, shape: &[usize]) -> BuiltinResult<Vec<usize>> {
    let mut result = vec![1usize; shape.len()];
    for axis in 1..shape.len() {
        result[axis] = result[axis - 1]
            .checked_mul(shape[axis - 1])
            .ok_or_else(|| invalid(kind, "array stride exceeds platform capacity"))?;
    }
    Ok(result)
}

pub(super) fn reduce(
    kind: ReductionKind,
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
    options: &Options,
) -> BuiltinResult<Value> {
    if !matches!(kind, ReductionKind::Geometric | ReductionKind::Harmonic) {
        return Err(invalid(kind, "non-RMS reduction called with RMS data"));
    }
    let geometry = Geometry::new(kind, &shape, &options.axes)?;
    if checked_len(&geometry.input_shape) != Some(values.len()) {
        return Err(invalid(kind, "input storage does not match its shape"));
    }
    let identity_axes = geometry
        .reduced
        .iter()
        .enumerate()
        .all(|(axis, selected)| !selected || geometry.input_shape[axis] == 1);
    if identity_axes {
        match kind {
            ReductionKind::Harmonic => {
                return output_value(kind, values, geometry.output_shape, dtype)
            }
            ReductionKind::Geometric => {
                let output = values
                    .into_iter()
                    .map(|value| if value < 0.0 { f64::NAN } else { value })
                    .collect();
                return output_value(kind, output, geometry.output_shape, dtype);
            }
            ReductionKind::RootMeanSquare | ReductionKind::RootMeanSquaredError => {}
        }
    }
    let mut buckets = vec![Vec::<f64>::new(); geometry.output_len];
    for (linear, value) in values.into_iter().enumerate() {
        buckets[geometry.destination(linear)].push(value);
    }
    let output = buckets
        .into_iter()
        .map(|mut values| {
            if options.missing == MissingMode::Omit {
                values.retain(|value| !value.is_nan());
            }
            match kind {
                ReductionKind::Geometric => numerics::geometric_mean(&values),
                ReductionKind::Harmonic => numerics::harmonic_mean(&values),
                ReductionKind::RootMeanSquare | ReductionKind::RootMeanSquaredError => {
                    unreachable!("RMS data has a separate scaled reduction")
                }
            }
        })
        .collect();
    output_value(kind, output, geometry.output_shape, dtype)
}

pub(super) fn reduce_rms(
    kind: ReductionKind,
    values: Vec<ScaledMagnitude>,
    shape: Vec<usize>,
    dtype: NumericDType,
    options: &Options,
) -> BuiltinResult<Value> {
    let geometry = Geometry::new(kind, &shape, &options.axes)?;
    if checked_len(&geometry.input_shape) != Some(values.len()) {
        return Err(invalid(kind, "input storage does not match its shape"));
    }
    let mut buckets = vec![Vec::<ScaledMagnitude>::new(); geometry.output_len];
    for (linear, value) in values.into_iter().enumerate() {
        buckets[geometry.destination(linear)].push(value);
    }
    let output = buckets
        .into_iter()
        .map(|mut values| {
            if options.missing == MissingMode::Omit {
                values.retain(|value| !value.is_nan());
            }
            numerics::root_mean_square_scaled(&values)
        })
        .collect();
    output_value(kind, output, geometry.output_shape, dtype)
}

pub(super) struct ResidualData {
    pub(super) values: Vec<ScaledMagnitude>,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: NumericDType,
    pub(super) forecast_shape: Vec<usize>,
    pub(super) actual_shape: Vec<usize>,
}

pub(super) fn residuals(
    left: HostData,
    right: HostData,
    kind: ReductionKind,
) -> BuiltinResult<ResidualData> {
    let dtype = if left.dtype == NumericDType::F32 && right.dtype == NumericDType::F32 {
        NumericDType::F32
    } else {
        NumericDType::F64
    };
    let plan = BroadcastPlan::new(&left.shape, &right.shape)
        .map_err(|source| invalid(kind, &source.to_string()))?;
    let mut values = vec![ScaledMagnitude::from_normalized(0.0, 0.0, 0.0); plan.len()];
    for (destination, l, r) in plan.iter() {
        values[destination] = left.points[l].scaled_difference(right.points[r]);
    }
    Ok(ResidualData {
        values,
        shape: plan.output_shape().to_vec(),
        dtype,
        forecast_shape: left.shape,
        actual_shape: right.shape,
    })
}

enum Weights {
    Scalar(f64),
    Elements(Vec<f64>),
    Axis(Vec<f64>),
}

pub(super) fn weighted_rmse(
    values: Vec<ScaledMagnitude>,
    shape: Vec<usize>,
    left_shape: &[usize],
    right_shape: &[usize],
    dtype: NumericDType,
    options: &Options,
    weights: NonnegativeWeights,
) -> BuiltinResult<Value> {
    let kind = ReductionKind::RootMeanSquaredError;
    let geometry = Geometry::new(kind, &shape, &options.axes)?;
    if geometry
        .reduced
        .iter()
        .filter(|selected| **selected)
        .count()
        != 1
    {
        return Err(invalid(
            kind,
            "weighted RMSE requires one operating dimension",
        ));
    }
    let (weight_values, weight_shape) = weights.into_parts();
    let weights = if weight_values.len() == 1 {
        Weights::Scalar(weight_values[0])
    } else if same_shape(&weight_shape, &shape)
        || same_shape(&weight_shape, left_shape)
        || same_shape(&weight_shape, right_shape)
    {
        let plan = BroadcastPlan::new(&weight_shape, &shape)
            .map_err(|source| invalid(kind, &source.to_string()))?;
        if !same_shape(plan.output_shape(), &shape) {
            return Err(invalid(
                kind,
                "weight shape does not match the broadcast result",
            ));
        }
        let mut expanded = vec![0.0; values.len()];
        for (destination, source, _) in plan.iter() {
            expanded[destination] = weight_values[source];
        }
        Weights::Elements(expanded)
    } else if weight_values.len() == geometry.input_shape[geometry.axis.unwrap()]
        && weight_shape.iter().filter(|extent| **extent > 1).count() == 1
    {
        Weights::Axis(weight_values)
    } else {
        return Err(invalid(
            kind,
            "weight shape must match an input, result, or operating axis",
        ));
    };
    let mut buckets = vec![Vec::<(ScaledMagnitude, f64)>::new(); geometry.output_len];
    for (linear, value) in values.into_iter().enumerate() {
        let weight = match &weights {
            Weights::Scalar(weight) => *weight,
            Weights::Elements(weights) => weights[linear],
            Weights::Axis(weights) => weights[geometry.axis_coordinate(linear)],
        };
        buckets[geometry.destination(linear)].push((value, weight));
    }
    let output = buckets
        .into_iter()
        .map(|bucket| numerics::weighted_root_mean_square_scaled(&bucket, options.missing))
        .collect();
    output_value(kind, output, geometry.output_shape, dtype)
}

fn same_shape(left: &[usize], right: &[usize]) -> bool {
    let rank = left.len().max(right.len());
    (0..rank)
        .all(|axis| left.get(axis).copied().unwrap_or(1) == right.get(axis).copied().unwrap_or(1))
}

fn output_value(
    kind: ReductionKind,
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    Tensor::new_with_dtype(values, shape, dtype)
        .map(crate::builtins::common::tensor::tensor_into_value)
        .map_err(|source| invalid(kind, &source))
}
