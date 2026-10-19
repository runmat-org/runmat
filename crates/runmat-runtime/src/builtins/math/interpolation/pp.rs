use runmat_value::{StructValue, Tensor, Value};

use crate::builtins::common::random_args::keyword_of;
use crate::builtins::common::tensor;
use crate::{build_runtime_error, dispatcher, BuiltinResult, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpMethod {
    Linear,
    Nearest,
    Spline,
    Pchip,
    /// Uses PCHIP coefficients, but retains the spelled method's NaN default.
    Cubic,
    Makima,
}

#[derive(Clone, Debug)]
pub enum Extrapolation {
    Nan,
    Extrapolate,
    Value(f64),
}

impl Default for Extrapolation {
    fn default() -> Self {
        Self::Nan
    }
}

#[derive(Clone, Debug)]
pub struct NumericSeries {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub series: usize,
    /// Shape of the non-sample axes. They trail samples for interp1 and
    /// precede samples for pchip/spline's final-axis array form.
    pub component_shape: Vec<usize>,
    pub sample_axis: SampleAxis,
}

impl NumericSeries {
    fn validate_layout(&self, name: &'static str) -> BuiltinResult<()> {
        validate_breaks(&self.x, name)?;
        let components = PiecewisePolynomial::component_dim(&self.component_shape, name)?;
        if self.series == 0 || components != self.series {
            return Err(interp_error(
                name,
                format!("{name}: invalid sample dimensions"),
            ));
        }
        if self.x.len().checked_mul(self.series) != Some(self.y.len()) {
            return Err(interp_error(name, format!("{name}: invalid sample values")));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointSlopes {
    pub first: f64,
    pub last: f64,
}

#[derive(Clone, Debug)]
pub struct SplineSamples {
    pub series: NumericSeries,
    pub endpoint_slopes: Option<Vec<EndpointSlopes>>,
}

/// Where samples lie in a nonscalar input. Interp1 consumes columns, while
/// pchip and spline consume the final axis of an array-valued function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleAxis {
    First,
    Last,
}

#[derive(Clone, Debug)]
pub struct QueryPoints {
    pub values: Vec<f64>,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct PiecewisePolynomial {
    breaks: Vec<f64>,
    coefs: Vec<f64>,
    pieces: usize,
    order: usize,
    dim: usize,
    /// Component dimensions before query dimensions in public pp evaluation.
    /// The scalar `dim` remains the checked product for coefficient indexing.
    component_shape: Vec<usize>,
}

impl PiecewisePolynomial {
    fn component_dim(component_shape: &[usize], name: &'static str) -> BuiltinResult<usize> {
        component_shape
            .iter()
            .try_fold(1usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| interp_error(name, format!("{name}: malformed pp dimensions")))
    }

    fn from_parts(
        breaks: Vec<f64>,
        coefs: Vec<f64>,
        pieces: usize,
        order: usize,
        dim: usize,
        component_shape: Vec<usize>,
        name: &'static str,
    ) -> BuiltinResult<Self> {
        validate_breaks(&breaks, name)?;
        let value = Self {
            breaks,
            coefs,
            pieces,
            order,
            dim,
            component_shape,
        };
        value.validate_layout(name)?;
        Ok(value)
    }

    fn validate_layout(&self, name: &'static str) -> BuiltinResult<()> {
        let expected_dim = Self::component_dim(&self.component_shape, name)?;
        if self.pieces.checked_add(1) != Some(self.breaks.len()) {
            return Err(interp_error(name, format!("{name}: malformed pp breaks")));
        }
        if self.order == 0 || self.dim == 0 || self.dim != expected_dim {
            return Err(interp_error(
                name,
                format!("{name}: malformed pp dimensions"),
            ));
        }
        let count = checked_coefficient_len(self.pieces, self.dim, self.order, name)?;
        if self.coefs.len() != count {
            return Err(interp_error(
                name,
                format!("{name}: malformed pp coefficients"),
            ));
        }
        Ok(())
    }
}

fn checked_coefficient_len(
    pieces: usize,
    dim: usize,
    order: usize,
    name: &'static str,
) -> BuiltinResult<usize> {
    pieces
        .checked_mul(dim)
        .and_then(|rows| rows.checked_mul(order))
        .ok_or_else(|| interp_error(name, format!("{name}: coefficient size overflow")))
}

fn zeroed_buffer(len: usize, name: &'static str) -> BuiltinResult<Vec<f64>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len).map_err(|_| {
        interp_error(
            name,
            format!("{name}: insufficient memory for interpolation data"),
        )
    })?;
    data.resize(len, 0.0);
    Ok(data)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpOutputLayout {
    /// Public ppval/pchip/spline place function-value axes before query axes.
    ComponentLeading,
    /// Interp1 retains its existing query-leading matrix/array presentation.
    QueryLeading,
}

pub fn interp_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(name).build()
}

pub(crate) fn host_allocation_error(
    name: &'static str,
    source: crate::context::host_allocation::HostAllocationError,
) -> RuntimeError {
    use crate::context::host_allocation::HostAllocationError;
    let mut error = match source {
        HostAllocationError::Cancelled => {
            crate::execution::ExecutionServiceError::Cancelled.into_runtime_error()
        }
        HostAllocationError::StaleAuthority | HostAllocationError::MissingManagedOwner => {
            crate::execution::ExecutionServiceError::Infrastructure(
                "managed host allocation authority changed".into(),
            )
            .into_runtime_error()
        }
        _ => build_runtime_error(format!("{name}: host allocation failed: {source}"))
            .with_builtin(name)
            .with_identifier("RunMat:Interpolation:HostAllocation")
            .build(),
    };
    error.source = Some(Box::new(source));
    crate::builtins::common::map_control_flow_with_builtin(error, name)
}

async fn materialize_if_resident(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => dispatcher::gather_if_needed_async(&value).await,
        host => Ok(host),
    }
}

#[path = "pp/series.rs"]
mod series;
pub use series::*;

#[path = "pp/tensor2_spline.rs"]
mod tensor2_spline;
pub(crate) use coefficients::AdmittedSplineWork;
pub use tensor2_spline::TensorProductSpline2D;
#[path = "pp/tensor2_allocation.rs"]
mod tensor2_allocation;
pub(crate) use tensor2_allocation::tensor_spline_allocation_plan;

#[path = "pp/evaluation.rs"]
mod evaluation;
pub(super) use evaluation::{eval_pp_scalar, interval_index, out_of_range_value};
pub use evaluation::{evaluate_linear_or_nearest, evaluate_pp};
pub(crate) async fn build_spline_pp(
    series: &NumericSeries,
    name: &'static str,
    work: &mut impl SplineCooperation,
) -> BuiltinResult<PiecewisePolynomial> {
    build_spline_pp_with_slopes(series, None, name, work).await
}

pub(crate) async fn build_spline_pp_with_slopes(
    series: &NumericSeries,
    endpoint_slopes: Option<&[EndpointSlopes]>,
    name: &'static str,
    work: &mut impl SplineCooperation,
) -> BuiltinResult<PiecewisePolynomial> {
    series.validate_layout(name)?;
    let n = series.x.len();
    if endpoint_slopes.is_some_and(|slopes| slopes.len() != series.series) {
        return Err(interp_error(
            name,
            format!("{name}: endpoint slope shape mismatch"),
        ));
    }
    let pieces = n - 1;
    let order = 4;
    let rows = pieces
        .checked_mul(series.series)
        .ok_or_else(|| interp_error(name, format!("{name}: coefficient size overflow")))?;
    let mut coefs = zeroed_buffer(
        checked_coefficient_len(pieces, series.series, order, name)?,
        name,
    )?;
    for s in 0..series.series {
        let y = &series.y[s * n..(s + 1) * n];
        let local =
            spline_coefs_for_series(&series.x, y, endpoint_slopes.map(|slopes| slopes[s]), work)
                .await?;
        for piece in 0..pieces {
            let row = piece * series.series + s;
            for col in 0..order {
                coefs[row + col * rows] = local[piece][col];
            }
        }
    }
    validate_finite_coefs(&coefs, name)?;
    PiecewisePolynomial::from_parts(
        series.x.clone(),
        coefs,
        pieces,
        order,
        series.series,
        series.component_shape.clone(),
        name,
    )
}

pub fn build_pchip_pp(
    series: &NumericSeries,
    name: &'static str,
) -> BuiltinResult<PiecewisePolynomial> {
    build_local_cubic_pp(series, name, |x, y, _| Ok(pchip_coefs_for_series(x, y)))
}

pub fn build_makima_pp(
    series: &NumericSeries,
    name: &'static str,
) -> BuiltinResult<PiecewisePolynomial> {
    build_local_cubic_pp(series, name, makima_coefs_for_series)
}

type LocalCubicCoefficientBuilder =
    fn(&[f64], &[f64], &'static str) -> BuiltinResult<Vec<[f64; 4]>>;

fn build_local_cubic_pp(
    series: &NumericSeries,
    name: &'static str,
    coefficients: LocalCubicCoefficientBuilder,
) -> BuiltinResult<PiecewisePolynomial> {
    series.validate_layout(name)?;
    let n = series.x.len();
    let pieces = n - 1;
    let order = 4;
    let rows = pieces
        .checked_mul(series.series)
        .ok_or_else(|| interp_error(name, format!("{name}: coefficient size overflow")))?;
    let mut coefs = zeroed_buffer(
        checked_coefficient_len(pieces, series.series, order, name)?,
        name,
    )?;
    for s in 0..series.series {
        let y = &series.y[s * n..(s + 1) * n];
        let local = coefficients(&series.x, y, name)?;
        for piece in 0..pieces {
            let row = piece * series.series + s;
            for col in 0..order {
                coefs[row + col * rows] = local[piece][col];
            }
        }
    }
    validate_finite_coefs(&coefs, name)?;
    PiecewisePolynomial::from_parts(
        series.x.clone(),
        coefs,
        pieces,
        order,
        series.series,
        series.component_shape.clone(),
        name,
    )
}

#[path = "pp/format.rs"]
mod format;
pub use format::{pp_from_value, pp_to_value};

fn canonical_shape(shape: &[usize], len: usize) -> Vec<usize> {
    tensor::default_shape_for(shape, len)
}

fn first_non_singleton_len(tensor: &Tensor) -> Option<usize> {
    canonical_shape(tensor.shape(), tensor_len(tensor))
        .into_iter()
        .find(|&dim| dim > 1)
}

fn tensor_len(tensor: &Tensor) -> usize {
    tensor::tensor_element_len(tensor)
}

pub(super) fn is_vector_shape(shape: &[usize]) -> bool {
    match shape {
        [] => true,
        [_] => true,
        [rows, cols] => *rows == 1 || *cols == 1,
        dims => dims.iter().filter(|&&dim| dim > 1).count() <= 1,
    }
}

#[path = "pp/coefficients.rs"]
mod coefficients;
pub(crate) use coefficients::{
    is_execution_failure as is_spline_execution_failure, SplineCooperation,
};
use coefficients::{
    makima_coefs_for_series, pchip_coefs_for_series, spline_coefs_for_series, validate_finite_coefs,
};

#[cfg(test)]
#[path = "pp/tests.rs"]
mod tests;
