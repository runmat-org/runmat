use super::*;
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use runmat_value::ExactHostTensorAllocation;

#[path = "query/allocation.rs"]
mod allocation;
use allocation::admit_spline2_before_query;
pub(super) use allocation::admit_spline2_from_object;

pub(super) async fn evaluate_interpolant(
    spec: &InterpolantSpec,
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
    preceding_receipt: Option<&HostAllocationReceipt>,
) -> BuiltinResult<Value> {
    if spec.grid_vectors.is_empty() {
        return Err(invalid("cannot evaluate an empty griddedInterpolant"));
    }
    validate_grid_value_shape(
        &spec.grid_vectors,
        &spec.values,
        spec.method,
        spec.extrapolation,
    )?;
    ensure_query_integer_extensions(args.view())?;
    let grid_rank = spec.grid_vectors.len();
    let grid_size = spec
        .grid_vectors
        .iter()
        .map(Vec::len)
        .try_fold(1usize, |acc, len| acc.checked_mul(len))
        .ok_or_else(|| invalid("grid size exceeds platform limits"))?;
    let mut extra_shape = if spec.values.shape().len() > grid_rank {
        spec.values.shape()[grid_rank..].to_vec()
    } else {
        Vec::new()
    };
    while extra_shape.last() == Some(&1) {
        extra_shape.pop();
    }
    let extra_count = extra_shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, len| acc.checked_mul(len))
        .ok_or_else(|| invalid("value size exceeds platform limits"))?;
    let allocation = if preceding_receipt.is_none() {
        admit_spline2_before_query(
            spec,
            args.view(),
            grid_size,
            extra_count,
            extra_shape.len(),
            runtime,
        )?
    } else {
        None
    };
    let receipt = preceding_receipt.or(allocation.as_ref());
    if let (Some(receipt), Some(runtime)) = (receipt, runtime) {
        receipt
            .checkpoint(runtime)
            .map_err(|error| super::super::pp::host_allocation_error(BUILTIN_NAME, error))?;
    }
    let plan = parse_query_plan(&spec.grid_vectors, args)?;
    let mut out_shape = plan.shape.clone();
    out_shape.extend(extra_shape);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let values = tensor_utils::tensor_values_f64_cow(&spec.values);
    if let (Some(receipt), Some(runtime)) = (receipt, runtime) {
        receipt
            .checkpoint(runtime)
            .map_err(|error| super::super::pp::host_allocation_error(BUILTIN_NAME, error))?;
    }
    if is_piecewise_method(spec.method) {
        return evaluate_piecewise_interpolant(
            spec,
            values.as_ref(),
            &plan,
            grid_size,
            extra_count,
            out_shape,
            runtime,
            receipt,
        )
        .await;
    }
    let output_count = plan
        .len()
        .checked_mul(extra_count)
        .ok_or_else(|| invalid("interpolant output size exceeds platform limits"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(output_count)
        .map_err(|_| invalid("interpolant output exceeds available memory"))?;
    for series in 0..extra_count {
        let series_offset = series * grid_size;
        plan.for_each_point(|point| {
            out.push(interpolate_point(
                spec,
                values.as_ref(),
                point,
                series_offset,
            ))
        })?;
    }
    finish_interpolant_output(out, out_shape, spec.values.numeric_dtype())
}

pub(super) async fn evaluate_piecewise_interpolant(
    spec: &InterpolantSpec,
    values: &[f64],
    plan: &QueryPlan,
    grid_size: usize,
    extra_count: usize,
    out_shape: Vec<usize>,
    runtime: Option<&crate::context::RuntimeContext>,
    receipt: Option<&HostAllocationReceipt>,
) -> BuiltinResult<Value> {
    if spec.grid_vectors.len() == 2 {
        return evaluate_spline2_interpolant(
            spec,
            values,
            plan,
            grid_size,
            extra_count,
            out_shape,
            runtime,
            receipt,
        )
        .await;
    }
    let x = &spec.grid_vectors[0];
    let output_count = plan
        .len()
        .checked_mul(extra_count)
        .ok_or_else(|| invalid("interpolant output size exceeds platform limits"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(output_count)
        .map_err(|_| invalid("interpolant output exceeds available memory"))?;
    let mut work = runtime.map(crate::context::RuntimeContext::cpu_cooperation);
    for series in 0..extra_count {
        let series_offset = series * grid_size;
        let y = values[series_offset..series_offset + grid_size].to_vec();
        let series_data = NumericSeries {
            x: x.clone(),
            y,
            series: 1,
            component_shape: Vec::new(),
            sample_axis: super::super::pp::SampleAxis::First,
        };
        let pp = match spec.method {
            InterpMethod::Pchip => {
                build_pchip_pp(&series_data, BUILTIN_NAME).map_err(|err| internal(err.message()))?
            }
            InterpMethod::Spline => build_spline_pp(&series_data, BUILTIN_NAME, &mut work)
                .await
                .map_err(|err| {
                    if is_spline_execution_failure(&err) {
                        crate::builtins::common::map_control_flow_with_builtin(err, BUILTIN_NAME)
                    } else {
                        internal(err.message())
                    }
                })?,
            InterpMethod::Makima => build_makima_pp(&series_data, BUILTIN_NAME)
                .map_err(|err| internal(err.message()))?,
            _ => unreachable!("piecewise evaluator only handles higher-order methods"),
        };
        plan.for_each_point(|point| {
            out.push(evaluate_piecewise_scalar(
                spec,
                values,
                &pp,
                point[0],
                series_offset,
            ));
        })?;
    }
    finish_interpolant_output(out, out_shape, spec.values.numeric_dtype())
}

async fn evaluate_spline2_interpolant(
    spec: &InterpolantSpec,
    values: &[f64],
    plan: &QueryPlan,
    grid_size: usize,
    extra_count: usize,
    out_shape: Vec<usize>,
    runtime: Option<&crate::context::RuntimeContext>,
    receipt: Option<&HostAllocationReceipt>,
) -> BuiltinResult<Value> {
    let y = &spec.grid_vectors[0];
    let x = &spec.grid_vectors[1];
    let output_count = plan
        .len()
        .checked_mul(extra_count)
        .ok_or_else(|| invalid("interpolant output size exceeds platform limits"))?;
    let mut output = if let (Some(receipt), Some(runtime)) = (receipt, runtime) {
        if output_count == 1 && spec.values.numeric_dtype() == NumericDType::F64 {
            Spline2Output::Scalar(None)
        } else {
            let exact = receipt
                .with_output_allocation(runtime, |permit| {
                    ExactHostTensorAllocation::new(spec.values.numeric_dtype(), &out_shape, permit)
                })
                .map_err(|error| super::super::pp::host_allocation_error(BUILTIN_NAME, error))?;
            Spline2Output::Exact(exact)
        }
    } else {
        let mut out = Vec::new();
        out.try_reserve_exact(output_count)
            .map_err(|_| invalid("interpolant output exceeds available memory"))?;
        Spline2Output::Ordinary(out)
    };
    let mut budget = AdmittedSplineWork::new(runtime, receipt, BUILTIN_NAME);
    let exact_allocation = receipt.zip(runtime);
    let mut query_scratch = vec![0.0; 2];
    for series in 0..extra_count {
        let series_offset = series * grid_size;
        let samples = &values[series_offset..series_offset + grid_size];
        let spline = TensorProductSpline2D::prepare(
            x,
            y,
            |index| samples[index],
            &mut budget,
            BUILTIN_NAME,
            exact_allocation,
        )
        .await?;
        let mut pending = 0u64;
        for query in 0..plan.len() {
            let point = plan.point_at(query, &mut query_scratch);
            let outside = point_outside_grid(spec, point);
            let value = if outside && spec.extrapolation == ExtrapolationMethod::None {
                f64::NAN
            } else if outside && spec.extrapolation == ExtrapolationMethod::Nearest {
                interpolate_nearest(spec, values, point, series_offset)
            } else if outside && spec.extrapolation == ExtrapolationMethod::Linear {
                interpolate_linear(spec, values, point, series_offset)
            } else {
                spline.evaluate(x, y, point[1], point[0], |index| samples[index])
            };
            output.set(series * plan.len() + query, value);
            pending += 1;
            if pending == 4096 {
                budget.checkpoint(pending).await.map_err(|err| {
                    crate::builtins::common::map_control_flow_with_builtin(err, BUILTIN_NAME)
                })?;
                pending = 0;
            }
        }
        budget.checkpoint(pending).await.map_err(|err| {
            crate::builtins::common::map_control_flow_with_builtin(err, BUILTIN_NAME)
        })?;
    }
    let result = output.finish(out_shape, spec.values.numeric_dtype())?;
    if let (Some(receipt), Some(runtime)) = (receipt, runtime) {
        receipt
            .checkpoint(runtime)
            .map_err(|error| super::super::pp::host_allocation_error(BUILTIN_NAME, error))?;
    }
    Ok(result)
}

enum Spline2Output {
    Scalar(Option<f64>),
    Ordinary(Vec<f64>),
    Exact(ExactHostTensorAllocation),
}

impl Spline2Output {
    fn set(&mut self, index: usize, value: f64) {
        match self {
            Self::Scalar(slot) => *slot = Some(value),
            Self::Ordinary(values) => values.push(value),
            Self::Exact(values) => match values.as_f64_slice_mut() {
                Some(slice) => slice[index] = value,
                None => {
                    values.as_f32_slice_mut().expect("single spline output")[index] = value as f32;
                }
            },
        }
    }

    fn finish(self, shape: Vec<usize>, dtype: NumericDType) -> BuiltinResult<Value> {
        match self {
            Self::Scalar(Some(value)) => Ok(Value::Num(value)),
            Self::Scalar(None) => Err(internal("scalar spline output was not filled")),
            Self::Ordinary(values) => finish_interpolant_output(values, shape, dtype),
            Self::Exact(values) => values
                .into_tensor()
                .map(Value::Tensor)
                .map_err(|error| internal(format!("exact interpolant output: {error}"))),
        }
    }
}

pub(super) fn finish_interpolant_output(
    out: Vec<f64>,
    out_shape: Vec<usize>,
    dtype: NumericDType,
) -> BuiltinResult<Value> {
    if out.len() == 1 && dtype == NumericDType::F64 {
        Ok(Value::Num(out[0]))
    } else {
        let tensor = match dtype {
            NumericDType::F64 => Tensor::new(out, out_shape),
            NumericDType::F32 => Tensor::from_f32(
                out.into_iter().map(|value| value as f32).collect(),
                out_shape,
            ),
            _ => Err("interpolant output class must be single or double".to_string()),
        }
        .map_err(internal)?;
        Ok(Value::Tensor(tensor))
    }
}

pub(super) fn parse_query_plan(
    grid_vectors: &[Vec<f64>],
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<QueryPlan> {
    let ndim = grid_vectors.len();
    if args.len() == 1 {
        if let Value::Cell(cell) = args[0].clone() {
            let query_vectors = grid_vectors_from_cell(cell)?;
            if query_vectors.len() != ndim {
                return Err(invalid(
                    "query grid vector count must match interpolant dimension",
                ));
            }
            return full_query_grid(query_vectors);
        }
        if ndim == 1 {
            let (values, shape) = numeric_query_values(args.into_iter().next().unwrap(), "Xq")?;
            return Ok(QueryPlan {
                coords: QueryCoords::Explicit(
                    values.into_iter().map(|value| vec![value]).collect(),
                ),
                shape,
            });
        }
        let tensor = numeric_tensor(args.into_iter().next().unwrap(), "Xq")?;
        if tensor.shape().len() != 2 {
            return Err(invalid(
                "matrix query must be a two-dimensional m-by-n array",
            ));
        }
        if tensor.cols() != ndim {
            return Err(invalid(
                "matrix query must have one column per grid dimension",
            ));
        }
        let mut points = Vec::with_capacity(tensor.rows());
        let values = tensor_utils::tensor_values_f64_cow(&tensor);
        for row in 0..tensor.rows() {
            let mut point = Vec::with_capacity(ndim);
            for col in 0..ndim {
                point.push(values[row + col * tensor.rows()]);
            }
            points.push(point);
        }
        return Ok(QueryPlan {
            coords: QueryCoords::Explicit(points),
            shape: vec![tensor.rows(), 1],
        });
    }
    if args.len() != ndim {
        return Err(invalid(format!(
            "expected {ndim} query arguments, got {}",
            args.len()
        )));
    }
    let mut values_by_dim = Vec::with_capacity(ndim);
    let mut query_shape = None;
    for (dim, arg) in args.into_iter().enumerate() {
        let (values, shape) = numeric_query_values(arg, &format!("Xq{}", dim + 1))?;
        if let Some(expected) = &query_shape {
            if expected != &shape {
                return Err(invalid("query arrays must have matching shapes"));
            }
        } else {
            query_shape = Some(shape);
        }
        values_by_dim.push(values);
    }
    let shape = query_shape.unwrap_or_else(|| vec![0, 0]);
    let count = values_by_dim.first().map_or(0, Vec::len);
    let mut points = Vec::with_capacity(count);
    for idx in 0..count {
        points.push((0..ndim).map(|dim| values_by_dim[dim][idx]).collect());
    }
    Ok(QueryPlan {
        coords: QueryCoords::Explicit(points),
        shape,
    })
}

pub(super) fn numeric_query_values(
    value: Value,
    name: &str,
) -> BuiltinResult<(Vec<f64>, Vec<usize>)> {
    match value {
        Value::Num(x) => Ok((vec![x], vec![1, 1])),
        Value::Int(value) => {
            ensure_exact_integer_scalar(&value, name)?;
            Ok((vec![value.to_f64()], vec![1, 1]))
        }
        Value::Tensor(tensor) => {
            ensure_exact_integer_tensor(&tensor, name)?;
            let shape = tensor.shape().to_vec();
            Ok((tensor_utils::tensor_into_values_f64(tensor), shape))
        }
        other => Err(invalid(format!("{name} must be numeric, got {other:?}"))),
    }
}

pub(super) fn full_query_grid(query_vectors: Vec<Vec<f64>>) -> BuiltinResult<QueryPlan> {
    let shape = query_vectors.iter().map(Vec::len).collect::<Vec<_>>();
    let total = shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, len| acc.checked_mul(len))
        .ok_or_else(|| invalid("query grid size exceeds platform limits"))?;
    Ok(QueryPlan {
        coords: QueryCoords::FullGrid {
            vectors: query_vectors,
            total,
        },
        shape,
    })
}
