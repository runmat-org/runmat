use super::*;
use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor;
use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage};

pub(super) async fn elementwise_max(
    value: Value,
    args: ElementwiseArgs,
    demand: super::extrema_provider::ExtremaDemand,
) -> BuiltinResult<super::extrema_provider::ExtremaResult> {
    use super::extrema_provider::{ExtremaDemand, ExtremaKind, ExtremaResult};
    let ElementwiseArgs {
        other,
        nan_mode,
        comparison,
    } = args;
    let kind = ExtremaKind::Max;
    let resident = if comparison == ComparisonMethod::Auto && nan_mode == ReductionNaN::Include {
        match (&value, &other) {
            (Value::GpuTensor(left), Value::GpuTensor(right)) => {
                kind.pair_values(left, right).await?
            }
            (Value::GpuTensor(input), other) => match extract_scalar(other) {
                Some(scalar) => kind.scalar_values(input, other, scalar, false).await?,
                None => None,
            },
            (other, Value::GpuTensor(input)) => match extract_scalar(other) {
                Some(scalar) => kind.scalar_values(input, other, scalar, true).await?,
                None => None,
            },
            _ => None,
        }
    } else {
        None
    };
    if let Some(values) = resident {
        if demand == ExtremaDemand::ValuesOnly {
            return Ok(ExtremaResult::Values(Value::GpuTensor(values)));
        }
        // Generic <=/>= masks do not encode NaN or signed-zero extrema origins.
        // The existing typed host selector is the sole origin authority.
        let left = kind.gather_input(value).await?;
        let right = kind.gather_input(other).await?;
        let evaluated = elementwise_real_or_complex(left, right, nan_mode, comparison)?;
        return Ok(ExtremaResult::Pair {
            values: Value::GpuTensor(values),
            indices: evaluated.indices,
        });
    }
    let left = kind.gather_input(value).await?;
    let right = kind.gather_input(other).await?;
    let evaluated = elementwise_real_or_complex(left, right, nan_mode, comparison)?;
    Ok(ExtremaResult::from_pair(
        demand,
        evaluated.values,
        evaluated.indices,
    ))
}

fn extract_scalar(v: &Value) -> Option<f64> {
    match v {
        Value::Num(n) => Some(*n),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) && t.integer_storage().is_none() => {
            Some(tensor::tensor_value_f64(t, 0))
        }
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn elementwise_real_or_complex(
    lhs: Value,
    rhs: Value,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) -> BuiltinResult<MaxEvaluation> {
    let integer_comparison = match comparison {
        ComparisonMethod::Auto | ComparisonMethod::Real => {
            crate::builtins::math::reduction::integer_native::ExtremaComparison::Natural
        }
        ComparisonMethod::Abs => {
            crate::builtins::math::reduction::integer_native::ExtremaComparison::Absolute
        }
    };
    if let Some(eval) = crate::builtins::math::reduction::integer_native::elementwise_value_extrema(
        &lhs,
        &rhs,
        crate::builtins::math::reduction::integer_native::ExtremaDirection::Max,
        integer_comparison,
        nan_mode == ReductionNaN::Omit,
    )
    .map_err(|error| max_size_mismatch(format!("max: {error}")))?
    {
        return Ok(MaxEvaluation {
            values: eval.values,
            indices: eval.indices,
        });
    }
    if crate::builtins::math::reduction::integer_native::value_has_integer_storage(&lhs)
        || crate::builtins::math::reduction::integer_native::value_has_integer_storage(&rhs)
    {
        return Err(max_invalid_input(
            "max: integer pairwise inputs require the same integer class or a scalar double",
        ));
    }
    if let Some(eval) = scalar_elementwise_max(&lhs, &rhs, nan_mode, comparison) {
        return Ok(eval);
    }
    match (
        materialize_for_max("max", lhs)?,
        materialize_for_max("max", rhs)?,
    ) {
        (InputData::Complex(a), InputData::Complex(b)) => {
            elementwise_complex_max(a, b, nan_mode, comparison)
        }
        (InputData::Complex(a), InputData::Real(b)) => {
            let converted = promote_real_tensor_to_complex(b);
            elementwise_complex_max(a, converted, nan_mode, comparison)
        }
        (InputData::Real(a), InputData::Complex(b)) => {
            let converted = promote_real_tensor_to_complex(a);
            elementwise_complex_max(converted, b, nan_mode, comparison)
        }
        (InputData::Real(a), InputData::Real(b)) => {
            elementwise_real_max(a, b, nan_mode, comparison)
        }
    }
}

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Int(i) => Some(i.to_f64()),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if tensor::is_scalar_tensor(t) => Some(tensor::tensor_value_f64(t, 0)),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        _ => None,
    }
}

pub(super) fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(re, im) => Some((*re, *im)),
        Value::ComplexTensor(ct) if tensor::is_scalar_complex_tensor(ct) => {
            let value = tensor::complex_tensor_value_complex64(ct, 0);
            Some((value.re, value.im))
        }
        _ => None,
    }
}

fn scalar_elementwise_max(
    lhs: &Value,
    rhs: &Value,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) -> Option<MaxEvaluation> {
    if super::extrema_provider::needs_typed_scalar_selection(lhs)
        || super::extrema_provider::needs_typed_scalar_selection(rhs)
    {
        return None;
    }
    let left = scalar_complex_value(lhs).or_else(|| scalar_real_value(lhs).map(|v| (v, 0.0)))?;
    let right = scalar_complex_value(rhs).or_else(|| scalar_real_value(rhs).map(|v| (v, 0.0)))?;
    let (ar, ai) = left;
    let (br, bi) = right;
    if ai != 0.0 || bi != 0.0 {
        let (value, origin) = choose_complex_elementwise((ar, ai), (br, bi), nan_mode, comparison);
        return Some(MaxEvaluation {
            values: Value::Complex(value.0, value.1),
            indices: Value::Num(origin),
        });
    }
    let (value, origin) = choose_real_elementwise(ar, br, nan_mode, comparison);
    Some(MaxEvaluation {
        values: Value::Num(value),
        indices: Value::Num(origin),
    })
}

fn elementwise_real_max(
    lhs: Tensor,
    rhs: Tensor,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) -> BuiltinResult<MaxEvaluation> {
    let plan = BroadcastPlan::new(lhs.shape(), rhs.shape())
        .map_err(|err| max_size_mismatch(format!("max: {err}")))?;
    let mut indices = vec![0.0f64; plan.len()];
    let lhs_storage = lhs
        .into_numeric_storage()
        .map_err(|e| max_internal_error(format!("max: {e}")))?;
    let rhs_storage = rhs
        .into_numeric_storage()
        .map_err(|e| max_internal_error(format!("max: {e}")))?;

    macro_rules! select_same_class {
        ($left:expr, $right:expr, $variant:ident) => {{
            let mut values = Vec::with_capacity(plan.len());
            for (offset, index_a, index_b) in plan.iter() {
                let a = $left[index_a];
                let b = $right[index_b];
                let (_, origin) = choose_real_elementwise(a as f64, b as f64, nan_mode, comparison);
                values.push(if origin == 1.0 { a } else { b });
                indices[offset] = origin;
            }
            NumericStorage::$variant(values)
        }};
    }

    let values = match (lhs_storage, rhs_storage) {
        (NumericStorage::F64(left), NumericStorage::F64(right)) => {
            select_same_class!(left, right, F64)
        }
        (NumericStorage::F32(left), NumericStorage::F32(right)) => {
            select_same_class!(left, right, F32)
        }
        (NumericStorage::I8(left), NumericStorage::I8(right)) => {
            select_same_class!(left, right, I8)
        }
        (NumericStorage::I16(left), NumericStorage::I16(right)) => {
            select_same_class!(left, right, I16)
        }
        (NumericStorage::I32(left), NumericStorage::I32(right)) => {
            select_same_class!(left, right, I32)
        }
        (NumericStorage::I64(left), NumericStorage::I64(right)) => {
            select_same_class!(left, right, I64)
        }
        (NumericStorage::U8(left), NumericStorage::U8(right)) => {
            select_same_class!(left, right, U8)
        }
        (NumericStorage::U16(left), NumericStorage::U16(right)) => {
            select_same_class!(left, right, U16)
        }
        (NumericStorage::U32(left), NumericStorage::U32(right)) => {
            select_same_class!(left, right, U32)
        }
        (NumericStorage::U64(left), NumericStorage::U64(right)) => {
            select_same_class!(left, right, U64)
        }
        (left, right) => {
            let left = left.materialize_f64();
            let right = right.materialize_f64();
            let mut values = Vec::with_capacity(plan.len());
            for (offset, index_a, index_b) in plan.iter() {
                let (value, origin) =
                    choose_real_elementwise(left[index_a], right[index_b], nan_mode, comparison);
                values.push(value);
                indices[offset] = origin;
            }
            NumericStorage::F64(values)
        }
    };

    let value_tensor = Tensor::from_numeric_storage(values, plan.output_shape().to_vec())
        .map_err(|e| max_internal_error(format!("max: {e}")))?;
    let index_tensor = Tensor::new(indices, plan.output_shape().to_vec())
        .map_err(|e| max_internal_error(format!("max: {e}")))?;

    Ok(MaxEvaluation {
        values: tensor::tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn elementwise_complex_max(
    lhs: ComplexTensor,
    rhs: ComplexTensor,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) -> BuiltinResult<MaxEvaluation> {
    let plan = BroadcastPlan::new(lhs.shape(), rhs.shape())
        .map_err(|err| max_size_mismatch(format!("max: {err}")))?;
    let mut indices = vec![0.0f64; plan.len()];
    let lhs_storage = lhs.into_complex_storage();
    let rhs_storage = rhs.into_complex_storage();

    macro_rules! select_same_class {
        ($left:expr, $right:expr, $variant:ident) => {{
            let mut values = Vec::with_capacity(plan.len());
            for (offset, index_a, index_b) in plan.iter() {
                let a = $left[index_a];
                let b = $right[index_b];
                let (_, origin) = choose_complex_elementwise(
                    (f64::from(a.0), f64::from(a.1)),
                    (f64::from(b.0), f64::from(b.1)),
                    nan_mode,
                    comparison,
                );
                values.push(if origin == 1.0 { a } else { b });
                indices[offset] = origin;
            }
            ComplexStorage::$variant(values.into())
        }};
    }

    let values = match (lhs_storage, rhs_storage) {
        (ComplexStorage::F64(left), ComplexStorage::F64(right)) => {
            select_same_class!(left, right, F64)
        }
        (ComplexStorage::F32(left), ComplexStorage::F32(right)) => {
            select_same_class!(left, right, F32)
        }
        (left, right) => {
            let left = left.materialize_f64();
            let right = right.materialize_f64();
            let mut values = Vec::with_capacity(plan.len());
            for (offset, index_a, index_b) in plan.iter() {
                let (value, origin) =
                    choose_complex_elementwise(left[index_a], right[index_b], nan_mode, comparison);
                values.push(value);
                indices[offset] = origin;
            }
            ComplexStorage::F64(values.into())
        }
    };

    let value_tensor = ComplexTensor::from_complex_storage(values, plan.output_shape().to_vec())
        .map_err(|e| max_internal_error(format!("max: {e}")))?;
    let index_tensor = Tensor::new(indices, plan.output_shape().to_vec())
        .map_err(|e| max_internal_error(format!("max: {e}")))?;

    Ok(MaxEvaluation {
        values: complex_tensor_into_value(value_tensor),
        indices: tensor::tensor_into_value(index_tensor),
    })
}

fn promote_real_tensor_to_complex(tensor: Tensor) -> ComplexTensor {
    let shape = tensor.shape().to_vec();
    let storage = match tensor
        .into_numeric_storage()
        .expect("real extrema input has numeric storage")
    {
        NumericStorage::F32(values) => {
            ComplexStorage::F32(values.into_iter().map(|re| (re, 0.0)).collect())
        }
        storage => ComplexStorage::F64(
            storage
                .materialize_f64()
                .into_iter()
                .map(|re| (re, 0.0))
                .collect(),
        ),
    };
    ComplexTensor::from_complex_storage(storage, shape)
        .expect("real tensor shape remains valid after promotion")
}

fn choose_real_elementwise(
    a: f64,
    b: f64,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) -> (f64, f64) {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => (f64::NAN, 1.0),
        (true, false) if nan_mode == ReductionNaN::Omit => (b, 2.0),
        (false, true) if nan_mode == ReductionNaN::Omit => (a, 1.0),
        (true, false) => (f64::NAN, 1.0),
        (false, true) => (f64::NAN, 2.0),
        (false, false) => {
            if should_replace_real(a, b, comparison) {
                (b, 2.0)
            } else {
                (a, 1.0)
            }
        }
    }
}

fn choose_complex_elementwise(
    a: (f64, f64),
    b: (f64, f64),
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
) -> ((f64, f64), f64) {
    let a_nan = a.0.is_nan() || a.1.is_nan();
    let b_nan = b.0.is_nan() || b.1.is_nan();
    match (a_nan, b_nan) {
        (true, true) => ((f64::NAN, f64::NAN), 1.0),
        (true, false) if nan_mode == ReductionNaN::Omit => (b, 2.0),
        (false, true) if nan_mode == ReductionNaN::Omit => (a, 1.0),
        (true, false) => ((f64::NAN, f64::NAN), 1.0),
        (false, true) => ((f64::NAN, f64::NAN), 2.0),
        (false, false) => {
            if should_replace_complex(a, b, comparison) {
                (b, 2.0)
            } else {
                (a, 1.0)
            }
        }
    }
}
