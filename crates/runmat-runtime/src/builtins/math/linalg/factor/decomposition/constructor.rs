use super::*;

pub(super) async fn parse_constructor(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<DecompositionSpec> {
    if args.is_empty() {
        return Err(invalid("decomposition: expected coefficient matrix"));
    }
    let mut host_args = gather_values(args, runtime).await?;
    let raw_matrix = host_args
        .pop_front()
        .ok_or_else(|| invalid("decomposition: expected coefficient matrix"))?;
    let selected_runtime = runtime.ok_or_else(|| {
        invalid("decomposition constructor requires its selected runtime context")
    })?;
    let (matrix, is_sparse) = native::normalize_selected(raw_matrix, selected_runtime)?;
    let mut decomp_type = DecompositionType::Auto;
    let mut triangular_flag: Option<TriangularFlag> = None;
    let mut option_start = 0usize;

    if let Some(next) = host_args.first() {
        if let Some(text) = text_from_value(next)? {
            if is_decomposition_type(&text) {
                decomp_type = parse_decomposition_type(&text)?;
                option_start = 1;
                if let Some(candidate) = host_args.get(1) {
                    if let Some(flag) = text_from_value(candidate)?
                        .and_then(|value| parse_triangular_flag(&value).ok())
                    {
                        triangular_flag = Some(flag);
                        option_start = 2;
                    }
                }
            } else if is_decomposition_option(&text) {
                option_start = 0;
            } else {
                return Err(parse_decomposition_type(&text).unwrap_err());
            }
        }
    }

    let options = host_args
        .view()
        .range(option_start..)
        .expect("bounded decomposition option range");
    let (check_condition, rank_tolerance) = parse_options(options)?;
    initialize::initialize(
        DecompositionSpec {
            matrix,
            decomp_type,
            check_condition,
            rank_tolerance,
            is_conjugate_transposed: false,
            is_sparse,
            scale: (1.0, 0.0),
            factor_data: None,
            triangle: triangular_flag,
            source_is_real: None,
        },
        selected_runtime,
    )
    .await
}

pub(super) async fn gather_values(
    args: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    let mut resident = false;
    for value in &args {
        crate::dispatcher::try_visit_gathered_gpu_handles(value, &mut |_| resident = true)?;
    }
    if !resident {
        return Ok(args);
    }
    args.try_map_preserving_roots_async(args.len(), runtime, async |index| {
        crate::dispatcher::gather_if_needed_async(&args[index])
            .await
            .map_err(|err| {
                build_runtime_error(err.message())
                    .with_builtin(NAME)
                    .with_source(err)
                    .build()
            })
    })
    .await
}

pub(super) fn normalize_matrix(value: Value) -> BuiltinResult<(Value, bool)> {
    match value {
        Value::Tensor(tensor) => {
            validate_matrix_shape(tensor.shape())?;
            Ok((Value::Tensor(tensor), false))
        }
        Value::ComplexTensor(tensor) => {
            validate_matrix_shape(tensor.shape())?;
            Ok((Value::ComplexTensor(tensor), false))
        }
        Value::LogicalArray(array) => {
            validate_matrix_shape(&array.shape)?;
            Ok((Value::LogicalArray(array), false))
        }
        Value::SparseTensor(matrix) => Ok((Value::SparseTensor(matrix), true)),
        Value::Num(n) => Ok((
            Value::Tensor(
                Tensor::new(vec![n], vec![1, 1])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ),
            false,
        )),
        Value::Int(i) => Ok((
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::from_scalar(i), vec![1, 1])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ),
            false,
        )),
        Value::Bool(b) => Ok((
            Value::LogicalArray(
                LogicalArray::new(vec![u8::from(b)], vec![1, 1])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ),
            false,
        )),
        Value::Complex(re, im) => Ok((
            Value::ComplexTensor(
                ComplexTensor::new(vec![(re, im)], vec![1, 1])
                    .map_err(|e| internal(format!("decomposition: {e}")))?,
            ),
            false,
        )),
        other => Err(invalid(format!(
            "decomposition: coefficient matrix must be numeric, got {other:?}"
        ))),
    }
}

pub(super) fn validate_matrix_shape(shape: &[usize]) -> BuiltinResult<()> {
    if shape.iter().skip(2).any(|d| *d != 1) {
        return Err(invalid(
            "decomposition: coefficient matrix must be two-dimensional",
        ));
    }
    Ok(())
}

pub(super) fn parse_options(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<(bool, Option<f64>)> {
    let mut check_condition = true;
    let mut rank_tolerance = None;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid(
                "decomposition: name-value options must be provided in pairs",
            ));
        }
        let name = text_from_value(&args[idx])?
            .ok_or_else(|| invalid("decomposition: option name must be text"))?;
        match DecompositionOption::parse(&name) {
            Some(DecompositionOption::CheckCondition) => {
                check_condition = bool_scalar(&args[idx + 1], "CheckCondition")?
            }
            Some(DecompositionOption::RankTolerance) => {
                if crate::builtins::common::validation::is_typed_complex_integer(&args[idx + 1]) {
                    return Err(invalid(
                        "decomposition: RankTolerance must be a real scalar, not a paired complex integer",
                    ));
                }
                ensure_exact_integer_scalar_boundary(&args[idx + 1], "RankTolerance")?;
                let tol = real_scalar(&args[idx + 1], "RankTolerance")?;
                if tol < 0.0 || tol.is_nan() {
                    return Err(invalid(
                        "decomposition: RankTolerance must be a nonnegative scalar",
                    ));
                }
                rank_tolerance = Some(tol);
            }
            None => {
                return Err(invalid(format!(
                    "decomposition: unsupported option '{name}'"
                )))
            }
        }
        idx += 2;
    }
    Ok((check_condition, rank_tolerance))
}

pub(super) fn is_decomposition_type(value: &str) -> bool {
    parse_decomposition_type(value).is_ok()
}

pub(super) fn is_decomposition_option(value: &str) -> bool {
    DecompositionOption::parse(value).is_some()
}

pub(super) fn parse_decomposition_type(value: &str) -> BuiltinResult<DecompositionType> {
    DecompositionType::parse(value).ok_or_else(|| {
        invalid(format!(
            "decomposition: unsupported decomposition type '{value}'"
        ))
    })
}

pub(super) fn decomposition_type_name(value: DecompositionType) -> &'static str {
    value.as_str()
}

pub(super) fn parse_triangular_flag(value: &str) -> BuiltinResult<TriangularFlag> {
    TriangularFlag::parse(value).ok_or_else(|| {
        invalid(format!(
            "decomposition: unsupported triangular flag '{value}'"
        ))
    })
}
