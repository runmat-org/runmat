use super::*;

pub(super) async fn parse_constructor(args: Vec<Value>) -> BuiltinResult<DecompositionSpec> {
    if args.is_empty() {
        return Err(invalid("decomposition: expected coefficient matrix"));
    }
    let host_args = gather_values(args).await?;
    let mut iter = host_args.into_iter();
    let raw_matrix = iter
        .next()
        .ok_or_else(|| invalid("decomposition: expected coefficient matrix"))?;
    let (matrix, is_sparse) = normalize_matrix(raw_matrix)?;
    let mut decomp_type = DecompositionType::Auto;
    let mut triangular_flag: Option<TriangularFlag> = None;
    let mut options = Vec::new();

    if let Some(next) = iter.next() {
        if let Some(text) = text_from_value(&next) {
            if is_decomposition_type(&text) {
                decomp_type = parse_decomposition_type(&text)?;
                if let Some(candidate) = iter.next() {
                    if let Some(flag) = text_from_value(&candidate)
                        .and_then(|value| parse_triangular_flag(&value).ok())
                    {
                        triangular_flag = Some(flag);
                    } else {
                        options.push(candidate);
                    }
                }
            } else if is_decomposition_option(&text) {
                options.push(next);
            } else {
                return Err(parse_decomposition_type(&text).unwrap_err());
            }
        } else {
            options.push(next);
        }
    }
    options.extend(iter);

    let (check_condition, rank_tolerance) = parse_options(&options)?;
    let matrix = apply_constructor_structure(matrix, decomp_type, triangular_flag)?;
    let decomp_type = if decomp_type == DecompositionType::Auto {
        infer_decomposition_type(&matrix)
    } else {
        validate_type_compatibility(&matrix, decomp_type)?;
        if decomp_type == DecompositionType::Chol {
            validate_chol_positive_definite(&matrix).await?;
        }
        decomp_type
    };

    Ok(DecompositionSpec {
        matrix,
        decomp_type,
        check_condition,
        rank_tolerance,
        is_conjugate_transposed: false,
        is_sparse,
        scale: (1.0, 0.0),
    })
}

pub(super) async fn gather_values(args: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(args.len());
    for value in args {
        out.push(
            crate::dispatcher::gather_if_needed_async(&value)
                .await
                .map_err(|err| {
                    build_runtime_error(err.message())
                        .with_builtin(NAME)
                        .with_source(err)
                        .build()
                })?,
        );
    }
    Ok(out)
}

pub(super) fn normalize_matrix(value: Value) -> BuiltinResult<(Value, bool)> {
    match value {
        Value::Tensor(tensor) => {
            validate_matrix_shape(&tensor.shape)?;
            Ok((Value::Tensor(tensor), false))
        }
        Value::ComplexTensor(tensor) => {
            validate_matrix_shape(&tensor.shape)?;
            Ok((Value::ComplexTensor(tensor), false))
        }
        Value::LogicalArray(array) => {
            validate_matrix_shape(&array.shape)?;
            Ok((Value::LogicalArray(array), false))
        }
        Value::SparseTensor(_) => Err(invalid(
            "decomposition: sparse coefficient matrices are not supported yet",
        )),
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
    if shape.len() > 2 {
        return Err(invalid(
            "decomposition: coefficient matrix must be two-dimensional",
        ));
    }
    Ok(())
}

pub(super) fn parse_options(args: &[Value]) -> BuiltinResult<(bool, Option<f64>)> {
    let mut check_condition = true;
    let mut rank_tolerance = None;
    let mut idx = 0usize;
    while idx < args.len() {
        if idx + 1 >= args.len() {
            return Err(invalid(
                "decomposition: name-value options must be provided in pairs",
            ));
        }
        let name = text_from_value(&args[idx])
            .ok_or_else(|| invalid("decomposition: option name must be text"))?;
        match name.to_ascii_lowercase().as_str() {
            "checkcondition" => check_condition = bool_scalar(&args[idx + 1], "CheckCondition")?,
            "ranktolerance" => {
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
            other => {
                return Err(invalid(format!(
                    "decomposition: unsupported option '{other}'"
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
    matches!(
        value.to_ascii_lowercase().as_str(),
        "checkcondition" | "ranktolerance"
    )
}

pub(super) fn parse_decomposition_type(value: &str) -> BuiltinResult<DecompositionType> {
    match value.to_ascii_lowercase().as_str() {
        "auto" => Ok(DecompositionType::Auto),
        "lu" => Ok(DecompositionType::Lu),
        "qr" => Ok(DecompositionType::Qr),
        "chol" => Ok(DecompositionType::Chol),
        "triangular" => Ok(DecompositionType::Triangular),
        "diagonal" => Ok(DecompositionType::Diagonal),
        other => Err(invalid(format!(
            "decomposition: unsupported decomposition type '{other}'"
        ))),
    }
}

pub(super) fn decomposition_type_name(value: DecompositionType) -> &'static str {
    match value {
        DecompositionType::Auto => "auto",
        DecompositionType::Lu => "lu",
        DecompositionType::Qr => "qr",
        DecompositionType::Chol => "chol",
        DecompositionType::Triangular => "triangular",
        DecompositionType::Diagonal => "diagonal",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TriangularFlag {
    Upper,
    Lower,
}

pub(super) fn parse_triangular_flag(value: &str) -> BuiltinResult<TriangularFlag> {
    match value.to_ascii_lowercase().as_str() {
        "upper" => Ok(TriangularFlag::Upper),
        "lower" => Ok(TriangularFlag::Lower),
        other => Err(invalid(format!(
            "decomposition: unsupported triangular flag '{other}'"
        ))),
    }
}

pub(super) fn apply_constructor_structure(
    matrix: Value,
    decomp_type: DecompositionType,
    triangular_flag: Option<TriangularFlag>,
) -> BuiltinResult<Value> {
    match (decomp_type, triangular_flag) {
        (DecompositionType::Triangular, Some(flag)) => keep_triangle(matrix, flag),
        (DecompositionType::Chol, Some(flag)) => symmetrize_from_triangle(matrix, flag),
        (DecompositionType::Triangular, None)
        | (DecompositionType::Auto, None)
        | (DecompositionType::Lu, None)
        | (DecompositionType::Qr, None)
        | (DecompositionType::Chol, None)
        | (DecompositionType::Diagonal, None) => Ok(matrix),
        (_, Some(_)) => Err(invalid(
            "decomposition: triangularFlag is supported only with types 'triangular' and 'chol'",
        )),
    }
}
