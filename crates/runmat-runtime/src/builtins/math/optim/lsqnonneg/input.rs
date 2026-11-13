//! Original operand grammar and explicit datatype/residency admission.
use super::*;
use runmat_value::SparseTensor;
pub(super) async fn parse(
    args: crate::call::BuiltinArguments,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(SparseTensor, Vec<f64>, options::Controls)> {
    let values = if args.len() == 1 {
        let Value::Struct(problem) = args.into_iter().next().unwrap() else {
            return Err(invalid("single argument must be a scalar problem struct"));
        };
        for field in ["C", "d", "solver", "options"] {
            if !problem.fields.contains_key(field) {
                return Err(invalid(format!("problem requires {field}")));
            }
        }
        if !common::field_name(&problem.fields["solver"])?.eq_ignore_ascii_case(NAME) {
            return Err(invalid("problem solver must be lsqnonneg"));
        }
        ["C", "d", "options"]
            .into_iter()
            .map(|name| problem.fields[name].clone())
            .collect::<Vec<_>>()
    } else {
        if !matches!(args.len(), 2 | 3) {
            return Err(invalid(
                "expected C,d, optional options, or a problem struct",
            ));
        }
        let mut values = args.into_iter().collect::<Vec<_>>();
        values.resize_with(3, empty);
        values
    };
    preflight(
        &values[0],
        &runmat_builtins::LSQNONNEG_INTEGER_C_EXTENSION,
        cooperation,
    )?;
    preflight(
        &values[1],
        &runmat_builtins::LSQNONNEG_INTEGER_D_EXTENSION,
        cooperation,
    )?;
    let mut values = values.into_iter();
    let c = values.next().unwrap();
    let d = values.next().unwrap();
    let controls = values.next().unwrap();
    let runtime = cooperation
        .runtime()
        .cloned()
        .expect("NNLS ingress owns context");
    let controls = options::Controls::parse(controls, &runtime, cooperation).await?;
    let c = crate::dispatcher::gather_if_needed_in_context_async(&c, Some(&runtime)).await?;
    let c = common::conic::matrix::parse(
        c,
        common::conic::matrix::IntegerPolicy::Exact,
        cooperation,
        |detail: &str| invalid(detail),
    )
    .await?
    .ok_or_else(|| invalid("C must have at least one row and column"))?;
    let d = crate::dispatcher::gather_if_needed_in_context_async(&d, Some(&runtime)).await?;
    let d = vector(d, cooperation).await?;
    if c.m != d.len() {
        return Err(invalid("d must be a real vector with one entry per C row"));
    }
    let c = SparseTensor::new(c.m, c.n, c.colptr, c.rowval, c.nzval).map_err(invalid)?;
    Ok((c, d, controls))
}
fn preflight(
    value: &Value,
    integer: &BuiltinExtensionDescriptor,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    // Runtime admission has a concrete representation: reject aggregates and
    // callables before richer value facts would visit their properties/captures.
    if !matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::GpuTensor(_)
    ) {
        return Err(invalid("C and d must be real numeric data"));
    }
    if !runmat_builtins::OptimNumericInput::OptionalSparseMatrix
        .accepts_metadata(&crate::value_fact::value_fact(value))
    {
        return Err(invalid(
            "C and d must be real numeric data, not logical, complex or object data",
        ));
    }
    let runtime = cooperation.runtime();
    common::preflight_host_residency(NAME, value, None, runtime)?;
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime, integer, NAME,
        )?;
    }
    let class=crate::builtins::introspection::underlying_type::underlying_class_identity_for_value_checked(value)?;
    if class == runmat_types::standard::SINGLE {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            &runmat_builtins::LSQNONNEG_SINGLE_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}
async fn vector(value: Value, cooperation: &mut ScalarCooperation) -> BuiltinResult<Vec<f64>> {
    if !cooperation.integer_boundary(&value).await?.is_exact() {
        return Err(invalid(
            "integer observations must be exactly representable as binary64",
        ));
    }
    let data = match value {
        Value::Num(value) => vec![value],
        Value::Int(value) => vec![value.to_f64()],
        Value::Tensor(t) if t.shape().len() <= 2 && (t.rows() == 1 || t.cols() == 1) => {
            common::tensor_values(&t, cooperation).await?
        }
        Value::SparseTensor(t) if t.rows == 1 || t.cols == 1 => {
            let c = common::conic::matrix::parse(
                Value::SparseTensor(t),
                common::conic::matrix::IntegerPolicy::Exact,
                cooperation,
                |detail: &str| invalid(detail),
            )
            .await?
            .ok_or_else(|| invalid("d must be nonempty"))?;
            let mut values = runmat_value::exact_capacity_vec(
                c.m.checked_mul(c.n)
                    .ok_or_else(|| invalid("observation size overflow"))?,
            )
            .map_err(allocation)?;
            values.resize(c.m * c.n, 0.0);
            for col in 0..c.n {
                for k in c.colptr[col]..c.colptr[col + 1] {
                    values[col * c.m + c.rowval[k]] = c.nzval[k];
                    if k % 1024 == 1023 {
                        cooperation.completed(1024).await?;
                    }
                }
                cooperation.completed(1).await?;
            }
            values
        }
        _ => return Err(invalid("d must be a nonempty real numeric vector")),
    };
    for chunk in data.chunks(1024) {
        if chunk.iter().any(|v| !v.is_finite()) {
            return Err(invalid("observations must be finite"));
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(data)
}
