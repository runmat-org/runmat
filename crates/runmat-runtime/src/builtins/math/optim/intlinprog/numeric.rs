//! Original-context admission before any selected numeric role downloads.
use super::*;
use clarabel::algebra::CscMatrix;
use common::conic;
use runmat_types::standard;

#[derive(Clone, Copy)]
pub(super) enum Role {
    Objective,
    IntegerIndices,
    InequalityMatrix,
    InequalityRhs,
    EqualityMatrix,
    EqualityRhs,
    Lower,
    Upper,
    Initial,
    Control,
}
impl Role {
    fn ordinary_single(self) -> bool {
        matches!(
            self,
            Self::InequalityMatrix | Self::InequalityRhs | Self::EqualityRhs | Self::Control
        )
    }
}
pub(super) fn preflight(
    value: &Value,
    role: Role,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    // Admit concrete numeric carriers before inspecting metadata; captures,
    // object properties and unrelated struct fields are never walked.
    if !matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::GpuTensor(_)
    ) {
        return Err(invalid(
            "expected real numeric data, not logical, complex, object or callable data",
        ));
    }
    let fact = crate::value_fact::value_fact(value);
    if matches!(
        role,
        Role::InequalityMatrix | Role::EqualityMatrix | Role::InequalityRhs | Role::EqualityRhs
    ) && fact.shape.rank().is_some_and(|n| n > 2)
    {
        return Err(invalid(
            "constraint matrices and right sides must be two-dimensional",
        ));
    }
    if !matches!(
        fact.kind,
        runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
            domain: runmat_types::NumericDomain::Real,
            ..
        })
    ) {
        return Err(invalid("expected real numeric data"));
    }
    common::preflight_host_residency(NAME, value, None, runtime)?;
    if crate::builtins::common::validation::value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            &runmat_builtins::INTLINPROG_INTEGER_EXTENSION,
            NAME,
        )?;
    }
    if !role.ordinary_single() && crate::builtins::introspection::underlying_type::underlying_class_identity_for_value_checked(value)?==standard::SINGLE {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(runtime,
            &runmat_builtins::INTLINPROG_SINGLE_EXTENSION,NAME)?;
    }
    Ok(())
}
pub(super) async fn gather(value: Value, cooperation: &ScalarCooperation) -> BuiltinResult<Value> {
    crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime()).await
}
pub(super) async fn matrix(
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<CscMatrix<f64>>> {
    conic::matrix::parse(
        gather(value, cooperation).await?,
        conic::matrix::IntegerPolicy::Exact,
        cooperation,
        |detail: &str| invalid(detail),
    )
    .await
}
/// Flatten array roles in column-major order. Sparse RHS restrictions are
/// checked by the typed public parser before this common numeric conversion.
pub(super) async fn vector(
    value: Value,
    array: bool,
    allow_infinity: bool,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let value = gather(value, cooperation).await?;
    if !cooperation.integer_boundary(&value).await?.is_exact() {
        return Err(invalid(
            "native integer data must be exactly representable as double",
        ));
    }
    let values = match value {
        Value::Num(v) => vec![v],
        Value::Int(v) => vec![v.to_f64()],
        Value::Tensor(v)
            if array
                || v.is_empty()
                || v.shape().len() <= 2 && (v.rows() == 1 || v.cols() == 1) =>
        {
            common::tensor_values(&v, cooperation).await?
        }
        Value::SparseTensor(v) => {
            let matrix = conic::matrix::parse(
                Value::SparseTensor(v),
                conic::matrix::IntegerPolicy::Exact,
                cooperation,
                |detail: &str| invalid(detail),
            )
            .await?;
            let Some(matrix) = matrix else {
                return Ok(Vec::new());
            };
            if !array && matrix.m != 1 && matrix.n != 1 {
                return Err(invalid("expected a numeric vector"));
            }
            let length = matrix
                .m
                .checked_mul(matrix.n)
                .ok_or_else(|| invalid("array size overflow"))?;
            let mut values = runmat_value::exact_capacity_vec(length).map_err(allocation)?;
            values.resize(length, 0.0);
            for col in 0..matrix.n {
                for k in matrix.colptr[col]..matrix.colptr[col + 1] {
                    values[col * matrix.m + matrix.rowval[k]] = matrix.nzval[k];
                }
                cooperation
                    .completed((matrix.colptr[col + 1] - matrix.colptr[col]) as u64 + 1)
                    .await?;
            }
            values
        }
        _ => return Err(invalid("expected real numeric array")),
    };
    for chunk in values.chunks(1024) {
        if chunk
            .iter()
            .any(|v| v.is_nan() || !allow_infinity && !v.is_finite())
        {
            return Err(invalid("numeric data has invalid nonfinite values"));
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(values)
}
