//! Borrowed public integration forms: input ownership stays at invocation ingress.
use super::*;
use crate::call::FunctionArgumentView;
use runmat_builtins::NumericalReductionKind;
pub(in crate::builtins::math::reduction) struct IntegrationArguments<'a> {
    pub y: &'a Value,
    pub spacing: Option<&'a Value>,
    pub dim: Option<usize>,
    pub dim_value: Option<&'a Value>,
}
pub(in crate::builtins::math::reduction) fn parse_integration_arguments<'a>(
    kind: NumericalReductionKind,
    first: &'a Value,
    rest: FunctionArgumentView<'a>,
) -> BuiltinResult<IntegrationArguments<'a>> {
    let (y, spacing, dimension) = match rest.len() {
        0 => (first, None, None),
        1 if is_dimension_candidate(rest.get(0).expect("one argument"))
            && !is_scalar_like(first) =>
        {
            (first, None, rest.get(0))
        }
        1 => (rest.get(0).expect("one argument"), Some(first), None),
        2 => (
            rest.get(0).expect("two arguments"),
            Some(first),
            rest.get(1),
        ),
        _ => return Err(argument_error(kind, "expected Y, Y/dim, X/Y, or X/Y/dim")),
    };
    let dim = dimension
        .map(|value| {
            parse_optional_dim(kind.catalog_entry().identity.name, value)
                .map_err(|source| argument_error(kind, source.message()))
        })
        .transpose()?
        .flatten();
    Ok(IntegrationArguments {
        y,
        spacing,
        dim,
        dim_value: dimension,
    })
}
pub(in crate::builtins::math::reduction) fn argument_error(
    kind: NumericalReductionKind,
    detail: &str,
) -> RuntimeError {
    super::super::numerical_execution::terminal(crate::runtime_descriptor_error_with_detail(
        kind.catalog_entry().identity.name,
        kind.argument_error(),
        detail,
    ))
}
pub(in crate::builtins::math::reduction) fn input_error(
    kind: NumericalReductionKind,
    detail: &str,
) -> RuntimeError {
    super::super::numerical_execution::terminal(crate::runtime_descriptor_error_with_detail(
        kind.catalog_entry().identity.name,
        kind.input_error(),
        detail,
    ))
}
pub(in crate::builtins::math::reduction) fn validate_integration_inputs(
    kind: NumericalReductionKind,
    args: &IntegrationArguments<'_>,
) -> BuiltinResult<()> {
    if !matches!(
        args.y,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::Complex(..)
            | Value::ComplexTensor(_)
            | Value::GpuTensor(_)
    ) || crate::builtins::common::validation::is_typed_complex_integer(args.y)
    {
        return Err(input_error(kind,"expected dense real or floating complex sampled data; sparse and complex-integer samples are unsupported"));
    }
    if args.spacing.is_some_and(|value| !matches!(value,Value::Num(_)|Value::Int(_)|Value::Bool(_)|Value::Tensor(_)|Value::LogicalArray(_)|Value::GpuTensor(_))
        || matches!(value,Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle)!=GpuTensorStorage::Real)) {
        return Err(argument_error(kind,"point spacing must be real numeric or logical data"));
    }
    Ok(())
}
