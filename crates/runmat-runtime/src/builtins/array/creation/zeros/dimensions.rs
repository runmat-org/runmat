//! Dimension-only creation ABI for class methods. Class/prototype controls
//! are consumed by dispatch, never accepted and then ignored by this parser.
use crate::{call::FunctionArgumentView, context::RuntimeContext, BuiltinResult};
use runmat_value::Value;

pub(in crate::builtins) async fn parse_creation_dimensions(
    args: FunctionArgumentView<'_>,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<Vec<usize>> {
    crate::context::RuntimeScopeFactory::new(runtime.cloned(), move || parse_dimensions(args))
        .construct()
        .await
}

async fn parse_dimensions(args: FunctionArgumentView<'_>) -> BuiltinResult<Vec<usize>> {
    let mut dimensions = Vec::new();
    for value in args.iter() {
        let selected = argument(value)
            .await?
            .ok_or_else(|| super::builtin_error("zeros: expected numeric dimension arguments"))?;
        dimensions.try_reserve(selected.len()).map_err(|source| {
            crate::build_runtime_error("zeros: dimension metadata allocation failed")
                .with_builtin("zeros")
                .with_identifier("RunMat:zeros:AllocationFailed")
                .with_source(source)
                .build()
        })?;
        dimensions.extend(selected);
    }
    Ok(if args.is_empty() {
        vec![1, 1]
    } else if dimensions.is_empty() {
        vec![0, 0]
    } else {
        super::normalize_constructor_shape(dimensions)
    })
}

/// Shared with the numeric constructor, including its compatibility gates.
pub(super) async fn argument(value: &Value) -> BuiltinResult<Option<Vec<usize>>> {
    if matches!(value, Value::GpuTensor(_)) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::ZEROS_RESIDENT_SIZE_EXTENSION,
            "zeros",
        )?;
    }
    let Some(parsed) = super::extract_constructor_dimensions(value, "zeros")
        .await
        .map_err(super::builtin_error)?
    else {
        return Ok(None);
    };
    if parsed.is_column_vector {
        crate::compatibility::ensure_builtin_extension_enabled(
            &runmat_builtins::ZEROS_COLUMN_SIZE_VECTOR_EXTENSION,
            "zeros",
        )?;
    }
    Ok(Some(parsed.values))
}
