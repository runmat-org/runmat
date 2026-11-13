//! Canonical constructor ingress; retained arguments stay rooted through gather.
use super::*;
use crate::{call::BuiltinArguments, context::RuntimeContext};
use allocation::UnboundedTabularWork;
use runmat_builtins::{
    CATEGORICAL_GPU_INPUT_EXTENSION, DICTIONARY_GPU_INPUT_EXTENSION, ORDINAL_GPU_INPUT_EXTENSION,
    TABLE_GPU_INPUT_EXTENSION,
};

pub(in crate::builtins::table) fn extension(
    runtime: Option<&RuntimeContext>,
    descriptor: &'static BuiltinExtensionDescriptor,
    builtin: &'static str,
) -> BuiltinResult<()> {
    match runtime {
        Some(runtime) => crate::compatibility::ensure_builtin_extension_enabled_in_context(
            runtime, descriptor, builtin,
        ),
        None => crate::compatibility::ensure_builtin_extension_enabled(descriptor, builtin),
    }
}

#[runtime_builtin(
    name = "table",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn table_builtin(
    runtime: Option<RuntimeContext>,
    args: BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::TABLE_DESCRIPTOR,
        &TABLE_ERROR_INVALID_ARGUMENT,
        "table",
    )?;
    let work = UnboundedTabularWork::admit(runtime.as_ref(), "table")?;
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        extension(runtime.as_ref(), &TABLE_GPU_INPUT_EXTENSION, "table")?;
    }
    ensure_table_class_registered();
    let args = gather_retained_values(&args, runtime.as_ref()).await?;
    work.check()?;
    let (variables, options) = split_table_constructor_args(args.view())?;
    if options.size.is_some() || options.variable_types.is_some() {
        if variables.clone().next().is_some() {
            return Err(invalid_argument(
                "table: Size/VariableTypes preallocation cannot be combined with data variables",
            ));
        }
        return work.finish(preallocated_table(Vec::new(), options));
    }
    let names = options
        .variable_names
        .unwrap_or_else(|| generated_variable_names(variables.clone().count()));
    work.finish(table_from_retained_columns_with_class(
        runtime.as_ref(),
        TABLE_CLASS,
        names,
        variables,
        options.row_names,
    ))
}

#[runtime_builtin(
    name = "categorical",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn categorical_builtin(
    runtime: Option<RuntimeContext>,
    args: BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::CATEGORICAL_DESCRIPTOR,
        &TABLE_ERROR_INVALID_ARGUMENT,
        "categorical",
    )?;
    let work = UnboundedTabularWork::admit(runtime.as_ref(), "categorical")?;
    if args.iter().any(crate::value_contains_gpu) {
        extension(
            runtime.as_ref(),
            &CATEGORICAL_GPU_INPUT_EXTENSION,
            "categorical",
        )?;
    }
    ensure_table_class_registered();
    let args = gather_retained_values(&args, runtime.as_ref()).await?;
    work.check()?;
    work.finish(categorical_from_args(args.view()))
}
#[runtime_builtin(
    name = "ordinal",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn ordinal_builtin(
    runtime: Option<RuntimeContext>,
    args: BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::ORDINAL_DESCRIPTOR,
        &TABLE_ERROR_INVALID_ARGUMENT,
        "ordinal",
    )?;
    let work = UnboundedTabularWork::admit(runtime.as_ref(), "ordinal")?;
    if args
        .iter()
        .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        extension(runtime.as_ref(), &ORDINAL_GPU_INPUT_EXTENSION, "ordinal")?;
    }
    ensure_table_class_registered();
    let args = gather_retained_values(&args, runtime.as_ref()).await?;
    work.check()?;
    work.finish(ordinal_from_args(args.view()))
}
#[runtime_builtin(
    name = "dictionary",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn dictionary_builtin(
    runtime: Option<RuntimeContext>,
    args: BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::DICTIONARY_DESCRIPTOR,
        &TABLE_ERROR_INVALID_ARGUMENT,
        "dictionary",
    )?;
    let work = UnboundedTabularWork::admit(runtime.as_ref(), "dictionary")?;
    if args.iter().any(crate::value_contains_gpu) {
        extension(
            runtime.as_ref(),
            &DICTIONARY_GPU_INPUT_EXTENSION,
            "dictionary",
        )?;
    }
    ensure_table_class_registered();
    let args = gather_retained_values(&args, runtime.as_ref()).await?;
    work.check()?;
    work.finish(dictionary_from_args(args.view()))
}
