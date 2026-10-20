//! Shared arity/output framing; each identity retains its own control parser.
use super::{invalid_argument_error, require_arg_count, BuiltinResult, Value};
use runmat_builtins::{BuiltinCatalogEntry, BuiltinErrorDescriptor, BuiltinParamArity};

pub(super) fn inputs(
    entry: &'static BuiltinCatalogEntry,
    output_error: &'static BuiltinErrorDescriptor,
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<()> {
    crate::output_count::validate_fixed_outputs(
        entry.descriptor,
        output_error,
        entry.identity.name,
    )?;
    let forms = entry.descriptor.signatures;
    let admitted = forms.iter().any(|form| {
        let min = form
            .inputs
            .iter()
            .filter(|input| input.arity == BuiltinParamArity::Required)
            .count();
        let max = form.inputs.len();
        args.len() >= min && args.len() <= max
    });
    if !admitted {
        return Err(invalid_argument_error(
            entry.identity.name,
            "invalid number of inputs",
        ));
    }
    require_arg_count(entry.identity.name, args, 1, usize::MAX)
}

/// Numeric device values cannot be text; rejecting here never accesses a provider.
pub(super) fn reject_resident_text(
    entry: &'static BuiltinCatalogEntry,
    value: &Value,
) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        return Err(crate::build_runtime_error(format!(
            "{}: value does not satisfy validator",
            entry.identity.name
        ))
        .with_builtin(entry.identity.name)
        .with_identifier(format!("RunMat:{}:ValidationFailed", entry.identity.name))
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    Ok(())
}
