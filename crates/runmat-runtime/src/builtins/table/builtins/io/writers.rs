//! Canonical writer ingress and pre-effect option/class/output validation.
#[cfg(test)]
mod tests;
use super::super::constructors::{allocation::UnboundedTabularWork, basic::extension};
use super::*;
use crate::{
    call::{BuiltinArguments, FunctionArgumentView},
    context::RuntimeContext,
    execution::RootedValue,
};
use runmat_builtins::{TabularWriteKind, TabularWriteOption};

#[runtime_builtin(
    name = "writetable",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn writetable_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    write(runtime.as_ref(), TabularWriteKind::Table, value, rest)
        .await
        .map_err(terminal)
}
#[runtime_builtin(
    name = "writetimetable",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn writetimetable_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    write(runtime.as_ref(), TabularWriteKind::Timetable, value, rest)
        .await
        .map_err(terminal)
}
fn terminal(mut error: crate::RuntimeError) -> crate::RuntimeError {
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
pub(in crate::builtins::table) fn invalid(
    kind: TabularWriteKind,
    message: impl AsRef<str>,
) -> crate::RuntimeError {
    crate::build_runtime_error(format!("{}: {}", kind.name(), message.as_ref()))
        .with_builtin(kind.name())
        .with_identifier(
            kind.argument_error()
                .identifier
                .expect("canonical writer argument identity"),
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn semantic(kind: TabularWriteKind, error: crate::RuntimeError) -> crate::RuntimeError {
    crate::build_runtime_error(format!("{}: {}", kind.name(), error.message()))
        .with_builtin(kind.name())
        .with_identifier(
            kind.argument_error()
                .identifier
                .expect("canonical writer argument identity"),
        )
        .with_source(error)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
pub(in crate::builtins::table) struct WriteOptions {
    pub delimiter: String,
    pub write_variable_names: bool,
}
impl WriteOptions {
    pub fn parse(kind: TabularWriteKind, args: FunctionArgumentView<'_>) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(invalid(kind, "name-value options must be paired"));
        }
        let mut options = Self {
            delimiter: ",".into(),
            write_variable_names: true,
        };
        for index in (0..args.len()).step_by(2) {
            let name = scalar_text(&args[index], "writer option").map_err(|e| semantic(kind, e))?;
            match TabularWriteOption::parse(&name) {
                Some(TabularWriteOption::Delimiter) => {
                    options.delimiter =
                        scalar_text(&args[index + 1], "Delimiter").map_err(|e| semantic(kind, e))?
                }
                Some(TabularWriteOption::VariableNames) => {
                    options.write_variable_names = match &args[index + 1] {
                        Value::LogicalArray(value) if value.len() == 1 => value.data[0] != 0,
                        Value::Tensor(value) if value.len() == 1 => {
                            zero_one_bool_scalar(&args[index + 1], "WriteVariableNames")
                                .map_err(|e| semantic(kind, e))?
                        }
                        value => bool_scalar(value, "WriteVariableNames")
                            .map_err(|e| semantic(kind, e))?,
                    }
                }
                None => return Err(invalid(kind, format!("unsupported option '{name}'"))),
            }
        }
        Ok(options)
    }
}
async fn write(
    runtime: Option<&RuntimeContext>,
    kind: TabularWriteKind,
    value: Value,
    rest: BuiltinArguments,
) -> BuiltinResult<Value> {
    let outputs = crate::output_count::current_output_count();
    if outputs.is_some_and(|n| n > 1) {
        return Err(invalid(kind, "too many output arguments"));
    }
    if outputs.is_some_and(|n| n > 0) {
        extension(runtime, kind.bytes_extension(), kind.name())?;
    }
    let work = UnboundedTabularWork::admit(runtime, kind.name())?;
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value)
        || rest
            .iter()
            .any(crate::builtins::common::validation::value_contains_explicit_gpu)
    {
        extension(runtime, kind.gpu_extension(), kind.name())?;
    }
    if rest.is_empty() {
        return Err(invalid(kind, "filename is required"));
    }
    for index in (1..rest.len().saturating_sub(1)).step_by(2) {
        let name = scalar_text(&rest[index], "writer option").map_err(|e| semantic(kind, e))?;
        if TabularWriteOption::parse(&name) == Some(TabularWriteOption::VariableNames)
            && super::value_contains_typed_integer(&rest[index + 1])
        {
            extension(runtime, kind.integer_extension(), kind.name())?;
        }
    }
    if !matches!(&value,Value::Object(object) if object.is_class(kind.class())) {
        return Err(invalid(
            kind,
            format!("expected {}", kind.class().display_name()),
        ));
    }
    // Capture provider/cwd before any operand gather can suspend. The exact
    // selected provider and its resource lease remain owned through publication.
    let value = RootedValue::in_optional_context(value, runtime)?;
    let capture = async { crate::builtins::io::tabular::table_write::capture(kind, None) };
    let owner = match runtime {
        Some(runtime) => runtime.scope(capture).await,
        None => capture.await,
    }?;
    let gathered = gather_retained_value(value.value(), runtime).await?;
    let gathered = RootedValue::in_optional_context(gathered, runtime)?;
    let rest = gather_retained_values(&rest, runtime).await?;
    work.check()?;
    let options = WriteOptions::parse(kind, rest.view().range(1..)?)?;
    let path = resolve_path(&rest[0]).map_err(|e| semantic(kind, e))?;
    if path.as_os_str().is_empty() {
        return Err(invalid(kind, "filename must be nonempty scalar text"));
    }
    let selected = crate::builtins::io::tabular::table_write::resolve(kind, owner, path)?;
    let Value::Object(object) = gathered.value() else {
        return Err(invalid(kind, "gathered input is not tabular"));
    };
    let text = table_delimited_text(
        object,
        &options.delimiter,
        options.write_variable_names,
        kind == TabularWriteKind::Timetable,
    )
    .map_err(|e| semantic(kind, e))?;
    // The numeric extension must be exact before any filesystem publication.
    if text.len() as u128 > (1u128 << 53) {
        return Err(invalid(
            kind,
            "bytes-written output exceeds exact double range",
        ));
    }
    work.check()?;
    let publish = crate::builtins::io::tabular::table_write::publish(
        runtime,
        kind,
        &selected,
        text.as_bytes(),
    );
    match runtime {
        Some(runtime) => runtime.scope(publish).await,
        None => publish.await,
    }?;
    work.finish(Ok(Value::Num(text.len() as f64)))
}
