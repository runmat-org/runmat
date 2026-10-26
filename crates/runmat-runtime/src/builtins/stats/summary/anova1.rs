//! One-way ANOVA: observation grouping, F survival, and real presentation.
use super::inferential::{self, invalid, Cooperation};
use crate::BuiltinResult;
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, Tensor, Value};
mod engine;
mod groups;
mod presentation;
#[cfg(test)]
mod tests;
const NAME: &str = "anova1";

#[runtime_builtin(
    name = "anova1",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::stats::summary::anova1"
)]
async fn anova1_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    evaluate(runtime.as_ref(), args).await
}
pub(super) async fn evaluate(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let count = crate::output_count::current_output_count().unwrap_or(1);
    if args.is_empty() || args.len() > 3 || count > 3 {
        return Err(invalid(
            NAME,
            "expected y, optional group and display, and at most three outputs",
        ));
    }
    let display = match args.get(2) {
        None => true,
        Some(value) => {
            match crate::builtins::common::validation::value_text_scalar_string(value).as_deref() {
                Some(value) if value.eq_ignore_ascii_case("on") => true,
                Some(value) if value.eq_ignore_ascii_case("off") => false,
                _ => return Err(invalid(NAME, "display must be 'on' or 'off'")),
            }
        }
    };
    presentation::admit(display)?;
    crate::context::host_allocation::require_unbounded_workspace(runtime,NAME,
        "finite host-memory execution is unsupported; ANOVA group and result workspaces have no complete resource bound")?;
    inferential::admit_automatic_host_arguments(NAME, args.view())?;
    let _sources = args.fork();
    inferential::real_sample(NAME, &args[0])?;
    let mut cooperation = Cooperation::new(runtime);
    cooperation.completed(0).await?;
    let samples = inferential::gather(NAME, &args[0], runtime).await?;
    let group = match args.get(1) {
        Some(value) if !default_group_sentinel(value) => {
            let value =
                crate::dispatcher::gather_if_needed_in_context_async(value, runtime).await?;
            // Resident numeric [] has the same sentinel semantics after its
            // ordinary gather; empty text/cell/categorical values do not.
            if default_group_sentinel(&value) {
                None
            } else {
                Some(value)
            }
        }
        _ => None,
    };
    let data = groups::collect(&samples, group.as_ref(), &mut cooperation).await?;
    let mut outputs = engine::evaluate(&data, &mut cooperation).await?;
    if display {
        presentation::show(&data, &outputs, runtime, &mut cooperation).await?;
    }
    cooperation.completed(0).await?;
    outputs.truncate(count);
    Ok(Value::output_list(outputs))
}

fn default_group_sentinel(value: &Value) -> bool {
    matches!(value,Value::Tensor(array) if array.is_empty())
}
