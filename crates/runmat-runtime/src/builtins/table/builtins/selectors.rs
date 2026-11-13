use super::*;
use runmat_builtins::HEAD_GPU_COUNT_EXTENSION;
use runmat_macros::runtime_builtin;

#[runtime_builtin(
    name = "head",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::table::builtins"
)]
pub(crate) async fn head_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(invalid_argument(
            "head: expected at most one row-count argument",
        ));
    }
    let n = match rest.first() {
        Some(Value::GpuTensor(handle)) => {
            crate::compatibility::ensure_builtin_extension_enabled(
                &HEAD_GPU_COUNT_EXTENSION,
                "head",
            )?;
            let provider = runmat_accelerate_api::provider_for_handle(handle)
                .ok_or_else(|| invalid_argument("head: GPU row count has no owning provider"))?;
            let host =
                crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
                    provider, handle,
                )
                .await
                .map_err(map_control_flow)?;
            positive_usize(&host, "head row count")?
        }
        Some(value) => positive_usize(value, "head row count")?,
        None => 8,
    };
    if let Value::GpuTensor(handle) = value {
        let provider = runmat_accelerate_api::provider_for_handle(&handle)
            .ok_or_else(|| invalid_argument("head: GPU input has no owning provider"))?;
        let host = crate::builtins::common::gpu_helpers::download_value_preserving_residency_async(
            provider, &handle,
        )
        .await
        .map_err(map_control_flow)?;
        let rows = value_row_count(&host)?;
        let selected = (0..rows.min(n)).collect::<Vec<_>>();
        let selected_value = select_rows_in_context(runtime.as_ref(), &host, &selected)?;
        return crate::builtins::common::gpu_helpers::restore_class_preserving_value(
            &handle,
            selected_value,
            "head",
        );
    }
    let value = gather_if_needed_async(&value)
        .await
        .map_err(map_control_flow)?;
    let rows = value_row_count(&value)?;
    let selected = (0..rows.min(n)).collect::<Vec<_>>();
    if let Some(object) = table_object(&value) {
        let names = table_variable_names_from_object(object)?;
        let variables = table_variables(object)?;
        let mut columns = Vec::with_capacity(names.len());
        for name in &names {
            columns.push(select_rows_in_context(
                runtime.as_ref(),
                variables
                    .fields
                    .get(name)
                    .ok_or_else(|| invalid_variable(format!("head: missing variable '{name}'")))?,
                &selected,
            )?);
        }
        return subset_tabular_object_in_context(
            runtime.as_ref(),
            object,
            names,
            columns,
            &selected,
        );
    }
    select_rows_in_context(runtime.as_ref(), &value, &selected)
}
