use super::*;

async fn in_selected_runtime<T>(
    runtime: Option<&crate::context::RuntimeContext>,
    future: impl std::future::Future<Output = T>,
) -> T {
    match runtime {
        Some(runtime) => runtime.scope(future).await,
        None => future.await,
    }
}

pub(super) async fn execute_saveas(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<bool> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_saveas_in_context(runtime.as_ref(), args),
    )
    .await
}

async fn execute_saveas_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<bool> {
    reject_resident_persistence_arguments(&args, SAVEAS)?;
    let request = parse_saveas_args(&args)?;
    let selected = FigureFsPath::capture(&request.output_path).map_err(|error| {
        persistence_error_with_source(
            &ERROR_IO,
            SAVEAS,
            "saveas: unable to capture output path",
            error,
        )
    })?;
    match request.format {
        SaveFormat::Fig => save_figures(runtime, &selected, &[request.figure], SAVEAS).await?,
        SaveFormat::Png => {
            let bytes = render_png(request.figure, DEFAULT_WIDTH, DEFAULT_HEIGHT, SAVEAS).await?;
            write_bytes(runtime, &selected, &bytes, SAVEAS).await?;
        }
    }
    Ok(true)
}

pub(super) async fn execute_savefig(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<bool> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_savefig_in_context(runtime.as_ref(), args),
    )
    .await
}

async fn execute_savefig_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<bool> {
    reject_resident_persistence_arguments(&args, SAVEFIG)?;
    let request = parse_savefig_args(&args, SAVEFIG)?;
    let selected = FigureFsPath::capture(&request.output_path).map_err(|error| {
        persistence_error_with_source(
            &ERROR_IO,
            SAVEFIG,
            "savefig: unable to capture output path",
            error,
        )
    })?;
    save_figures(runtime, &selected, &request.figures, SAVEFIG).await?;
    Ok(true)
}

pub(super) async fn execute_openfig(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_openfig_in_context(runtime.as_ref(), args),
    )
    .await
}

async fn execute_openfig_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let request = parse_openfig_args(&args, OPENFIG)?;
    let selected = FigureFsPath::capture(&request.path).map_err(|error| {
        persistence_error_with_source(
            &ERROR_IO,
            OPENFIG,
            "openfig: unable to capture input path",
            error,
        )
    })?;
    let handles = open_figures(runtime, &selected, &request, OPENFIG).await?;
    Ok(handles_value(&handles))
}

pub(super) async fn execute_hgsave(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<bool> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_hgsave_in_context(runtime.as_ref(), args),
    )
    .await
}

async fn execute_hgsave_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<bool> {
    reject_resident_persistence_arguments(&args, HGSAVE)?;
    let request = parse_savefig_args(&args, HGSAVE)?;
    let selected = FigureFsPath::capture(&request.output_path).map_err(|error| {
        persistence_error_with_source(
            &ERROR_IO,
            HGSAVE,
            "hgsave: unable to capture output path",
            error,
        )
    })?;
    save_figures(runtime, &selected, &request.figures, HGSAVE).await?;
    Ok(true)
}

pub(super) async fn execute_hgload(
    runtime: Option<crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_hgload_in_context(runtime.as_ref(), args),
    )
    .await
}

async fn execute_hgload_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    args: Vec<Value>,
) -> BuiltinResult<Value> {
    let (request, overrides) = parse_hgload_args(&args)?;
    let selected = FigureFsPath::capture(&request.path).map_err(|error| {
        persistence_error_with_source(
            &ERROR_IO,
            HGLOAD,
            "hgload: unable to capture input path",
            error,
        )
    })?;
    let handles = open_figures(runtime, &selected, &request, HGLOAD).await?;
    let old_values = apply_hgload_overrides(&handles, overrides.as_ref())?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            return Ok(Value::OutputList(Vec::new()));
        }
        let outputs = vec![handles_value(&handles), old_values];
        return Ok(crate::output_count::output_list_with_padding(
            out_count, outputs,
        ));
    }
    Ok(handles_value(&handles))
}

pub(super) fn reject_resident_persistence_arguments(
    values: &[Value],
    builtin: &'static str,
) -> BuiltinResult<()> {
    if values.iter().any(|value| {
        crate::builtins::common::validation::value_has_native_integer_class(value)
            || matches!(value, Value::GpuTensor(_))
    }) {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!(
                "{builtin}: typed integer and gpuArray values are not valid figure, filename, or format arguments"
            ),
        ));
    }
    Ok(())
}
