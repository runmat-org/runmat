use super::*;

pub(super) async fn execute_saveas(args: Vec<Value>) -> BuiltinResult<bool> {
    reject_resident_persistence_arguments(&args, SAVEAS)?;
    let args = gather_values(&args, SAVEAS).await?;
    let request = parse_saveas_args(&args)?;
    match request.format {
        SaveFormat::Fig => save_figures(&request.output_path, &[request.figure], SAVEAS).await?,
        SaveFormat::Png => {
            let bytes = render_png(request.figure, DEFAULT_WIDTH, DEFAULT_HEIGHT, SAVEAS).await?;
            write_bytes(&request.output_path, &bytes, SAVEAS).await?;
        }
    }
    Ok(true)
}

pub(super) async fn execute_savefig(args: Vec<Value>) -> BuiltinResult<bool> {
    reject_resident_persistence_arguments(&args, SAVEFIG)?;
    let args = gather_values(&args, SAVEFIG).await?;
    let request = parse_savefig_args(&args, SAVEFIG)?;
    save_figures(&request.output_path, &request.figures, SAVEFIG).await?;
    Ok(true)
}

pub(super) async fn execute_openfig(args: Vec<Value>) -> BuiltinResult<Value> {
    let request = parse_openfig_args(&args, OPENFIG)?;
    let handles = open_figures(&request, OPENFIG).await?;
    Ok(handles_value(&handles))
}

pub(super) async fn execute_hgsave(args: Vec<Value>) -> BuiltinResult<bool> {
    let request = parse_savefig_args(&args, HGSAVE)?;
    save_figures(&request.output_path, &request.figures, HGSAVE).await?;
    Ok(true)
}

pub(super) async fn execute_hgload(args: Vec<Value>) -> BuiltinResult<Value> {
    let (request, overrides) = parse_hgload_args(&args)?;
    let handles = open_figures(&request, HGLOAD).await?;
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

pub(super) async fn gather_values(
    values: &[Value],
    builtin: &'static str,
) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather_if_needed_async(value).await.map_err(|err| {
            let message = err.message().to_string();
            persistence_error_with_source(&ERROR_INTERNAL, builtin, message, err)
        })?);
    }
    Ok(out)
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
