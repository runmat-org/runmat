use super::support::{reject_resident_persistence_arguments, validate_status_output};
use super::*;

pub(in super::super) async fn execute_saveas(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<bool> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_saveas_in_context(runtime.as_ref(), requested_outputs, args),
    )
    .await
}

async fn execute_saveas_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<bool> {
    validate_status_output(runtime, requested_outputs, SAVEAS)?;
    reject_resident_persistence_arguments(args.view(), SAVEAS)?;
    let request = parse_saveas_args(args.view())?;
    let selected = FigureFsPath::capture(&request.output_path).map_err(|source| {
        persistence_error_with_source(
            &ERROR_IO,
            SAVEAS,
            "saveas: unable to capture output path",
            source,
        )
    })?;
    match request.format {
        SaveFormat::Fig => {
            save_figures(
                runtime,
                &selected,
                &[request.figure],
                FigureArchiveProfile::Default,
                SAVEAS,
            )
            .await?
        }
        SaveFormat::Png => {
            let bytes = render_png(request.figure, DEFAULT_WIDTH, DEFAULT_HEIGHT, SAVEAS).await?;
            write_bytes(runtime, &selected, &bytes, SAVEAS).await?;
        }
    }
    Ok(true)
}

pub(in super::super) async fn execute_savefig(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<bool> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_savefig_in_context(runtime.as_ref(), requested_outputs, args),
    )
    .await
}

async fn execute_savefig_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<bool> {
    validate_status_output(runtime, requested_outputs, SAVEFIG)?;
    reject_resident_persistence_arguments(args.view(), SAVEFIG)?;
    let request = parse_savefig_args(args.view(), SAVEFIG)?;
    let selected = FigureFsPath::capture(&request.output_path).map_err(|source| {
        persistence_error_with_source(
            &ERROR_IO,
            SAVEFIG,
            "savefig: unable to capture output path",
            source,
        )
    })?;
    save_figures(
        runtime,
        &selected,
        &selection_parent_figures(&request.selection),
        request.profile,
        SAVEFIG,
    )
    .await?;
    Ok(true)
}

pub(in super::super) async fn execute_hgsave(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<bool> {
    let selected_runtime = runtime.clone();
    in_selected_runtime(
        selected_runtime.as_ref(),
        execute_hgsave_in_context(runtime.as_ref(), requested_outputs, args),
    )
    .await
}

async fn execute_hgsave_in_context(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<bool> {
    validate_status_output(runtime, requested_outputs, HGSAVE)?;
    reject_resident_persistence_arguments(args.view(), HGSAVE)?;
    let request = parse_savefig_args(args.view(), HGSAVE)?;
    let selected = FigureFsPath::capture(&request.output_path).map_err(|source| {
        persistence_error_with_source(
            &ERROR_IO,
            HGSAVE,
            "hgsave: unable to capture output path",
            source,
        )
    })?;
    save_figures(
        runtime,
        &selected,
        &selection_parent_figures(&request.selection),
        request.profile,
        HGSAVE,
    )
    .await?;
    Ok(true)
}
