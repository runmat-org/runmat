use super::*;

pub(super) async fn save_figures(
    path: &Path,
    figures: &[FigureHandle],
    builtin: &'static str,
) -> BuiltinResult<()> {
    if figures.is_empty() {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: at least one figure is required"),
        ));
    }
    let mut payloads = Vec::with_capacity(figures.len());
    for handle in figures {
        let bytes = export_figure(*handle, builtin).await?;
        payloads.push(bytes);
    }
    let bytes = encode_figure_bundle(&payloads, builtin)?;
    write_bytes(path, &bytes, builtin).await
}

pub(super) async fn open_figures(
    request: &OpenFigRequest,
    builtin: &'static str,
) -> BuiltinResult<Vec<FigureHandle>> {
    let bytes = runmat_filesystem::read_async(&request.path)
        .await
        .map_err(|err| {
            persistence_error_with_source(
                &ERROR_IO,
                builtin,
                format!(
                    "{builtin}: unable to read figure file '{}'",
                    request.path.display()
                ),
                err,
            )
        })?;
    let payloads = decode_figure_bundle(&bytes, builtin)?;
    let mut handles = Vec::with_capacity(payloads.len());
    for payload in payloads {
        let handle = import_figure(payload, builtin).await?;
        if let Some(visible) = request.visibility {
            set_figure_visible(handle, visible).map_err(|err| {
                persistence_error_with_source(&ERROR_IMPORT, builtin, err.to_string(), err)
            })?;
        }
        handles.push(handle);
    }
    Ok(handles)
}

#[cfg(feature = "plot-core")]
pub(super) async fn export_figure(
    handle: FigureHandle,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    super::export_figure_scene(handle)
        .await
        .map_err(|err| {
            persistence_error_with_source(
                &ERROR_RENDER,
                builtin,
                format!("{builtin}: unable to export figure scene"),
                err,
            )
        })?
        .ok_or_else(|| {
            persistence_error(
                &ERROR_RENDER,
                builtin,
                format!("{builtin}: plot-core exporter did not produce a figure scene"),
            )
        })
}

#[cfg(not(feature = "plot-core"))]
pub(super) async fn export_figure(
    _handle: FigureHandle,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    Err(persistence_error(
        &ERROR_RENDER,
        builtin,
        format!("{builtin}: plot-core support is not enabled in this build"),
    ))
}

#[cfg(feature = "plot-core")]
pub(super) async fn import_figure(
    bytes: &[u8],
    builtin: &'static str,
) -> BuiltinResult<FigureHandle> {
    super::import_figure_scene_async(bytes)
        .await
        .map_err(|err| {
            persistence_error_with_source(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: unable to import figure scene"),
                err,
            )
        })?
        .ok_or_else(|| {
            persistence_error(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: figure file did not contain an importable figure"),
            )
        })
}

#[cfg(not(feature = "plot-core"))]
pub(super) async fn import_figure(
    _bytes: &[u8],
    builtin: &'static str,
) -> BuiltinResult<FigureHandle> {
    Err(persistence_error(
        &ERROR_IMPORT,
        builtin,
        format!("{builtin}: plot-core support is not enabled in this build"),
    ))
}

#[cfg(feature = "plot-core")]
pub(super) async fn render_png(
    handle: FigureHandle,
    width: u32,
    height: u32,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    super::render_figure_snapshot(handle, width, height, None)
        .await
        .map_err(|err| {
            persistence_error_with_source(
                &ERROR_RENDER,
                builtin,
                format!("{builtin}: unable to render PNG"),
                err,
            )
        })
}

#[cfg(not(feature = "plot-core"))]
pub(super) async fn render_png(
    _handle: FigureHandle,
    _width: u32,
    _height: u32,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    Err(persistence_error(
        &ERROR_RENDER,
        builtin,
        format!("{builtin}: plot-core support is not enabled in this build"),
    ))
}
