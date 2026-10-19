use super::*;

pub(in super::super) async fn open_figures(
    runtime: Option<&crate::context::RuntimeContext>,
    selected: &FigureFsPath,
    request: &OpenFigRequest,
    builtin: &'static str,
) -> BuiltinResult<ImportedFigureBatch> {
    let bytes = selected.read(runtime, builtin).await?;
    let payloads = decode_figure_bundle(runtime, &bytes, builtin).await?;
    #[cfg(feature = "plot-core")]
    {
        let mut figures = Vec::with_capacity(payloads.len());
        for payload in &payloads {
            let mut figure = prepare_figure(payload, builtin).await?;
            if let Some(visible) = request.visibility {
                figure.visible = visible;
            }
            figures.push(figure);
        }
        let previous = super::super::super::state::current_figure_handle_if_exists();
        let handles = figures
            .into_iter()
            .map(|figure| FigureHandle::from(super::super::super::import_runtime_figure(figure)))
            .collect();
        Ok(ImportedFigureBatch::new(handles, previous))
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = (payloads, request);
        Err(persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: plot-core support is not enabled in this build"),
        ))
    }
}

pub(in super::super) struct ImportedFigureBatch {
    handles: Vec<FigureHandle>,
    previous_current: Option<FigureHandle>,
    committed: bool,
}

impl ImportedFigureBatch {
    #[cfg(feature = "plot-core")]
    fn new(handles: Vec<FigureHandle>, previous_current: Option<FigureHandle>) -> Self {
        Self {
            handles,
            previous_current,
            committed: false,
        }
    }

    pub(in super::super) fn handles(&self) -> &[FigureHandle] {
        &self.handles
    }

    pub(in super::super) fn commit(mut self) -> Vec<FigureHandle> {
        self.committed = true;
        std::mem::take(&mut self.handles)
    }
}

impl Drop for ImportedFigureBatch {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for handle in self.handles.drain(..).rev() {
            let _ = super::super::super::state::close_figure(Some(handle));
        }
        if let Some(previous) = self.previous_current {
            let _ = super::super::super::state::select_current_figure_if_exists(previous);
        }
    }
}

#[cfg(feature = "plot-core")]
pub(in super::super) async fn prepare_figure(
    bytes: &[u8],
    builtin: &'static str,
) -> BuiltinResult<runmat_plot::plots::Figure> {
    let scene = crate::replay::import_figure_scene_payload_async(bytes)
        .await
        .map_err(|source| {
            persistence_error_with_source(
                &ERROR_IMPORT,
                builtin,
                format!("{builtin}: unable to decode figure scene"),
                source,
            )
        })?;
    scene.into_figure().map_err(|message| {
        persistence_error(
            &ERROR_IMPORT,
            builtin,
            format!("{builtin}: invalid figure scene content: {message}"),
        )
    })
}
