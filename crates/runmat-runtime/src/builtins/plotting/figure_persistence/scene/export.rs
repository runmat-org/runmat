use super::*;

pub(in super::super) async fn save_figures(
    runtime: Option<&crate::context::RuntimeContext>,
    selected: &FigureFsPath,
    figures: &[FigureHandle],
    profile: FigureArchiveProfile,
    builtin: &'static str,
) -> BuiltinResult<()> {
    if figures.is_empty() {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: at least one figure is required"),
        ));
    }
    if figures.len() > MAX_FIGURE_BUNDLE_COUNT {
        return Err(persistence_error(
            &ERROR_INVALID_ARGUMENT,
            builtin,
            format!("{builtin}: too many figures to save in one FIG file"),
        ));
    }
    #[cfg(not(feature = "plot-core"))]
    {
        let _ = (runtime, selected, figures, profile);
        return Err(persistence_error(
            &ERROR_RENDER,
            builtin,
            format!("{builtin}: plot-core support is not enabled in this build"),
        ));
    }
    #[cfg(feature = "plot-core")]
    {
        let snapshot = capture_figure_persistence_snapshot(
            figures,
            MAX_FIGURE_BUNDLE_COUNT,
            MAX_FIGURE_ARCHIVE_BYTES,
        )
        .map_err(|source| {
            persistence_error_with_source(
                &ERROR_RENDER,
                builtin,
                format!("{builtin}: unable to capture a bounded figure snapshot"),
                source,
            )
        })?;
        let mut payloads = Vec::with_capacity(snapshot.entries.len());
        for entry in snapshot.entries {
            payloads.push(export_figure_snapshot(entry.figure.as_ref(), builtin).await?);
        }
        let bytes = encode_figure_bundle(runtime, &payloads, profile, builtin).await?;
        write_bytes(runtime, selected, &bytes, builtin).await
    }
}

#[cfg(feature = "plot-core")]
pub(in super::super) async fn export_figure_snapshot(
    figure: &runmat_plot::plots::Figure,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    let policy = runmat_plot::event::resolve_scene_export_policy(Some(
        super::super::super::perf::scene_export_budget_bytes(),
    ));
    let scene = runmat_plot::event::FigureScene::capture_for_export_with_cause(figure, policy)
        .await
        .map_err(|source| {
            persistence_error_with_source(
                &ERROR_RENDER,
                builtin,
                format!("{builtin}: unable to export figure scene"),
                source,
            )
        })?;
    crate::replay::export_figure_scene_payload_with_limits(
        &scene,
        ReplayLimits {
            max_scene_payload_bytes: policy.max_scene_bytes,
            ..ReplayLimits::default()
        },
    )
    .map_err(|source| {
        persistence_error_with_source(
            &ERROR_RENDER,
            builtin,
            format!("{builtin}: unable to encode figure scene"),
            source,
        )
    })
}

#[cfg(feature = "plot-core")]
pub(in super::super) async fn render_png(
    handle: FigureHandle,
    width: u32,
    height: u32,
    builtin: &'static str,
) -> BuiltinResult<Vec<u8>> {
    super::super::super::render_figure_snapshot(handle, width, height, None)
        .await
        .map_err(|source| {
            persistence_error_with_source(
                &ERROR_RENDER,
                builtin,
                format!("{builtin}: unable to render PNG"),
                source,
            )
        })
}

#[cfg(not(feature = "plot-core"))]
pub(in super::super) async fn render_png(
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
