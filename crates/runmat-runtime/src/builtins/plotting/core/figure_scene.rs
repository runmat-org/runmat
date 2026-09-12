use super::import_registry::{register_imported_figure, take_imported_figure};
use super::state::{clone_figure, import_figure, FigureHandle};

#[cfg(feature = "plot-core")]
pub async fn export_figure_scene(handle: FigureHandle) -> crate::BuiltinResult<Option<Vec<u8>>> {
    export_figure_scene_with_policy(
        handle,
        runmat_plot::event::resolve_scene_export_policy(Some(
            super::perf::scene_export_budget_bytes(),
        )),
    )
    .await
}

#[cfg(feature = "plot-core")]
pub async fn export_figure_scene_with_policy(
    handle: FigureHandle,
    policy: runmat_plot::event::SceneExportPolicy,
) -> crate::BuiltinResult<Option<Vec<u8>>> {
    let Some(figure) = clone_figure(handle) else {
        return Ok(None);
    };
    let scene = runmat_plot::event::FigureScene::capture_for_export(&figure, policy)
        .await
        .map_err(|err| {
            crate::replay_error_with_source(
                crate::ReplayErrorKind::ExportRejected,
                "invalid figure scene content",
                err,
            )
        })?;
    crate::replay::export_figure_scene_payload_with_limits(
        &scene,
        crate::replay::limits::ReplayLimits {
            max_scene_payload_bytes: policy.max_scene_bytes,
            ..crate::replay::limits::ReplayLimits::default()
        },
    )
    .map(Some)
}

#[cfg(feature = "plot-core")]
pub fn import_figure_scene(bytes: &[u8]) -> crate::BuiltinResult<Option<FigureHandle>> {
    let scene = crate::replay::import_figure_scene_payload(bytes)?;
    let figure = scene.into_figure().map_err(|err| {
        crate::replay_error_with_source(
            crate::ReplayErrorKind::ImportRejected,
            "invalid figure scene content",
            std::io::Error::new(std::io::ErrorKind::InvalidData, err),
        )
    })?;
    let handle = import_figure(figure);
    register_imported_figure(handle.as_u32());
    Ok(Some(handle))
}

#[cfg(feature = "plot-core")]
pub async fn import_figure_scene_async(bytes: &[u8]) -> crate::BuiltinResult<Option<FigureHandle>> {
    let scene = crate::replay::import_figure_scene_payload_async(bytes).await?;
    let figure = scene.into_figure().map_err(|err| {
        crate::replay_error_with_source(
            crate::ReplayErrorKind::ImportRejected,
            "invalid figure scene content",
            std::io::Error::new(std::io::ErrorKind::InvalidData, err),
        )
    })?;
    let handle = import_figure(figure);
    register_imported_figure(handle.as_u32());
    Ok(Some(handle))
}

#[cfg(feature = "plot-core")]
pub async fn import_figure_scene_from_path_async(
    path: &str,
) -> crate::BuiltinResult<Option<FigureHandle>> {
    let bytes = runmat_filesystem::read_async(path).await.map_err(|err| {
        crate::replay_error_with_source(
            crate::ReplayErrorKind::ImportRejected,
            format!("failed to read figure scene payload '{path}'"),
            err,
        )
    })?;
    import_figure_scene_async(&bytes).await
}

pub fn present_figure_on_surface(surface_id: u32, handle: u32) -> crate::BuiltinResult<()> {
    super::web::present_figure_on_surface(surface_id, handle)?;
    if take_imported_figure(handle) {
        let _ = super::web::reset_surface_camera(surface_id);
    }
    Ok(())
}

#[cfg(feature = "plot-core")]
pub fn import_runtime_figure(figure: runmat_plot::plots::Figure) -> u32 {
    let handle = import_figure(figure);
    register_imported_figure(handle.as_u32());
    handle.as_u32()
}
