//! MATLAB-compatible `drawnow` builtin.

use runmat_macros::runtime_builtin;

use crate::BuiltinResult;

/// Flush pending figure updates to any bound plot surfaces.
///
/// On Web/WASM/RunMat Desktop, this presents the current figure revision to any bound surfaces and yields so
/// the browser can process rendering work. On other targets, this is a no-op.
#[runtime_builtin(
    name = "drawnow",
    category = "plotting",
    summary = "Flush pending graphics updates.",
    keywords = "drawnow,graphics,flush,plot",
    sink = true,
    suppress_auto_output = true,
    builtin_path = "crate::builtins::plotting::drawnow"
)]
pub async fn drawnow_builtin() -> BuiltinResult<String> {
    #[cfg(all(target_arch = "wasm32", feature = "plot-web"))]
    {
        use crate::builtins::plotting;
        let handle = plotting::current_figure_handle();
        plotting::render_current_scene(handle.as_u32()).map_err(|e| {
            crate::build_runtime_error(format!("drawnow: {e}"))
                .with_builtin("drawnow")
                .build()
        })?;
        return Ok("drawnow: presented".to_string());
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "plot-web")))]
    {
        Ok("drawnow: ok".to_string())
    }
}