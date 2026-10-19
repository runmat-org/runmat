use anyhow::Result;
use serde::Serialize;
use std::path::Path;
mod camera;
mod profiles;
pub(super) use profiles::CaptureProfile;

#[derive(Serialize)]
pub(super) struct View {
    pub id: &'static str,
    pub label: String,
    pub field: Option<String>,
    pub scalar_scale: f64,
    pub unit: &'static str,
    pub scene: String,
    pub poster: String,
    pub deformation_scale: f64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub initial_camera: runmat_runtime::builtins::plotting::PlotSurfaceCameraState,
}

pub(super) async fn capture(
    run_id: &str,
    output: &Path,
    profile: CaptureProfile,
    results: &runmat_runtime::analysis::AnalysisResultsData,
) -> Result<Vec<View>> {
    use runmat_runtime::builtins::plotting;
    let mut captured = Vec::new();
    for spec in profile.views() {
        let profiles::ViewSpec {
            id,
            label,
            field,
            unit,
            shape,
            relative,
        } = spec;
        let display_label = profiles::display_label(id, label, results)?;
        let (mut scene, scale) = runmat_runtime::analysis::field_scene::build_scene(
            run_id,
            field.as_deref(),
            shape,
            1.0,
            true,
        )
        .map_err(anyhow::Error::msg)?;
        let mut scalar_scale = 1.0;
        if let Some(legend) = scene
            .overlay
            .as_mut()
            .and_then(|overlay| overlay.scalar_legend.as_mut())
        {
            legend.label = label.into();
            if relative {
                anyhow::ensure!(legend.maximum > 0.0, "mode shape has no nonzero amplitude");
                scalar_scale = 1.0 / legend.maximum;
                legend.minimum *= scalar_scale;
                legend.maximum = 1.0;
                legend.unit = unit.into();
            }
        }
        let legend = scene
            .overlay
            .as_ref()
            .and_then(|overlay| overlay.scalar_legend.as_ref());
        let minimum = legend.map(|legend| legend.minimum);
        let maximum = legend.map(|legend| legend.maximum);
        captured.push((
            id,
            display_label,
            field,
            unit,
            scale,
            minimum,
            maximum,
            scalar_scale,
            scene,
        ));
    }
    let camera = camera::shared_camera(captured.iter().map(|view| view.8.bounds));
    let mut result = Vec::new();
    for (id, label, field, unit, scale, minimum, maximum, scalar_scale, scene) in captured {
        let bytes = runmat_runtime::replay::export_figure_scene_payload(
            &runmat_plot::event::FigureScene::from_geometry_scene(&scene),
        )?;
        // Render the persisted representation: the poster and live view must share
        // the same geometry, field colours, edges, and legend after cold decoding.
        let handle = plotting::import_geometry_scene_payload(&bytes)?
            .ok_or_else(|| anyhow::anyhow!("snapshot scene did not contain geometry"))?;
        let poster = plotting::render_geometry_scene_snapshot_with_camera_state(
            handle,
            1200,
            750,
            camera.clone(),
        )
        .await?;
        plotting::close_geometry_scene(handle);
        let scene_path = format!("{id}.scene.json");
        let poster_path = format!("{id}.png");
        std::fs::write(output.join(&scene_path), bytes)?;
        std::fs::write(output.join(&poster_path), poster)?;
        result.push(View {
            id,
            label,
            field,
            scalar_scale,
            unit,
            scene: scene_path,
            poster: poster_path,
            deformation_scale: scale,
            minimum,
            maximum,
            initial_camera: camera.clone(),
        });
    }
    Ok(result)
}
