//! `design.render`: a part or assembly view through the Desktop's renderer.
//!
//! The model scene ([`crate::scene`]) is the figure scene the model canvas imports; it becomes
//! a `runmat_plot` geometry scene and renders headless on wgpu with the canvas's presentation
//! (display mode, selection). With no GPU adapter, the renderer's CPU rasterizer draws the
//! same figure and the result says `cpu_reference`.

use crate::scene::{
    build_model_scene, camera_for, DisplayMode, FramedCamera, RenderInput, RenderOutput,
};
use crate::{kind_of, text_of, DesignKind, Files, ToolError};
use base64::Engine;
use runmat_plot::core::camera::{Camera, ProjectionType};
use runmat_plot::event::FigureScene;
use runmat_plot::geometry_scene::{GeometrySceneDisplayMode, GeometryScenePresentation};
use runmat_plot::styling::{PlotThemeConfig, ThemeVariant};
use serde_json::Value;

const MAX_PIXELS: u32 = 4096;

/// The display (`{ kind, data }`) of a part or assembly.
pub(crate) fn model_display(path: &str, files: &Files) -> Result<Value, ToolError> {
    let text = text_of(files, path)?;
    match kind_of(path)? {
        DesignKind::Part => {
            let part = runmat_cad_wasm::runtime::compile_part(path, text).map_err(|d| {
                ToolError::new(
                    "not_compiled",
                    format!(
                        "{path} has syntax errors: {}",
                        d.first().map(|d| d.message.as_str()).unwrap_or("")
                    ),
                )
            })?;
            Ok(serde_json::json!({ "kind": "part", "data": part.display }))
        }
        DesignKind::Assembly => {
            let (_, display, _) = runmat_cad_wasm::runtime::compile_assembly(path, text, files)
                .map_err(|d| {
                    ToolError::new(
                        "not_compiled",
                        format!(
                            "{path} has syntax errors: {}",
                            d.first().map(|d| d.message.as_str()).unwrap_or("")
                        ),
                    )
                })?;
            Ok(serde_json::json!({ "kind": "assembly", "data": display }))
        }
        other => Err(ToolError::new(
            "unsupported",
            format!("design.render draws parts and assemblies; {path} is a {other:?} file"),
        )),
    }
}

fn camera(framed: &FramedCamera, aspect: f32) -> Camera {
    let v = |a: [f64; 3]| glam::Vec3::new(a[0] as f32, a[1] as f32, a[2] as f32);
    let mut camera = Camera::new();
    camera.position = v(framed.position);
    camera.target = v(framed.target);
    camera.up = v(framed.up);
    camera.aspect_ratio = aspect;
    camera.axis_scale = glam::Vec3::ONE;
    camera.projection = if framed.projection == "perspective" {
        ProjectionType::Perspective {
            fov: framed.fov as f32,
            near: framed.near as f32,
            far: framed.far as f32,
        }
    } else {
        ProjectionType::Orthographic {
            left: -framed.half_width as f32,
            right: framed.half_width as f32,
            bottom: -framed.half_height as f32,
            top: framed.half_height as f32,
            near: framed.near as f32,
            far: framed.far as f32,
        }
    };
    camera.mark_dirty();
    camera
}

pub fn render(input: &RenderInput, files: &Files) -> Result<RenderOutput, ToolError> {
    let display = model_display(&input.path, files)?;
    let built = build_model_scene(&display);
    let Some(bounds) = built.bounds else {
        return Err(ToolError::new(
            "not_compiled",
            format!(
                "{} has no geometry to draw yet (check it with design.check)",
                input.path
            ),
        ));
    };
    let width = input.view.width.unwrap_or(1280).clamp(16, MAX_PIXELS);
    let height = input.view.height.unwrap_or(800).clamp(16, MAX_PIXELS);
    let aspect = width as f64 / height as f64;
    let framed = camera_for(&input.view, bounds, aspect);
    let cam = camera(&framed, aspect as f32);
    let dark = input.view.theme.as_deref() == Some("dark");
    let theme = PlotThemeConfig {
        variant: if dark {
            ThemeVariant::ModernDark
        } else {
            ThemeVariant::ClassicLight
        },
        ..PlotThemeConfig::default()
    };
    let mut figure = built.figure.clone();
    if dark {
        figure["metadata"]["backgroundRgba"] = serde_json::json!([0.07, 0.08, 0.1, 1.0]);
    }
    let scene_json = figure.clone();
    let figure_scene: FigureScene = serde_json::from_value(scene_json)
        .map_err(|e| ToolError::new("internal", format!("the model scene doesn't decode: {e}")))?;
    let scene = figure_scene
        .clone()
        .into_geometry_scene(format!("design:{}", input.path), 1)
        .map_err(|e| ToolError::new("internal", e))?;
    let mode = input.view.display.unwrap_or_default();
    let mut presentation = GeometryScenePresentation {
        display_mode: match mode {
            DisplayMode::Shaded | DisplayMode::ShadedEdges => GeometrySceneDisplayMode::Shaded,
            DisplayMode::HiddenLines => GeometrySceneDisplayMode::Edges,
            DisplayMode::Wireframe => GeometrySceneDisplayMode::Wireframe,
        },
        edge_overlay_enabled: !matches!(mode, DisplayMode::Shaded),
        ..GeometryScenePresentation::default()
    };
    let mut notes = Vec::new();
    for h in input.view.highlight.iter().flatten() {
        let ids: Vec<String> = built
            .regions
            .iter()
            .filter(|r| crate::name_matches(Some(h), &r.label))
            .map(|r| r.id.clone())
            .collect();
        if ids.is_empty() {
            notes.push(format!("nothing in {} is named like `{h}`", input.path));
        }
        presentation.selected_region_ids.extend(ids);
    }
    presentation.selected_region_id = presentation.selected_region_ids.first().cloned();
    let rendered = pollster::block_on(async {
        let mut context =
            runmat_plot::export::native_surface::create_headless_context(width, height).await?;
        context.set_theme_config(theme.clone());
        context
            .render_geometry_scene_to_rgba_with_presentation(
                &scene,
                Some(&cam),
                Some(&presentation),
            )
            .await
    });
    let (png, renderer) = match rendered {
        Ok(rgba) => (
            runmat_plot::export::cpu_surface::encode_png_bytes(width, height, &rgba)
                .map_err(|e| ToolError::new("internal", e))?,
            "gpu",
        ),
        Err(e) if runmat_plot::export::native_surface::is_headless_gpu_unavailable_error(&e) => {
            notes.push(format!(
                "no GPU adapter on this host ({e}); drawn by the renderer's CPU rasterizer"
            ));
            let figure = figure_scene
                .into_figure()
                .map_err(|e| ToolError::new("internal", e))?;
            let png =
                pollster::block_on(runmat_plot::export::cpu_surface::render_figure_png_bytes(
                    figure,
                    width,
                    height,
                    Some(theme),
                    Some(&cam),
                    None,
                    None,
                ))
                .map_err(|e| ToolError::new("render_unavailable", e))?;
            (png, "cpu_reference")
        }
        Err(e) => return Err(ToolError::new("render_unavailable", e)),
    };
    Ok(RenderOutput {
        path: input.path.clone(),
        mime_type: "image/png".into(),
        png_base64: base64::engine::general_purpose::STANDARD.encode(&png),
        width,
        height,
        renderer: renderer.into(),
        camera: serde_json::to_value(&framed).unwrap_or(Value::Null),
        triangles: built.triangles,
        boxed_occurrences: built.boxed,
        notes,
    })
}
