//! Renders a saved figure scene (the payload the Desktop's model canvas imports) to a PNG on the
//! native GPU, so CAD pictures can be checked without a browser.
//!
//! ```text
//! cargo run -p runmat-plot --example render_scene -- <scene.json> <out.png>
//!     [--presentation <presentation.json>] [--size <w>x<h>] [--dark]
//! ```
//!
//! The presentation file is a `GeometryScenePresentation` in its camelCase JSON form (the
//! object the Desktop sends to `setGeometryScenePresentation`).
use runmat_plot::{
    event::FigureScene,
    export::native_surface::{create_headless_context, encode_png_bytes},
    geometry_scene::GeometryScenePresentation,
    styling::{PlotThemeConfig, ThemeVariant},
};

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct Payload {
    figure: FigureScene,
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let (mut presentation, mut size, mut dark) = (None, (1600_u32, 1000_u32), false);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--presentation" => {
                i += 1;
                let text = std::fs::read_to_string(&args[i]).map_err(|e| e.to_string())?;
                presentation = Some(
                    serde_json::from_str::<GeometryScenePresentation>(&text)
                        .map_err(|e| format!("presentation: {e}"))?,
                );
            }
            "--size" => {
                i += 1;
                let (w, h) = args[i].split_once('x').ok_or("--size takes <w>x<h>")?;
                size = (
                    w.parse().map_err(|_| "bad width")?,
                    h.parse().map_err(|_| "bad height")?,
                );
            }
            "--dark" => dark = true,
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    let [scene_path, out_path] = positional.as_slice() else {
        return Err("usage: render_scene <scene.json> <out.png> [--presentation p.json] [--size WxH] [--dark]".into());
    };
    let bytes = std::fs::read(scene_path).map_err(|e| e.to_string())?;
    let payload: Payload = serde_json::from_slice(&bytes).map_err(|e| format!("scene: {e}"))?;
    let scene = payload.figure.into_geometry_scene("render-scene", 1)?;
    let rgba = futures::executor::block_on(async {
        let mut context = create_headless_context(size.0, size.1).await?;
        context.set_theme_config(PlotThemeConfig {
            variant: if dark {
                ThemeVariant::ModernDark
            } else {
                ThemeVariant::ClassicLight
            },
            ..Default::default()
        });
        context
            .render_geometry_scene_to_rgba_with_presentation(&scene, None, presentation.as_ref())
            .await
    })?;
    std::fs::write(out_path, encode_png_bytes(size.0, size.1, &rgba)?)
        .map_err(|e| e.to_string())?;
    println!("wrote {out_path} ({}x{})", size.0, size.1);
    Ok(())
}
