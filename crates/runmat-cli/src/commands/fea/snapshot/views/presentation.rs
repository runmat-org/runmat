//! How a snapshot draws its scenes and posters: translucent context geometry around the
//! result, a caption, the poster size and the camera direction.
use anyhow::{bail, ensure, Context, Result};
use runmat_geometry_core::LengthUnit;
use runmat_plot::{GeometryScene, GeometrySceneChunk};
use serde::Serialize;
use std::path::Path;

/// Chord tolerance of context tessellation, in meters.
const CONTEXT_CHORD_M: f64 = 0.5e-3;
/// Largest merged context chunk.
const CONTEXT_CHUNK_VERTICES: usize = 1 << 20;
/// Default surface opacity of context geometry.
const CONTEXT_OPACITY: f32 = 0.12;
/// Context surfaces: a light neutral slate, unlike any colormap colour of a result.
const CONTEXT_FACE_RGB: [f32; 3] = [0.74, 0.79, 0.86];
const CONTEXT_EDGE_RGB: [f32; 3] = [0.64, 0.75, 0.90];
/// Posters are drawn with text at this many device pixels per point for each 1200 px of
/// width, so a wide poster's labels keep their size relative to the picture.
const POSTER_REFERENCE_WIDTH: f32 = 1200.0;
/// A poster's legend labels are at least this share of its height, so they stay readable on a
/// card a few hundred pixels tall.
pub(in crate::commands::fea) const LEGEND_LABEL_SHARE: f32 = 0.025;

/// One context file drawn in every view, as the view records it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(in crate::commands::fea) struct ContextRecord {
    pub file: String,
    pub sha256: String,
    /// Surface opacity (0 to 1).
    pub opacity: f32,
    /// Always "geometry": context carries no result.
    pub shows: &'static str,
}

struct ContextGeometry {
    record: ContextRecord,
    /// One surface or edge chunk per CAD part, as imported; merged when drawn.
    parts: Vec<GeometrySceneChunk>,
}

/// One poster drawn of each view: its theme and the suffix its file name takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::commands::fea) struct PosterTheme {
    pub variant: runmat_plot::styling::ThemeVariant,
    /// Inserted before `.png`: empty for the main poster, `.light` for the light one beside it.
    pub suffix: &'static str,
}

/// What the poster camera frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::commands::fea) enum Frame {
    /// Every view's whole scene: the result and its context geometry.
    All,
    /// The solved mesh alone; context geometry may reach past the frame.
    Result,
}

pub(in crate::commands::fea) struct Presentation {
    context: Vec<ContextGeometry>,
    /// Every context part merged into a few large chunks, drawn in every view's scene; empty
    /// when the context is cropped to the frame.
    merged_context: Vec<GeometrySceneChunk>,
    pub caption: Option<String>,
    pub poster_width: u32,
    pub poster_height: u32,
    /// MATLAB `view(az, el)` in degrees, when not the standard poster view.
    pub view_angle: Option<(f32, f32)>,
    pub frame: Frame,
    /// The camera frames this multiple of the framed bounds, about their center.
    pub frame_padding: f32,
    /// The framed bounds span this share of the poster width, when set; otherwise they fill
    /// the poster less its margin.
    pub frame_width: Option<f32>,
    /// Draw only the context parts that lie wholly inside the poster frame.
    pub crop_context: bool,
    pub show_grid: bool,
    pub show_triad: bool,
    /// Text and overlay size relative to the standard poster.
    pub text_scale: f32,
    /// Section-view plane; the default plane when absent.
    pub section: Option<runmat_runtime::analysis::field_scene::volume::SectionPlaneChoice>,
    /// Largest scene file one view may write, in bytes.
    pub max_scene_bytes: usize,
    /// The posters drawn of each view; scenes carry no theme.
    pub themes: Vec<PosterTheme>,
}

impl Default for Presentation {
    fn default() -> Self {
        Self {
            context: Vec::new(),
            merged_context: Vec::new(),
            caption: None,
            poster_width: 1200,
            poster_height: 750,
            view_angle: None,
            frame: Frame::All,
            frame_padding: 1.0,
            frame_width: None,
            crop_context: false,
            show_grid: true,
            show_triad: true,
            text_scale: 1.0,
            section: None,
            max_scene_bytes: 16 * 1024 * 1024,
            themes: parse_theme("both").expect("the default poster themes parse"),
        }
    }
}

impl Presentation {
    /// Parse the arguments and load every context file once.
    pub fn from_args(args: &crate::cli::SnapshotPresentationArgs) -> Result<Self> {
        let (poster_width, poster_height) = parse_size(&args.poster_size)?;
        let units = parse_units(&args.context_units)?;
        let context = args
            .context
            .iter()
            .map(|spec| {
                let (file, opacity) = parse_context(spec)?;
                load_context(Path::new(file), opacity, units)
            })
            .collect::<Result<Vec<_>>>()?;
        let frame = parse_frame(&args.frame)?;
        let crop_context = args.crop_context;
        ensure!(
            !crop_context || frame == Frame::Result,
            "--crop-context needs --frame result: the frame must not depend on the context"
        );
        let merged_context = if crop_context {
            Vec::new()
        } else {
            merge_context(&context, |_| true)?
        };
        Ok(Self {
            context,
            merged_context,
            caption: args.caption.clone(),
            poster_width,
            poster_height,
            view_angle: args.view_angle.as_deref().map(parse_angle).transpose()?,
            frame,
            frame_padding: parse_padding(args.frame_padding)?,
            frame_width: args.frame_width.map(parse_frame_width).transpose()?,
            crop_context,
            show_grid: !args.no_grid,
            show_triad: !args.no_triad,
            text_scale: parse_text_scale(args.text_scale)?,
            section: args.section.as_deref().map(parse_section).transpose()?,
            max_scene_bytes: parse_scene_budget(args.max_scene_bytes)?,
            themes: parse_theme(&args.theme)?,
        })
    }

    /// Add the context geometry, caption and display settings to one view's scene.
    pub fn dress(&self, scene: &mut GeometryScene) {
        scene.show_grid &= self.show_grid;
        scene.show_orientation_triad &= self.show_triad;
        if !self.merged_context.is_empty() {
            scene.append_chunks(self.merged_context.iter().cloned());
        }
        if let Some(caption) = &self.caption {
            scene.title = Some(caption.clone());
        }
    }

    pub fn records(&self) -> Vec<ContextRecord> {
        self.context
            .iter()
            .map(|context| context.record.clone())
            .collect()
    }

    /// Text density of the posters: at least one pixel per point for each 1200 px of width,
    /// and enough that the legend's labels are LEGEND_LABEL_SHARE of the poster height; then
    /// `text_scale` times that.
    pub fn pixels_per_point(&self) -> f32 {
        let by_width = (self.poster_width as f32 / POSTER_REFERENCE_WIDTH).max(1.0);
        let by_legend = runmat_plot::scalar_legend::export_pixels_per_point_for_label(
            LEGEND_LABEL_SHARE * self.poster_height as f32,
        );
        by_width.max(by_legend) * self.text_scale
    }

    /// The legend's footprint in the poster, from its top-right corner, as shares of the
    /// poster's width and height.
    pub fn legend_footprint(&self, has_note: bool) -> runmat_plot::scalar_legend::LegendFootprint {
        runmat_plot::scalar_legend::exported_footprint(
            self.poster_width,
            self.poster_height,
            self.pixels_per_point(),
            has_note,
        )
    }

    /// The app's plot theme in `variant`: its dark or light tokens.
    pub fn theme_config(
        variant: runmat_plot::styling::ThemeVariant,
    ) -> runmat_plot::styling::PlotThemeConfig {
        runmat_plot::styling::PlotThemeConfig {
            variant,
            ..Default::default()
        }
    }

    /// The context parts `keep` accepts (by their bounds), merged for drawing.
    pub fn cropped_context(
        &self,
        keep: impl Fn(&runmat_plot::core::BoundingBox) -> bool,
    ) -> Result<Vec<GeometrySceneChunk>> {
        merge_context(&self.context, keep)
    }

    /// Every context part, with its bounds.
    pub fn context_parts(&self) -> impl Iterator<Item = &GeometrySceneChunk> {
        self.context.iter().flat_map(|context| context.parts.iter())
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.poster_width as f32 / self.poster_height as f32
    }
}

fn load_context(
    path: &Path,
    opacity: f32,
    units: LengthUnit,
) -> Result<ContextGeometry> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("cannot read context geometry {}", path.display()))?;
    let name = path.display().to_string();
    // Context is drawn translucent at poster scale: a 0.5 mm chord keeps curved parts smooth
    // and the scene a few megabytes (the import default, 0.01 source units, made the battery
    // pack a 68.6 MB scene).
    let options = runmat_geometry_io::GeometryImportOptions {
        units,
        source_units: Some(units),
        tessellation_profile: runmat_geometry_core::TessellationProfile {
            profile_id: "snapshot-context-v1".into(),
            chord_tolerance: Some(CONTEXT_CHORD_M / units.meters_per_unit()),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut asset = runmat_runtime::geometry::geometry_load_with_options_op(
        &name,
        &bytes,
        options,
        runmat_runtime::operations::OperationContext::new(None, None),
    )
    .map_err(|error| anyhow::anyhow!("context geometry {name}: {}", error.message))?
    .data;
    // Preview coordinates stay in the file's unit; results are drawn in meters.
    let meters_per_unit = asset
        .coordinate_units
        .with_context(|| format!("context geometry {name} has no resolved length unit"))?
        .meters_per_unit();
    if meters_per_unit != 1.0 {
        for mesh in &mut asset.surface_meshes {
            for vertex in &mut mesh.vertices {
                for coordinate in vertex.iter_mut() {
                    *coordinate *= meters_per_unit;
                }
            }
        }
    }
    let scene = runmat_runtime::geometry::geometry_preview_scene(
        &asset,
        name.clone(),
        runmat_runtime::geometry::GeometryPreviewSceneOptions::default(),
    )
    .map_err(|error| anyhow::anyhow!("context geometry {name}: {error}"))?;
    // Faint edges: an assembly of thousands of parts drawn with strong edges reads as a
    // wireframe and buries the result.
    let edge_opacity = (0.03 + 0.6 * opacity).min(0.9);
    let [r, g, b] = CONTEXT_FACE_RGB;
    let [er, eg, eb] = CONTEXT_EDGE_RGB;
    let parts: Vec<_> = scene
        .chunks
        .into_iter()
        .map(|chunk| chunk.into_context([r, g, b, opacity], [er, eg, eb, edge_opacity]))
        .collect();
    log::info!(
        "context geometry {name}: {} triangles; peak resident {} MiB",
        asset
            .surface_meshes
            .iter()
            .map(|mesh| mesh.triangles.len())
            .sum::<usize>(),
        super::peak_resident_mib()
    );
    Ok(ContextGeometry {
        record: ContextRecord {
            file: name,
            sha256: asset.source.sha256.clone(),
            opacity,
            shows: "geometry",
        },
        parts,
    })
}

/// Merge the context parts `keep` accepts into a few large chunks: an assembly of thousands of
/// parts is drawn as one piece.
fn merge_context(
    context: &[ContextGeometry],
    keep: impl Fn(&runmat_plot::core::BoundingBox) -> bool,
) -> Result<Vec<GeometrySceneChunk>> {
    let mut chunks = Vec::new();
    for (index, context) in context.iter().enumerate() {
        let (surfaces, edges): (Vec<_>, Vec<_>) = context
            .parts
            .iter()
            .filter(|part| keep(&part.bounds))
            .cloned()
            .partition(|chunk| {
                chunk.render_data.pipeline_type == runmat_plot::core::PipelineType::Triangles
            });
        for (kind, group) in [("surface", surfaces), ("edges", edges)] {
            let mut batch = Vec::new();
            let mut batch_vertices = 0;
            let mut flush = |batch: &mut Vec<GeometrySceneChunk>| -> Result<()> {
                if batch.is_empty() {
                    return Ok(());
                }
                // Unique beside the result's chunks and the other context files' chunks.
                let id = format!("context{index}:{kind}:{}", chunks.len());
                chunks.push(
                    GeometrySceneChunk::merged(id, std::mem::take(batch))
                        .context("context geometry chunks do not merge")?,
                );
                Ok(())
            };
            for chunk in group {
                if batch_vertices + chunk.vertices.len() > CONTEXT_CHUNK_VERTICES {
                    flush(&mut batch)?;
                    batch_vertices = 0;
                }
                batch_vertices += chunk.vertices.len();
                batch.push(chunk);
            }
            flush(&mut batch)?;
        }
    }
    Ok(chunks)
}

/// `FILE` or `FILE@OPACITY`.
fn parse_context(spec: &str) -> Result<(&str, f32)> {
    if let Some((file, opacity)) = spec.rsplit_once('@') {
        if let Ok(opacity) = opacity.parse::<f32>() {
            ensure!(
                opacity > 0.0 && opacity <= 1.0,
                "context opacity must be in (0, 1], got {opacity} in '{spec}'"
            );
            ensure!(!file.is_empty(), "context file missing in '{spec}'");
            return Ok((file, opacity));
        }
    }
    Ok((spec, CONTEXT_OPACITY))
}

fn parse_size(text: &str) -> Result<(u32, u32)> {
    let (width, height) = text
        .split_once(['x', 'X'])
        .with_context(|| format!("poster size must be WIDTHxHEIGHT, got '{text}'"))?;
    let (width, height): (u32, u32) = (width.trim().parse()?, height.trim().parse()?);
    ensure!(
        (64..=8192).contains(&width) && (64..=8192).contains(&height),
        "poster size must be 64 to 8192 px on each side, got '{text}'"
    );
    Ok((width, height))
}

fn parse_angle(text: &str) -> Result<(f32, f32)> {
    let (azimuth, elevation) = text
        .split_once(',')
        .with_context(|| format!("view angle must be AZ,EL in degrees, got '{text}'"))?;
    let (azimuth, elevation): (f32, f32) = (azimuth.trim().parse()?, elevation.trim().parse()?);
    ensure!(
        azimuth.is_finite() && (-90.0..=90.0).contains(&elevation),
        "view elevation must be -90 to 90 degrees, got '{text}'"
    );
    Ok((azimuth, elevation))
}

fn parse_section(
    text: &str,
) -> Result<runmat_runtime::analysis::field_scene::volume::SectionPlaneChoice> {
    let (axis, fraction) = text.split_once('@').unwrap_or((text, "0.5"));
    let axis = match axis.trim().to_ascii_lowercase().as_str() {
        "x" => 0,
        "y" => 1,
        "z" => 2,
        other => bail!("section axis must be x, y or z, got '{other}'"),
    };
    let fraction: f64 = fraction
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("section fraction must be a number, got '{fraction}'"))?;
    ensure!(
        (0.0..=1.0).contains(&fraction),
        "section fraction must be 0 to 1, got {fraction}"
    );
    Ok(runmat_runtime::analysis::field_scene::volume::SectionPlaneChoice { axis, fraction })
}

fn parse_frame(text: &str) -> Result<Frame> {
    Ok(match text.to_ascii_lowercase().as_str() {
        "all" => Frame::All,
        "result" => Frame::Result,
        other => bail!("poster frame must be 'all' or 'result', got '{other}'"),
    })
}

fn parse_padding(padding: f32) -> Result<f32> {
    ensure!(
        padding.is_finite() && (1.0..=50.0).contains(&padding),
        "frame padding must be 1 to 50, got {padding}"
    );
    Ok(padding)
}

fn parse_scene_budget(bytes: usize) -> Result<usize> {
    anyhow::ensure!(bytes > 0, "--max-scene-bytes must be positive");
    Ok(bytes)
}

/// `both` (the default): a dark poster and a light one beside it, `<view>.light.png`. `dark` or
/// `light`: that poster alone, `<view>.png`.
fn parse_theme(text: &str) -> Result<Vec<PosterTheme>> {
    use runmat_plot::styling::ThemeVariant;
    let (dark, light) = (ThemeVariant::ModernDark, ThemeVariant::ClassicLight);
    let main = |variant| PosterTheme {
        variant,
        suffix: "",
    };
    Ok(match text {
        "both" => vec![
            main(dark),
            PosterTheme {
                variant: light,
                suffix: ".light",
            },
        ],
        "dark" => vec![main(dark)],
        "light" => vec![main(light)],
        _ => bail!("--theme must be both, dark or light, not '{text}'"),
    })
}

fn parse_frame_width(share: f32) -> Result<f32> {
    ensure!(
        share.is_finite() && (0.05..=1.0 - 2.0 * super::camera::FRAME_MARGIN).contains(&share),
        "frame width must be 0.05 to {} of the poster width, got {share}",
        1.0 - 2.0 * super::camera::FRAME_MARGIN
    );
    Ok(share)
}

fn parse_text_scale(scale: f32) -> Result<f32> {
    ensure!(
        scale.is_finite() && (0.5..=4.0).contains(&scale),
        "text scale must be 0.5 to 4, got {scale}"
    );
    Ok(scale)
}

fn parse_units(text: &str) -> Result<LengthUnit> {
    Ok(match text.to_ascii_lowercase().as_str() {
        "m" | "meter" | "meters" | "metre" | "metres" => LengthUnit::Meter,
        "mm" | "millimeter" | "millimeters" | "millimetre" | "millimetres" => {
            LengthUnit::Millimeter
        }
        "in" | "inch" | "inches" => LengthUnit::Inch,
        other => bail!("unsupported context unit '{other}'"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_specs_take_an_optional_opacity() {
        assert_eq!(
            parse_context("pack.step").unwrap(),
            ("pack.step", CONTEXT_OPACITY)
        );
        assert_eq!(parse_context("lid.step@0.3").unwrap(), ("lid.step", 0.3));
        // An '@' in the path without a number is part of the path.
        assert_eq!(
            parse_context("runs/a@b/pack.step").unwrap(),
            ("runs/a@b/pack.step", CONTEXT_OPACITY)
        );
        assert!(parse_context("lid.step@0").is_err());
        assert!(parse_context("lid.step@1.5").is_err());
    }

    #[test]
    fn poster_size_angle_and_units_parse() {
        assert_eq!(parse_size("2400x1500").unwrap(), (2400, 1500));
        assert!(parse_size("2400").is_err());
        assert!(parse_size("10x10").is_err());
        assert_eq!(parse_angle("-37.5,30").unwrap(), (-37.5, 30.0));
        assert!(parse_angle("0,120").is_err());
        assert_eq!(parse_units("mm").unwrap(), LengthUnit::Millimeter);
        assert!(parse_units("furlong").is_err());
    }

    /// The legend size rule: at every poster size, the legend's labels are at least 2.5% of
    /// the poster height, and its footprint is a fixed share of the frame at a fixed aspect.
    #[test]
    fn legend_labels_are_sized_from_the_frame_height() {
        for (width, height) in [
            (1200, 750),
            (1600, 1000),
            (2400, 1500),
            (800, 600),
            (640, 400),
            (1200, 1200),
            (3000, 750),
        ] {
            let poster = Presentation {
                poster_width: width,
                poster_height: height,
                ..Default::default()
            };
            for has_note in [false, true] {
                let footprint = poster.legend_footprint(has_note);
                assert!(
                    footprint.label_px >= LEGEND_LABEL_SHARE * height as f32 * 0.9999,
                    "{width}x{height}: labels {} px",
                    footprint.label_px
                );
                assert!(
                    footprint.width_share < 0.5 && footprint.height_share < 0.4,
                    "{width}x{height}: {footprint:?}"
                );
            }
        }
        // A 16:10 poster's legend covers the same share of the frame at any size.
        let share = |width, height| {
            Presentation {
                poster_width: width,
                poster_height: height,
                ..Default::default()
            }
            .legend_footprint(false)
        };
        let (small, large) = (share(1200, 750), share(2400, 1500));
        assert!((small.width_share - large.width_share).abs() < 1e-4);
        assert!((small.height_share - large.height_share).abs() < 1e-4);
        // A very wide poster keeps at least its width's text density.
        assert_eq!(share(3000, 750).label_px, 11.0 * 1.25 * 2.5);
    }

    #[test]
    fn text_density_follows_the_poster_size() {
        let wide = Presentation {
            poster_width: 2400,
            poster_height: 1500,
            ..Default::default()
        };
        // 2.5% of 1500 px over 11 pt at the exporter's 1.25 font scale.
        assert!((wide.pixels_per_point() - 37.5 / 13.75).abs() < 1e-5);
        assert!((Presentation::default().pixels_per_point() - 18.75 / 13.75).abs() < 1e-5);
        assert_eq!(wide.aspect_ratio(), 1.6);
    }

    #[test]
    fn posters_come_in_dark_and_light_by_default() {
        use runmat_plot::styling::ThemeVariant;
        let both = &Presentation::default().themes;
        assert_eq!(both.len(), 2);
        assert_eq!(
            (both[0].variant, both[0].suffix),
            (ThemeVariant::ModernDark, "")
        );
        assert_eq!(
            (both[1].variant, both[1].suffix),
            (ThemeVariant::ClassicLight, ".light")
        );
        let light = parse_theme("light").unwrap();
        assert_eq!(
            (light[0].variant, light[0].suffix),
            (ThemeVariant::ClassicLight, "")
        );
        assert!(parse_theme("sepia").is_err());
    }

    #[test]
    fn dressing_adds_context_and_caption() {
        let triangle = |id: &str| {
            GeometrySceneChunk::indexed_triangles(
                id,
                (0..3)
                    .map(|i| {
                        runmat_plot::geometry_scene_vertex(
                            [i as f32, (i * i) as f32, 0.0],
                            [0.9, 0.1, 0.1, 1.0],
                            [0.0, 0.0, 1.0],
                        )
                    })
                    .collect(),
                vec![0, 1, 2],
                runmat_plot::cad_default_material(),
            )
        };
        let presentation = Presentation {
            context: vec![ContextGeometry {
                record: ContextRecord {
                    file: "pack.step".into(),
                    sha256: "00".into(),
                    opacity: 0.12,
                    shows: "geometry",
                },
                parts: vec![triangle("context0:a")
                    .into_context([0.7, 0.8, 0.9, 0.12], [0.6, 0.7, 0.9, 0.33])],
            }],
            caption: Some("Steady thermal on the cells".into()),
            ..Default::default()
        };
        let presentation = Presentation {
            merged_context: merge_context(&presentation.context, |_| true).unwrap(),
            ..presentation
        };
        let mut scene = GeometryScene::new("result", 1, vec![triangle("result")]);
        presentation.dress(&mut scene);
        assert_eq!(scene.chunks.len(), 2);
        assert_eq!(scene.title.as_deref(), Some("Steady thermal on the cells"));
        assert_eq!(
            scene.chunks[0].render_data.material.alpha_mode,
            runmat_plot::core::AlphaMode::Opaque
        );
        assert_eq!(
            scene.chunks[1].render_data.material.alpha_mode,
            runmat_plot::core::AlphaMode::Blend
        );
        assert_eq!(presentation.records()[0].shows, "geometry");
        // The standard presentation keeps the scene's grid and corner triad.
        assert!(scene.show_grid && scene.show_orientation_triad);
    }

    #[test]
    fn hero_dressing_turns_the_grid_and_corner_triad_off() {
        let presentation = Presentation {
            show_grid: false,
            show_triad: false,
            ..Default::default()
        };
        let triangle = GeometrySceneChunk::indexed_triangles(
            "result",
            (0..3)
                .map(|i| {
                    runmat_plot::geometry_scene_vertex(
                        [i as f32, (i * i) as f32, 0.0],
                        [0.9, 0.1, 0.1, 1.0],
                        [0.0, 0.0, 1.0],
                    )
                })
                .collect(),
            vec![0, 1, 2],
            runmat_plot::cad_default_material(),
        );
        let mut scene = GeometryScene::new("result", 1, vec![triangle]);
        presentation.dress(&mut scene);
        assert!(!scene.show_grid);
        assert!(!scene.show_orientation_triad);
        // Both survive the saved scene payload the poster and the live viewer draw.
        let saved = runmat_plot::event::FigureScene::from_geometry_scene(&scene);
        let json = serde_json::to_string(&saved).unwrap();
        let restored: runmat_plot::event::FigureScene = serde_json::from_str(&json).unwrap();
        assert!(!restored.geometry_orientation_triad);
        assert!(!restored.metadata.grid_enabled);
    }

    #[test]
    fn framing_and_text_scale_parse() {
        assert_eq!(parse_frame("all").unwrap(), Frame::All);
        assert_eq!(parse_frame("Result").unwrap(), Frame::Result);
        assert!(parse_frame("module").is_err());
        assert_eq!(parse_padding(1.6).unwrap(), 1.6);
        assert!(parse_padding(0.8).is_err());
        assert!(parse_padding(f32::NAN).is_err());
        assert_eq!(parse_text_scale(2.0).unwrap(), 2.0);
        assert!(parse_text_scale(0.0).is_err());
        let hero = Presentation {
            poster_width: 2400,
            poster_height: 1500,
            text_scale: 1.5,
            ..Default::default()
        };
        assert!((hero.pixels_per_point() - 1.5 * 37.5 / 13.75).abs() < 1e-5);
        assert_eq!(parse_frame_width(0.6).unwrap(), 0.6);
        assert!(parse_frame_width(0.9).is_err());
        assert!(parse_frame_width(0.0).is_err());
    }

    /// Cropping keeps the parts whose bounds the frame accepts and merges them; a part the
    /// frame cuts is left out.
    #[test]
    fn cropped_context_keeps_the_parts_inside_the_frame() {
        let triangle = |id: &str, x: f32| {
            GeometrySceneChunk::indexed_triangles(
                id,
                (0..3)
                    .map(|i| {
                        runmat_plot::geometry_scene_vertex(
                            [x + i as f32, (i * i) as f32, 0.0],
                            [0.7, 0.8, 0.9, 0.12],
                            [0.0, 0.0, 1.0],
                        )
                    })
                    .collect(),
                vec![0, 1, 2],
                runmat_plot::cad_default_material(),
            )
            .into_context([0.7, 0.8, 0.9, 0.12], [0.6, 0.7, 0.9, 0.33])
        };
        let presentation = Presentation {
            context: vec![ContextGeometry {
                record: ContextRecord {
                    file: "pack.step".into(),
                    sha256: "00".into(),
                    opacity: 0.12,
                    shows: "geometry",
                },
                parts: vec![triangle("near", 0.0), triangle("far", 100.0)],
            }],
            crop_context: true,
            ..Default::default()
        };
        assert_eq!(presentation.context_parts().count(), 2);
        let kept = presentation
            .cropped_context(|bounds| bounds.max.x < 10.0)
            .unwrap();
        assert_eq!(kept.len(), 1);
        assert!(kept[0].bounds.max.x < 10.0);
        // A cropped context is not drawn into the scenes; the poster pass adds the kept parts.
        let mut scene = GeometryScene::new("result", 1, vec![triangle("result", 0.0)]);
        presentation.dress(&mut scene);
        assert_eq!(scene.chunks.len(), 1);
    }
}
