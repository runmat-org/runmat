//! `design.export` with `format: "fabrication"`: a board's fabrication set as one folder.
//!
//! The circuit runtime makes the text outputs (Gerber X2 and the job file, Excellon, the drill
//! map and fabrication drawing as Gerber, pick-and-place and BOM per variant, IPC-2581C,
//! IPC-D-356A and the ODB++ tree). This module adds the drill map and the fabrication
//! drawing as PDF sheets and archives the ODB++ tree as `odb/<board>-odb.tgz`.

use super::{binary_file, text_file, ExportedFile};
use crate::formats::sheet::{Layer, Shape, Sheet};
use crate::{Files, ToolError};
use runmat_circuit::fab::drawing::{drill_map, fabrication_drawing, Drawing, Mark};
use runmat_circuit::fab::set::FabRequest;
use runmat_circuit::geom::to_mm;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The drawing on a sheet at true size, with a 10 mm margin. The board outline and table
/// rules (pens of 0.1 mm and up) draw on `OUTLINE`, the rest on `THIN`.
pub fn drawing_sheet(d: &Drawing) -> Sheet {
    let ext = d.extent();
    let margin = 10.0;
    let (x0, y0) = (to_mm(ext.min.x) - margin, to_mm(ext.min.y) - margin);
    let at = |p: runmat_circuit::geom::Point| [to_mm(p.x) - x0, to_mm(p.y) - y0];
    let mut sheet = Sheet {
        paper: [
            to_mm(ext.width()) + 2.0 * margin,
            to_mm(ext.height()) + 2.0 * margin,
        ],
        layers: vec![
            Layer {
                name: "OUTLINE".into(),
                color: 7,
                linetype: "CONTINUOUS".into(),
            },
            Layer {
                name: "THIN".into(),
                color: 8,
                linetype: "CONTINUOUS".into(),
            },
        ],
        linetypes: crate::formats::sheet::standard_linetypes(),
        ..Default::default()
    };
    for m in &d.marks {
        match m {
            Mark::Line { points, width } => {
                let layer = if *width >= 100_000 { "OUTLINE" } else { "THIN" };
                let closed = points.len() > 2 && points.first() == points.last();
                let mut pts: Vec<[f64; 2]> = points.iter().map(|p| at(*p)).collect();
                if closed {
                    pts.pop();
                }
                if pts.len() == 2 {
                    sheet.line(layer, pts[0], pts[1]);
                } else {
                    sheet.add(
                        layer,
                        Shape::Polyline {
                            points: pts,
                            closed,
                        },
                    );
                }
            }
            // The sheet's monospace text advances 0.6 of its height; at 1.5 × the cap
            // height it advances as the stroke font the tables were measured with.
            Mark::Text { at: p, size, text } => {
                sheet.text("THIN", at(*p), 1.5 * to_mm(*size), text)
            }
        }
    }
    sheet
}

/// What a fabrication export covered: the board, its layer table, the variants, the
/// settings it ran with and, when asked, the pre-flight checks and the layout with the
/// settings saved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FabricationSummary {
    pub board: String,
    /// The layer table, top to bottom: paste, legend and mask, the stackup, then the bottom's.
    pub layers: Vec<FabricationLayer>,
    /// The board's variants; `null` is the default.
    pub variants: Vec<Option<String>>,
    /// The settings the export ran with (the board's saved ones with the call's over them).
    pub settings: serde_json::Value,
    /// The output folder the settings name, relative to the board's folder.
    pub folder: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preflight: Option<Preflight>,
    /// The board's layout with the settings saved into it (`remember`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<LayoutUpdate>,
    /// The layer `previewLayer` named, drawn as SVG.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<LayerPreview>,
}

/// One layer drawn as its Gerber makes it: the image in `currentColor`, mm with y up.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct LayerPreview {
    pub layer: String,
    pub svg: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FabricationLayer {
    pub name: String,
    /// `copper`, `core`, `prepreg`, `mask`, `paste`, `silk`, `courtyard` or `outline`.
    pub kind: String,
    pub material: String,
    pub thickness_mm: Option<f64>,
    pub copper_oz: Option<f64>,
    /// The Gerber the layer makes, under `gerber/`; none for a dielectric.
    pub file: Option<String>,
    /// Whether the settings include it.
    pub included: bool,
}

/// The checks a board house cares about before it takes the files.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Preflight {
    pub drc_errors: usize,
    pub erc_errors: usize,
    pub unrouted: usize,
    /// Schematic changes the board hasn't taken.
    pub pending_changes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct LayoutUpdate {
    pub path: String,
    pub text: String,
}

/// The board's layout path for a `.circuit` or `.circuit.layout` path.
fn layout_path(path: &str) -> String {
    if path.ends_with(".circuit.layout") {
        path.to_owned()
    } else {
        format!("{path}.layout")
    }
}

/// The board's fabrication set as exported files, with what it covered.
pub fn fabrication(
    path: &str,
    text: &str,
    files: &Files,
    opts: &super::FabricationOptions,
    notes: &mut Vec<String>,
) -> Result<(Vec<ExportedFile>, FabricationSummary), ToolError> {
    let layout = layout_path(path);
    let layout_text = if layout == path { Some(text) } else { files.get(&layout).map(String::as_str) };
    let saved = layout_text.and_then(|t| runmat_circuit::syntax::board::parse_board(&layout, t).board).and_then(|b| b.fabrication);
    let settings = opts.settings(saved.as_ref())?;
    let mut request: FabRequest = settings.request();
    if let Some(date) = &opts.date {
        if !is_iso_date(date) {
            return Err(ToolError::invalid(format!("the date {date} isn't ISO 8601 (2026-11-06T09:30:00Z)")));
        }
        request.date = Some(date.clone());
    }
    if opts.plan_only() {
        request = FabRequest { gerber: false, drill: false, drill_map: false, drawing: false, pick_place: false, bom: false, assembly_drawing: false, ipc2581: false, odb: false, ipc356: false, ..request };
    }
    let variants = (!settings.variants.is_empty()).then_some(settings.variants.as_slice());
    let out = runmat_circuit::runtime::export_fabrication(path, text, files, &Default::default(), &request, variants).map_err(|d| {
        ToolError::new(
            "not_compiled",
            format!("{path} doesn't make fabrication outputs: {}", d.first().map(|d| d.message.as_str()).unwrap_or("")),
        )
    })?;
    let mut result = Vec::new();
    let mut odb: Vec<(String, Vec<u8>)> = Vec::new();
    for (p, t) in out.files {
        match p.strip_prefix("odb/") {
            Some(rest) => odb.push((rest.to_owned(), t.into_bytes())),
            None => {
                let mime = match p.rsplit('.').next() {
                    Some("csv") => "text/csv",
                    Some("xml") => "application/xml",
                    Some("gbrjob") => "application/json",
                    Some("gbr") => "application/vnd.gerber",
                    _ => "text/plain",
                };
                result.push(text_file(p, mime, t));
            }
        }
    }
    if !odb.is_empty() {
        result.push(binary_file(format!("odb/{}-odb.tgz", out.board), "application/gzip", &crate::formats::tgz::tgz(&odb)));
    }
    let board = &out.filled;
    if request.drill_map {
        let sheet = drawing_sheet(&drill_map(board));
        result.push(binary_file(format!("drill/{}-drill-map.pdf", out.board), "application/pdf", &crate::formats::pdf::write(&sheet)));
    }
    if request.drawing {
        let sheet = drawing_sheet(&fabrication_drawing(board, request.revision.as_deref()));
        result.push(binary_file(format!("drawing/{}-fab-drawing.pdf", out.board), "application/pdf", &crate::formats::pdf::write(&sheet)));
    }
    if request.assembly_drawing {
        for (side, d) in runmat_circuit::fab::assembly_drawing::assembly_drawings(board) {
            let sheet = drawing_sheet(&d);
            result.push(binary_file(format!("assembly/{}-assembly-{side}.pdf", out.board), "application/pdf", &crate::formats::pdf::write(&sheet)));
        }
    }
    result.sort_by(|a, b| a.name.cmp(&b.name));
    if settings.zip && !result.is_empty() {
        let entries: Vec<(&str, Vec<u8>)> = result.iter().map(|f| (f.name.as_str(), file_bytes(f))).collect();
        let zip = crate::formats::opc::zip_stored(&entries);
        result.push(binary_file(format!("{}-fab.zip", out.board), "application/zip", &zip));
    }
    let named: Vec<String> = out.variants.iter().flatten().cloned().collect();
    if !named.is_empty() && (request.bom || request.pick_place) {
        notes.push(format!("assembly files for the default variant and {}", named.join(", ")));
    }
    let rows = runmat_circuit::fab::plan::layer_rows(board);
    let chosen: Vec<String> = runmat_circuit::fab::plan::included(&rows, settings.layers.as_ref()).into_iter().map(|(f, _)| f).collect();
    let preflight = opts
        .preflight
        .then(|| {
            runmat_circuit::fab::progress::step("preflight");
            preflight(path, files)
        })
        .transpose()?;
    let preview = match &opts.preview_layer {
        Some(name) => {
            let row = rows.iter().find(|r| &r.name == name).ok_or_else(|| ToolError::invalid(format!("the board has no layer {name}")))?;
            let layer = row.gerber.ok_or_else(|| ToolError::invalid(format!("{name} makes no Gerber")))?;
            Some(LayerPreview { layer: name.clone(), svg: runmat_circuit::fab::preview::layer_svg(board, layer) })
        }
        None => None,
    };
    let layout_update = match (opts.remember, layout_text) {
        (true, Some(t)) => Some(LayoutUpdate { path: layout.clone(), text: runmat_circuit::syntax::board::with_fabrication_settings(t, &settings).map_err(ToolError::invalid)? }),
        _ => None,
    };
    let all_variants: Vec<Option<String>> = std::iter::once(None).chain(out.design_variants.iter().cloned().map(Some)).collect();
    let summary = FabricationSummary {
        board: out.board.clone(),
        layers: rows
            .iter()
            .map(|r| FabricationLayer {
                name: r.name.clone(),
                kind: r.kind.clone(),
                material: r.material.clone(),
                thickness_mm: r.thickness_mm,
                copper_oz: r.copper_oz,
                included: r.file.as_ref().is_some_and(|f| chosen.contains(f)),
                file: r.file.clone(),
            })
            .collect(),
        variants: all_variants,
        folder: settings.folder_or_default(&out.board),
        settings: serde_json::to_value(&settings).unwrap_or_default(),
        preflight,
        layout: layout_update,
        preview,
    };
    Ok((result, summary))
}

/// `YYYY-MM-DD`, optionally followed by `T` and a time.
fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    let digits = |r: std::ops::Range<usize>| r.into_iter().all(|i| b.get(i).is_some_and(u8::is_ascii_digit));
    digits(0..4) && b.get(4) == Some(&b'-') && digits(5..7) && b.get(7) == Some(&b'-') && digits(8..10) && (b.len() == 10 || (b.get(10) == Some(&b'T') && digits(11..13) && b.get(13) == Some(&b':')))
}

fn file_bytes(f: &ExportedFile) -> Vec<u8> {
    use base64::Engine as _;
    if f.encoding == "base64" {
        base64::engine::general_purpose::STANDARD.decode(f.content.as_bytes()).unwrap_or_default()
    } else {
        f.content.as_bytes().to_vec()
    }
}

/// The pre-flight counts from the CI circuit checks (without simulations).
fn preflight(path: &str, files: &Files) -> Result<Preflight, ToolError> {
    let report = crate::ci::check_circuit(path, files, false)?;
    let count = |check: &str| report.findings.iter().filter(|f| f.check == check && f.severity == "error").count();
    Ok(Preflight { drc_errors: count("drc"), erc_errors: count("erc"), unrouted: count("unrouted"), pending_changes: count("eco") })
}
