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

/// What a fabrication export covered: the board, its layers top to bottom and the variants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FabricationSummary {
    pub board: String,
    pub layers: Vec<FabricationLayer>,
    /// The variants the assembly files were made for; `null` is the default.
    pub variants: Vec<Option<String>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FabricationLayer {
    pub name: String,
    /// `copper`, `core`, `prepreg`, `solder mask`, `legend` or `paste`.
    pub kind: String,
    pub material: String,
    pub thickness_mm: f64,
}

/// The board's fabrication set as exported files.
pub fn fabrication(
    path: &str,
    text: &str,
    files: &Files,
    request: &FabRequest,
    variants: Option<&[String]>,
    notes: &mut Vec<String>,
) -> Result<(Vec<ExportedFile>, FabricationSummary), ToolError> {
    let out = runmat_circuit::runtime::export_fabrication(
        path,
        text,
        files,
        &Default::default(),
        request,
        variants,
    )
    .map_err(|d| {
        ToolError::new(
            "not_compiled",
            format!(
                "{path} doesn't make fabrication outputs: {}",
                d.first().map(|d| d.message.as_str()).unwrap_or("")
            ),
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
        result.push(binary_file(
            format!("odb/{}-odb.tgz", out.board),
            "application/gzip",
            &crate::formats::tgz::tgz(&odb),
        ));
    }
    let board = &out.filled;
    {
        if request.drill_map {
            let sheet = drawing_sheet(&drill_map(board));
            result.push(binary_file(
                format!("drill/{}-drill-map.pdf", out.board),
                "application/pdf",
                &crate::formats::pdf::write(&sheet),
            ));
        }
        if request.drawing {
            let sheet = drawing_sheet(&fabrication_drawing(board));
            result.push(binary_file(
                format!("drawing/{}-fab-drawing.pdf", out.board),
                "application/pdf",
                &crate::formats::pdf::write(&sheet),
            ));
        }
    }
    result.sort_by(|a, b| a.name.cmp(&b.name));
    let named: Vec<String> = out.variants.iter().flatten().cloned().collect();
    if !named.is_empty() {
        notes.push(format!(
            "assembly files for the default variant and {}",
            named.join(", ")
        ));
    }
    let summary = FabricationSummary {
        board: out.board.clone(),
        layers: out
            .layers
            .iter()
            .map(|l| FabricationLayer {
                name: l.name.clone(),
                kind: l.kind.clone(),
                material: l.material.clone(),
                thickness_mm: l.thickness_mm,
            })
            .collect(),
        variants: out.variants.clone(),
    };
    Ok((result, summary))
}
