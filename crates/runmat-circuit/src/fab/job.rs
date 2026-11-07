//! The Gerber X2 job file (`.gbrjob`): the board's size, thickness, finish, design rules, the
//! attributes of every Gerber and drill file of the set, and the material stackup, as the
//! published job-file format describes them (JSON, millimetres).

use super::file_function;
use crate::geom::to_mm;
use crate::layout::{Board, StackKind};
use serde_json::{json, Value};

/// The job file for the Gerbers in `layers` (file name and layer) and the drill files named
/// in `drill_files` (file name → X2 file function). `date` is the ISO 8601 creation date;
/// without it the header carries none, so the output stays deterministic.
pub fn gerber_job(board: &Board, layers: &[(String, super::GerberLayer)], drill_files: &[(String, String)], date: Option<&str>, revision: Option<&str>) -> String {
    let n = board.copper_count();
    let bb = board.outline.bbox();
    let thickness: crate::geom::Coord = board.stackup.layers.iter().map(|l| l.thickness).sum();
    let mut header = json!({
        "GenerationSoftware": {
            "Vendor": "RunMat",
            "Application": "runmat-circuit",
            "Version": env!("CARGO_PKG_VERSION"),
        }
    });
    if let Some(d) = date {
        header["CreationDate"] = Value::String(d.to_owned());
    }
    let mut general = json!({
        "ProjectId": {
            "Name": board.name,
            "GUID": crate::ids::Uuid(crate::ids::fnv128(board.name.as_bytes())).with_version(8).to_string(),
            "Revision": revision.unwrap_or("rev?"),
        },
        "Size": { "X": to_mm(bb.width()), "Y": to_mm(bb.height()) },
        "LayerNumber": n,
        "BoardThickness": to_mm(thickness),
    });
    if !board.stackup.finish.is_empty() {
        general["Finish"] = Value::String(board.stackup.finish.clone());
    }
    let l = &board.limits;
    let rules = json!([{
        "Layers": "All",
        "PadToPad": to_mm(l.min_clearance),
        "PadToTrack": to_mm(l.min_clearance),
        "TrackToTrack": to_mm(l.min_clearance),
        "MinLineWidth": to_mm(l.min_track),
    }]);
    let mut files: Vec<Value> = layers
        .iter()
        .map(|(path, layer)| {
            let (function, polarity) = file_function(board, *layer);
            json!({ "Path": path, "FileFunction": function, "FilePolarity": polarity })
        })
        .collect();
    for (path, function) in drill_files {
        files.push(json!({ "Path": path, "FileFunction": function, "FilePolarity": "Positive" }));
    }
    let mut cu: crate::layout::CopperIndex = 0;
    let stack: Vec<Value> = board
        .stackup
        .layers
        .iter()
        .map(|s| {
            let mut v = json!({ "Name": s.name, "Thickness": to_mm(s.thickness) });
            match &s.kind {
                StackKind::Copper { .. } => {
                    v["Type"] = "Copper".into();
                    v["Name"] = crate::layout::copper_name(cu, n).into();
                    cu += 1;
                }
                StackKind::Dielectric { core } => {
                    v["Type"] = "Dielectric".into();
                    v["Material"] = s.material.name.clone().into();
                    v["DielectricConstant"] = s.material.dk.into();
                    v["LossTangent"] = s.material.df.into();
                    v["Notes"] = if *core { "Core" } else { "Prepreg" }.into();
                }
                StackKind::SolderMask => {
                    v["Type"] = "SolderMask".into();
                    v["Material"] = s.material.name.clone().into();
                }
                StackKind::Silkscreen => v["Type"] = "Legend".into(),
                StackKind::Paste => v["Type"] = "SolderPaste".into(),
            }
            v
        })
        .collect();
    let job = json!({
        "Header": header,
        "GeneralSpecs": general,
        "DesignRules": rules,
        "FilesAttributes": files,
        "MaterialStackup": stack,
    });
    let mut s = serde_json::to_string_pretty(&job).unwrap_or_default();
    s.push('\n');
    s
}
