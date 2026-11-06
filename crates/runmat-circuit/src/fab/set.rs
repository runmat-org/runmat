//! The fabrication set: every output a request names, by path in the output folder.
//!
//! | Output | Paths |
//! |---|---|
//! | Gerber X2 | `gerber/<board>-<layer>.gbr`, `gerber/<board>.gbrjob` |
//! | Drill | `drill/<board>-<span>.drl` (plated per span, `NPTH`) |
//! | Drill map | `drill/<board>-drill-map.gbr` |
//! | Fabrication drawing | `drawing/<board>-fab-drawing.gbr` |
//! | Pick-and-place | `assembly/<board>[-<variant>]-pos[-top\|-bottom].csv` |
//! | BOM | `assembly/<board>[-<variant>]-bom.csv` |
//! | Assembly drawing | `assembly/<board>-assembly-top.gbr`, `-bottom.gbr` |
//! | IPC-2581C | `ipc2581/<board>.xml` |
//! | ODB++ | `odb/<job>/…` (the job tree; archive it as `<board>-odb.tgz`) |
//! | IPC-D-356A | `netlist/<board>.ipc` |
//!
//! The default variant's assembly files carry no variant suffix.

use super::{assembly, drawing, drill, job, odb, BomLine};
use crate::layout::Board;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Which outputs to make. Every output is on by default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FabRequest {
    pub gerber: bool,
    pub drill: bool,
    pub drill_map: bool,
    pub drawing: bool,
    pub pick_place: bool,
    pub bom: bool,
    pub ipc2581: bool,
    pub odb: bool,
    pub ipc356: bool,
    /// The assembly drawing (outlines and designators, top and bottom).
    pub assembly_drawing: bool,
    /// Pick-and-place lists through-hole parts too.
    pub include_through_hole: bool,
    /// The Gerber layers to write, by layer name (`F.Cu`, `F.Paste`, `Edge.Cuts`); the
    /// default rows of [`super::plan::layer_rows`] when `None`.
    pub layers: Option<BTreeSet<String>>,
    /// ISO 8601 date for the Gerber job file's header; none keeps the set deterministic.
    pub date: Option<String>,
    /// The Gerber files' unit and decimals (the layers, drill map and drawings).
    pub gerber_format: super::GerberFormat,
    /// The pick-and-place's sides and unit.
    pub pick_place_format: assembly::PickPlaceFormat,
    /// The bill of materials' columns and grouping.
    pub bom_format: assembly::BomFormat,
    /// The board revision the Gerber files and the job file name (`rev?` when none).
    pub revision: Option<String>,
}

impl Default for FabRequest {
    fn default() -> Self {
        FabRequest {
            gerber: true,
            drill: true,
            drill_map: true,
            drawing: true,
            pick_place: true,
            bom: true,
            ipc2581: true,
            odb: true,
            ipc356: true,
            assembly_drawing: true,
            include_through_hole: false,
            layers: None,
            date: None,
            gerber_format: Default::default(),
            pick_place_format: Default::default(),
            bom_format: Default::default(),
            revision: None,
        }
    }
}

/// One assembly variant: its name (`None` for the design's default), the designators it fits
/// and its bill of materials.
#[derive(Clone, Debug, PartialEq)]
pub struct VariantSet {
    pub name: Option<String>,
    pub fitted: BTreeSet<String>,
    pub bom: Vec<BomLine>,
}

/// The output kinds a path of the set belongs to, for listing (`gerber`, `drill`, …).
pub fn kind_of(path: &str) -> &str {
    path.split('/').next().unwrap_or(path)
}

/// The fabrication set of `board`: path → text. `parts` maps a reference designator to its
/// part number (IPC-2581 and ODB++ name parts by it).
pub fn fabrication(
    board: &Board,
    variants: &[VariantSet],
    parts: &BTreeMap<String, String>,
    req: &FabRequest,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let name = &board.name;
    let rev = req.revision.as_deref();
    let drills = drill::excellon(board);
    use super::progress::step;
    if req.gerber {
        step("gerber");
        let layers = super::plan::included(&super::plan::layer_rows(board), req.layers.as_ref());
        for (file, layer) in &layers {
            out.insert(format!("gerber/{file}"), super::gerber_in(board, *layer, req.gerber_format, rev));
        }
        let drill_functions: Vec<(String, String)> = if req.drill {
            drills
                .iter()
                .map(|(file, text)| {
                    let function = text
                        .lines()
                        .find_map(|l| l.strip_prefix("; #@! TF.FileFunction,"))
                        .unwrap_or("Plated")
                        .to_owned();
                    (format!("../drill/{file}"), function)
                })
                .collect()
        } else {
            Vec::new()
        };
        out.insert(
            format!("gerber/{name}.gbrjob"),
            job::gerber_job(board, &layers, &drill_functions, req.date.as_deref(), rev),
        );
    }
    if req.drill {
        step("drill");
        for (file, text) in &drills {
            out.insert(format!("drill/{file}"), text.clone());
        }
    }
    if req.drill_map {
        step("drill_map");
        out.insert(
            format!("drill/{name}-drill-map.gbr"),
            drawing::drill_map(board).gerber_in(board, "Drillmap", req.gerber_format, rev),
        );
    }
    if req.drawing {
        step("drawing");
        out.insert(
            format!("drawing/{name}-fab-drawing.gbr"),
            drawing::fabrication_drawing(board).gerber_in(board, "FabricationDrawing", req.gerber_format, rev),
        );
    }
    if req.assembly_drawing {
        step("assembly_drawing");
        for (side, drawing) in super::assembly_drawing::assembly_drawings(board) {
            out.insert(
                format!("assembly/{name}-assembly-{side}.gbr"),
                drawing.gerber_in(board, &format!("AssemblyDrawing,{}", if side == "top" { "Top" } else { "Bot" }), req.gerber_format, rev),
            );
        }
    }
    let suffix = |v: &VariantSet| v.name.as_ref().map(|n| format!("-{n}")).unwrap_or_default();
    if req.pick_place {
        step("pick_place");
        let f = req.pick_place_format;
        for v in variants {
            for (side_suffix, side) in f.sides.files() {
                out.insert(
                    format!("assembly/{name}{}-pos{side_suffix}.csv", suffix(v)),
                    assembly::pick_place_in(board, req.include_through_hole, Some(&v.fitted), side, f.units),
                );
            }
        }
    }
    if req.bom {
        step("bom");
        for v in variants {
            out.insert(format!("assembly/{name}{}-bom.csv", suffix(v)), assembly::bom_csv_in(&v.bom, &req.bom_format));
        }
    }
    if req.ipc2581 {
        step("ipc2581");
        out.insert(
            format!("ipc2581/{name}.xml"),
            crate::ipc2581::export(board, parts),
        );
    }
    if req.odb {
        step("odb");
        for (path, text) in odb::odb(board, parts) {
            out.insert(format!("odb/{path}"), text);
        }
    }
    if req.ipc356 {
        step("ipc356");
        out.insert(format!("netlist/{name}.ipc"), assembly::ipc356(board));
    }
    out
}
