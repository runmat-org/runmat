//! The rows of a board's fabrication layer table, top to bottom: paste, legend and mask over
//! the stackup's copper and dielectrics, then the bottom's mask, legend and paste, the
//! courtyards and the board outline. Each row says which Gerber it makes (dielectrics make
//! none) and whether it is in the set by default: courtyards are off.

use super::GerberLayer;
use crate::geom::to_mm;
use crate::layout::{copper_name, Board, Side, StackKind};
use serde::Serialize;

/// One row of the layer table.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerRow {
    /// The layer's name (`F.Cu`, `F.Mask`, `core`, `Edge.Cuts`); `skip_layers` names it.
    pub name: String,
    /// `copper`, `core`, `prepreg`, `mask`, `paste`, `silk`, `courtyard` or `outline`.
    pub kind: String,
    /// The material, when the stackup gives one.
    pub material: String,
    pub thickness_mm: Option<f64>,
    pub copper_oz: Option<f64>,
    /// The Gerber the row makes, under `gerber/`.
    pub file: Option<String>,
    #[serde(skip)]
    pub gerber: Option<GerberLayer>,
    /// In the set unless left out (courtyards are out unless asked for).
    pub default_on: bool,
}

fn side_name(s: Side) -> &'static str {
    if s == Side::Top {
        "F"
    } else {
        "B"
    }
}

/// The board's layer rows, top to bottom.
pub fn layer_rows(board: &Board) -> Vec<LayerRow> {
    let n = board.copper_count();
    let file = |suffix: &str| Some(format!("{}-{suffix}.gbr", board.name));
    let mask_layer = board.stackup.layers.iter().find(|l| matches!(l.kind, StackKind::SolderMask));
    let side_rows = |s: Side, top: bool| -> Vec<LayerRow> {
        let p = side_name(s);
        let row = |name: String, kind: &str, layer: GerberLayer, suffix: &str| LayerRow {
            name,
            kind: kind.into(),
            material: String::new(),
            thickness_mm: None,
            copper_oz: None,
            file: file(&format!("{p}_{suffix}")),
            gerber: Some(layer),
            default_on: true,
        };
        let mut mask = row(format!("{p}.Mask"), "mask", GerberLayer::Mask(s), "Mask");
        if let Some(m) = mask_layer {
            mask.material = m.material.name.clone();
            mask.thickness_mm = Some(to_mm(m.thickness));
        }
        let mut rows = vec![
            row(format!("{p}.Paste"), "paste", GerberLayer::Paste(s), "Paste"),
            row(format!("{p}.Silkscreen"), "silk", GerberLayer::Legend(s), "Silkscreen"),
            mask,
        ];
        if !top {
            rows.reverse();
        }
        rows
    };
    let mut out = side_rows(Side::Top, true);
    let mut cu = 0u8;
    for l in &board.stackup.layers {
        let (name, kind, oz, gerber) = match &l.kind {
            StackKind::Copper { weight_oz } => {
                let name = copper_name(cu, n);
                let g = GerberLayer::Copper(cu);
                cu += 1;
                (name, "copper", Some(*weight_oz), Some(g))
            }
            StackKind::Dielectric { core } => (l.name.clone(), if *core { "core" } else { "prepreg" }, None, None),
            // Mask, legend and paste rows come from the sides above and below.
            _ => continue,
        };
        out.push(LayerRow {
            file: gerber.map(|_| format!("{}-{}.gbr", board.name, name.replace('.', "_"))),
            name,
            kind: kind.into(),
            material: l.material.name.clone(),
            thickness_mm: Some(to_mm(l.thickness)),
            copper_oz: oz,
            gerber,
            default_on: gerber.is_some(),
        });
    }
    out.extend(side_rows(Side::Bottom, false));
    for s in [Side::Top, Side::Bottom] {
        let p = side_name(s);
        out.push(LayerRow {
            name: format!("{p}.Courtyard"),
            kind: "courtyard".into(),
            material: String::new(),
            thickness_mm: None,
            copper_oz: None,
            file: file(&format!("{p}_Courtyard")),
            gerber: Some(GerberLayer::Courtyard(s)),
            default_on: false,
        });
    }
    out.push(LayerRow {
        name: "Edge.Cuts".into(),
        kind: "outline".into(),
        material: String::new(),
        thickness_mm: None,
        copper_oz: None,
        file: file("Edge_Cuts"),
        gerber: Some(GerberLayer::Profile),
        default_on: true,
    });
    out
}

/// The Gerbers the set writes: the rows named in `layers`, or every row that is on by
/// default when `layers` is `None`.
pub fn included(rows: &[LayerRow], layers: Option<&std::collections::BTreeSet<String>>) -> Vec<(String, GerberLayer)> {
    rows.iter()
        .filter(|r| match layers {
            Some(names) => names.contains(&r.name),
            None => r.default_on,
        })
        .filter_map(|r| Some((r.file.clone()?, r.gerber?)))
        .collect()
}
