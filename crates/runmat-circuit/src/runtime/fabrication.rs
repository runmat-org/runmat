//! The fabrication set of a circuit's board, with one BOM and pick-and-place per variant.

use super::*;
use crate::fab::set::{fabrication, FabRequest, VariantSet};

/// What [`export_fabrication`] made.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FabricationExport {
    /// The board's name (file names start with it).
    pub board: String,
    /// Path in the output folder → text (see [`crate::fab::set`] for the paths).
    pub files: BTreeMap<String, String>,
    /// The variants the assembly files were made for (`None` is the default).
    pub variants: Vec<Option<String>>,
    /// Every variant the design declares, by name.
    pub design_variants: Vec<String>,
    /// The board's layers, top to bottom, as the output dialog lists them.
    pub layers: Vec<FabLayer>,
    /// The board the set was made from, zones filled.
    #[serde(skip)]
    pub filled: Board,
}

/// One layer of the board's stackup.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FabLayer {
    pub name: String,
    pub kind: String,
    pub material: String,
    pub thickness_mm: f64,
}

/// The fabrication set of the board that `path` (a `.circuit` or its `.circuit.layout`)
/// belongs to. Zones are filled first. The default variant and every named variant of the
/// design get their own BOM and pick-and-place; `variants` narrows that to the named ones
/// (`""` names the default).
pub fn export_fabrication(
    path: &str,
    text: &str,
    files: &Files,
    options: &CompileOptions,
    request: &FabRequest,
    variants: Option<&[String]>,
) -> Result<FabricationExport, Vec<Diagnostic>> {
    let project = load(path, text, files, options)?;
    let Some(board) = project
        .layout
        .as_ref()
        .and_then(|l| l.parsed.board.as_ref())
    else {
        return Err(vec![Diagnostic::error(
            path,
            Some(text),
            0,
            0,
            format!("{path} has no board: fabrication outputs need its .circuit.layout"),
        )]);
    };
    crate::fab::progress::step("board");
    let mut board = board.clone();
    let (mut rules, _) = BoardRules::new(&board.limits, &board.rules);
    rules.rebuild(&BoardGraph::new(&board, &[]));
    fill_zones(&mut board, &rules, &[]);

    let mut names: Vec<Option<String>> = vec![None];
    names.extend(project.design.variants.keys().cloned().map(Some));
    if let Some(want) = variants {
        names.retain(|n| want.iter().any(|w| w == n.as_deref().unwrap_or("")));
    }
    let mut sets = Vec::new();
    for name in &names {
        let netlist = if name.is_none() {
            project.netlist.clone()
        } else {
            let opts = CompileOptions {
                variant: name.clone(),
                ..options.clone()
            };
            load(path, text, files, &opts)?.netlist
        };
        sets.push(VariantSet {
            name: name.clone(),
            fitted: netlist
                .components
                .iter()
                .filter(|c| c.fitted())
                .map(|c| c.designator.clone())
                .collect(),
            bom: crate::fab::bom(&netlist, &project.pool),
        });
    }
    let parts: BTreeMap<String, String> = project
        .netlist
        .components
        .iter()
        .filter_map(|c| {
            let p = c.part.and_then(|u| project.pool.parts.get(&u))?;
            (!p.mpn.is_empty()).then(|| (c.designator.clone(), p.mpn.clone()))
        })
        .collect();
    let layers = crate::fab::drawing::stackup_rows(&board)
        .into_iter()
        .zip(&board.stackup.layers)
        .map(|(row, l)| FabLayer {
            name: row[1].clone(),
            kind: row[2].clone(),
            material: l.material.name.clone(),
            thickness_mm: crate::geom::to_mm(l.thickness),
        })
        .collect();
    Ok(FabricationExport {
        board: board.name.clone(),
        files: fabrication(&board, &sets, &parts, request),
        variants: names,
        design_variants: project.design.variants.keys().cloned().collect(),
        layers,
        filled: board,
    })
}

/// The elaborated netlist (in `options.variant`) and part pool of the circuit `path` belongs
/// to, for the netlist exports ([`crate::netlists`], SPICE decks).
pub fn netlist_and_pool(
    path: &str,
    text: &str,
    files: &Files,
    options: &CompileOptions,
) -> Result<(Netlist, Pool), Vec<Diagnostic>> {
    load(path, text, files, options).map(|p| (p.netlist, p.pool))
}
