//! KiCad projects in and out of a circuit's sources.

use super::*;

mod parts;
mod rules;

/// Why a KiCad export wrote nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum KicadExportError {
    /// The circuit doesn't compile.
    NotCompiled(Vec<Diagnostic>),
    /// KiCad can't hold the board: more copper layers than KiCad's limit. The message names
    /// both counts and the outputs that write every layer.
    Unsupported(String),
}

/// The circuit `path` belongs to as a KiCad project: one `.kicad_sch` per sheet instance (the
/// root named `<project>.kicad_sch`) and, with a board, `<project>.kicad_pcb`. File name → text.
/// A board deeper than KiCad's copper stack fails the whole export: no file is written.
pub fn export_kicad(
    path: &str,
    text: &str,
    files: &Files,
    options: &CompileOptions,
    project_name: &str,
) -> Result<BTreeMap<String, String>, KicadExportError> {
    let project = load(path, text, files, options).map_err(KicadExportError::NotCompiled)?;
    if let Some(board) = project
        .layout
        .as_ref()
        .and_then(|l| l.parsed.board.as_ref())
    {
        crate::kicad::pcb::check_copper_count(board).map_err(KicadExportError::Unsupported)?;
    }
    let mut out = crate::kicad::sch_export::export(
        &project.netlist,
        &project.pool,
        &crate::kicad::sch_export::SchExportOptions {
            project: project_name.to_owned(),
        },
    );
    if let Some(board) = project
        .layout
        .as_ref()
        .and_then(|l| l.parsed.board.as_ref())
    {
        let mut board = board.clone();
        let (mut rules, _) = BoardRules::for_board(&board);
        rules.rebuild(&BoardGraph::new(&board, &[]));
        fill_zones(&mut board, &rules, &[]);
        out.insert(
            format!("{project_name}.kicad_pcb"),
            crate::kicad::pcb::write_board(&board).map_err(KicadExportError::Unsupported)?,
        );
    }
    Ok(out)
}

/// RunMat sources read from a KiCad project.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KicadImport {
    /// Workspace path → text: `<stem>.circuit` (a `part` declaration per symbol and value),
    /// `<stem>.circuit.lock` and, with a board, `<stem>.circuit.layout` and the board's
    /// footprints as `<stem>-footprints/<name>.kicad_mod`, the parts' packages.
    pub files: BTreeMap<String, String>,
    /// What the reader passed over (unsupported tokens, unknown footprints).
    pub warnings: Vec<String>,
    /// What the import carried over, and each warning as an item it left out.
    pub report: crate::import::Report,
}

/// Reads a KiCad project (`files`: KiCad file name → text) into RunMat sources at
/// `<target_stem>.circuit` (and `.circuit.layout`, `.circuit.lock`). `root` names the root
/// schematic; a project may also be a board alone.
pub fn import_kicad(
    files: &BTreeMap<String, String>,
    root: Option<&str>,
    target_stem: &str,
) -> Result<KicadImport, String> {
    let mut out = BTreeMap::new();
    let mut warnings = Vec::new();
    let mut report = crate::import::Report::new(".kicad_sch / .kicad_pcb");
    let circuit_path = format!("{target_stem}{CIRCUIT_SUFFIX}");
    let circuit_name = target_stem
        .rsplit('/')
        .next()
        .unwrap_or(target_stem)
        .to_owned();
    let root = root.map(str::to_owned).or_else(|| {
        let mut schematics: Vec<&String> =
            files.keys().filter(|k| k.ends_with(".kicad_sch")).collect();
        schematics.sort_by_key(|k| k.len());
        // The root is the schematic no other sheet names.
        schematics
            .iter()
            .find(|k| {
                let name = k.rsplit('/').next().unwrap_or(k);
                !files.iter().any(|(other, t)| {
                    other != **k && other.ends_with(".kicad_sch") && t.contains(name)
                })
            })
            .map(|k| (*k).clone())
    });
    let project_rules = rules::read(files, &mut report)?;
    let mut packages = BTreeMap::new();
    let mut board_read = None;
    if let Some((name, pcb)) = files.iter().find(|(k, _)| k.ends_with(".kicad_pcb")) {
        let read = crate::kicad::pcb::read(pcb).map_err(|e| format!("{name}: {e}"))?;
        warnings.extend(read.warnings.iter().map(|w| format!("{name}: {w}")));
        let b = &read.board;
        report.count("footprints placed", b.footprints.len());
        report.count("tracks", b.tracks.len());
        report.count("vias", b.vias.len());
        report.count("zones", b.zones.len());
        report.count("copper layers", b.stackup.copper_count() as usize);
        let uses = [runmat_design_syntax::document::UseItem {
            symbol: Some(("circuit".into(), circuit_name.clone())),
            // The circuit is written beside the layout, in the same folder.
            from: String::new(),
            alias: None,
            revision: None,
            span: Default::default(),
        }];
        // The layout takes the circuit's name, as its file does.
        let mut board = read.board.clone();
        board.name = circuit_name.clone();
        rules::apply_to_board(&project_rules, &mut board, &mut report);
        out.insert(
            format!("{target_stem}{LAYOUT_SUFFIX}"),
            crate::syntax::board::print_board(&board, &uses),
        );
        packages = parts::board_packages(&read.board);
        board_read = Some(board);
    }
    if let Some(root) = root {
        let imp = crate::kicad::sch::read_hierarchy(files, &root)?;
        warnings.extend(imp.warnings.iter().cloned());
        let mut design = imp.design.clone();
        design.name = circuit_name.clone();
        rules::apply_to_design(
            &project_rules,
            &mut design,
            board_read.as_ref(),
            &mut report,
        );
        let footprint_dir = format!("{circuit_name}-footprints");
        let folder = target_stem
            .rsplit_once('/')
            .map(|(d, _)| format!("{d}/"))
            .unwrap_or_default();
        for (path, text) in parts::declare_parts(
            &mut design,
            &imp.pool,
            &packages,
            &footprint_dir,
            &mut report,
        ) {
            out.insert(format!("{folder}{path}"), text);
        }
        report.count("sheets", design.sheets.len());
        report.count(
            "components",
            design.sheets.values().map(|s| s.components.len()).sum(),
        );
        report.count(
            "connections",
            design.sheets.values().map(|s| s.connections.len()).sum(),
        );
        let text = crate::syntax::print_circuit(&design).map_err(|e| format!("{e:?}"))?;
        out.insert(circuit_path.clone(), text);
        out.insert(lock_path_for(&circuit_path), imp.lock.to_text());
    } else if board_read.is_some() {
        out.insert(
            circuit_path.clone(),
            crate::import::circuit_stub(&circuit_name, &mut report)?,
        );
    }
    if out.is_empty() {
        return Err("give a .kicad_sch or .kicad_pcb file".into());
    }
    for w in &warnings {
        report.not_carried(w.clone());
    }
    Ok(KicadImport {
        files: out,
        warnings,
        report,
    })
}
