//! KiCad projects in and out of a circuit's sources.

use super::*;

/// The circuit `path` belongs to as a KiCad project: one `.kicad_sch` per sheet instance (the
/// root named `<project>.kicad_sch`) and, with a board, `<project>.kicad_pcb`. File name → text.
pub fn export_kicad(
    path: &str,
    text: &str,
    files: &Files,
    options: &CompileOptions,
    project_name: &str,
) -> Result<BTreeMap<String, String>, Vec<Diagnostic>> {
    let project = load(path, text, files, options)?;
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
        let (mut rules, _) = BoardRules::new(&board.limits, &board.rules);
        rules.rebuild(&BoardGraph::new(&board, &[]));
        fill_zones(&mut board, &rules, &[]);
        out.insert(
            format!("{project_name}.kicad_pcb"),
            crate::kicad::pcb::write_board(&board),
        );
    }
    Ok(out)
}

/// RunMat sources read from a KiCad project.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KicadImport {
    /// Workspace path → text: `<stem>.circuit`, `<stem>.circuit.lock` and, with a board,
    /// `<stem>.circuit.layout`.
    pub files: BTreeMap<String, String>,
    /// What the reader passed over (unsupported tokens, unknown footprints).
    pub warnings: Vec<String>,
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
    if let Some(root) = root {
        let imp = crate::kicad::sch::read_hierarchy(files, &root)?;
        warnings.extend(imp.warnings.iter().cloned());
        let mut design = imp.design.clone();
        design.name = circuit_name.clone();
        let text = crate::syntax::print_circuit(&design).map_err(|e| format!("{e:?}"))?;
        out.insert(circuit_path.clone(), text);
        out.insert(lock_path_for(&circuit_path), imp.lock.to_text());
    }
    if let Some((name, pcb)) = files.iter().find(|(k, _)| k.ends_with(".kicad_pcb")) {
        let read = crate::kicad::pcb::read(pcb).map_err(|e| format!("{name}: {e}"))?;
        warnings.extend(read.warnings.iter().map(|w| format!("{name}: {w}")));
        let uses = [runmat_design_syntax::document::UseItem {
            symbol: Some(("circuit".into(), circuit_name.clone())),
            from: format!(
                "{}{CIRCUIT_SUFFIX}",
                target_stem.rsplit('/').next().unwrap_or(target_stem)
            ),
            alias: None,
            revision: None,
            span: Default::default(),
        }];
        out.insert(
            format!("{target_stem}{LAYOUT_SUFFIX}"),
            crate::syntax::board::print_board(&read.board, &uses),
        );
    }
    if out.is_empty() {
        return Err("the KiCad project has no .kicad_sch or .kicad_pcb file".into());
    }
    Ok(KicadImport {
        files: out,
        warnings,
    })
}
