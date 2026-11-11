//! The overview's links: what the file uses, and the project files that use it.

use super::facts::problems;
use super::{plural, Level, Link, Ran, Relation};
use crate::check::{check, CheckInput};
use crate::{dependency_files, DesignKind, Files};
use runmat_design_syntax::{
    FeaDeclarationKind, FeaExpressionKind, FeaModuleItem, FeaStatementKind,
};
use std::sync::atomic::{AtomicBool, Ordering};

static MODELS_CHECKED: AtomicBool = AtomicBool::new(true);

/// Whether an overview checks the parts, assemblies and drawings it links (on by default). The
/// browser's CAD host turns it off: its heap is bounded, and a product's links regenerated one
/// after another exhaust it.
pub fn check_linked_models(on: bool) {
    MODELS_CHECKED.store(on, Ordering::Relaxed);
}

/// A part's, assembly's or drawing's link where the tools run without the design kernel.
pub const NOT_CHECKED_HERE: &str = "Checked when it opens in the editor";

/// The most related files an overview reads beyond the file and what it needs.
pub const MAX_RELATED: usize = 48;

/// The project files an overview of `path` reads to find what uses it: every other design file
/// and every `.fea` study, in path order, at most [`MAX_RELATED`]. The caller reads them (and
/// what they need) into the overview's [`Files`].
pub fn related_candidates(path: &str, project: &[String]) -> Vec<String> {
    let mut out: Vec<String> = project
        .iter()
        .filter(|p| p.as_str() != path)
        .filter(|p| {
            crate::is_design_path(p) && !p.ends_with(".lock")
                || p.to_ascii_lowercase().ends_with(".fea")
        })
        .cloned()
        .collect();
    out.sort();
    out.dedup();
    out.truncate(MAX_RELATED);
    out
}

/// The files a study's `geometry { file = "…" }` declarations name, resolved against the study's
/// folder. This is the FEA module graph's own rule for a study's geometry sources.
pub fn study_geometry_files(study_path: &str, text: &str) -> Vec<String> {
    let module = runmat_design_syntax::parse_fea(text);
    let mut out = Vec::new();
    for item in &module.items {
        let FeaModuleItem::Declaration(decl) = item else {
            continue;
        };
        if decl.kind != FeaDeclarationKind::Geometry {
            continue;
        }
        for statement in &decl.body.statements {
            let FeaStatementKind::Assignment { target, value, .. } = &statement.kind else {
                continue;
            };
            if target.segments.len() != 1 || target.segments[0].text != "file" {
                continue;
            }
            if let FeaExpressionKind::String(relative) = &value.kind {
                out.push(runmat_circuit::runtime::resolve_path(study_path, relative));
            }
        }
    }
    out
}

/// A study's syntax standing and the studies it declares.
fn study_status(text: &str) -> (String, Level) {
    let module = runmat_design_syntax::parse_fea(text);
    let studies: Vec<&str> = module
        .items
        .iter()
        .filter_map(|i| match i {
            FeaModuleItem::Declaration(d) if d.kind == FeaDeclarationKind::Study => {
                Some(d.name.text.as_str())
            }
            _ => None,
        })
        .collect();
    let declared = if studies.is_empty() {
        "no study declared yet".to_owned()
    } else {
        format!(
            "{}: {}",
            plural(studies.len(), "study", "studies"),
            studies.join(", ")
        )
    };
    if module.diagnostics.is_empty() {
        (format!("No syntax errors; {declared}"), Level::Ok)
    } else {
        (
            format!(
                "{}; {declared}",
                plural(module.diagnostics.len(), "syntax error", "syntax errors")
            ),
            Level::Error,
        )
    }
}

fn is_study(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".fea")
}

fn is_lock(path: &str) -> bool {
    path.to_ascii_lowercase().ends_with(".lock")
}

/// How `other`, which uses the overview's file, relates to it.
fn user_relation(other: &str) -> Option<Relation> {
    if is_study(other) {
        return Some(Relation::Study);
    }
    match DesignKind::of(other)? {
        DesignKind::Layout => Some(Relation::Board),
        DesignKind::Drawing => Some(Relation::Drawing),
        DesignKind::Assembly => Some(Relation::Assembly),
        DesignKind::Part | DesignKind::Circuit => None,
    }
}

/// Whether `other` (with `text`) uses `path`.
fn uses(other: &str, text: &str, path: &str, files: &Files) -> bool {
    if is_study(other) {
        study_geometry_files(other, text).iter().any(|p| p == path)
    } else {
        dependency_files(other, text, files)
            .iter()
            .any(|p| p == path)
    }
}

fn status_of(path: &str, files: &Files, ran: &mut Ran) -> (String, Level, String) {
    if is_study(path) {
        ran.note("fea.syntax");
        let (s, l) = study_status(files.get(path).map(String::as_str).unwrap_or(""));
        return (s, l, "fea.syntax".into());
    }
    // A board stands by its design rules: DRC fills the pours and checks the copper.
    if DesignKind::of(path) == Some(DesignKind::Layout) {
        let input = crate::rules::RulesInput {
            path: path.to_owned(),
            severities: None,
            filter: None,
            limit: Some(0),
        };
        if let Ok(d) = crate::rules::drc(&input, files) {
            if d.ran {
                ran.note("design.drc");
                let (s, l) = problems(d.errors, d.warnings);
                return (format!("Design rules: {s}"), l, "design.drc".into());
            }
        }
    }
    // A part, assembly or drawing regenerates on OCCT. Where the tools run without it (the
    // browser), checking one would compile it on the test kernel: slow for a whole product (the
    // drone recompiles its parts) and not the answer the editor gives. The link says so instead.
    // A drawing regenerates the models its views show, so it counts as one (the drone's drawing
    // projects the whole product).
    let regenerates =
        DesignKind::of(path).is_some_and(|k| k.is_model() || k == DesignKind::Drawing);
    if regenerates && !MODELS_CHECKED.load(Ordering::Relaxed) {
        // The link itself comes from reading the files, as design.inspect does.
        ran.note("design.inspect");
        return (
            NOT_CHECKED_HERE.into(),
            Level::Info,
            "design.inspect".into(),
        );
    }
    ran.note("design.check");
    match check(
        &CheckInput {
            path: path.to_owned(),
        },
        files,
    ) {
        Ok(c) => {
            let (s, l) = problems(c.errors, c.warnings);
            (s, l, "design.check".into())
        }
        Err(e) => (e.message, Level::Error, "design.check".into()),
    }
}

/// The files that use `path`, through `via` when it is a hop away.
fn users(
    path: &str,
    files: &Files,
    via: Option<&str>,
    seen: &mut Vec<String>,
    out: &mut Vec<Link>,
    ran: &mut Ran,
) {
    for (other, text) in files {
        if other == path
            || is_lock(other)
            || seen.contains(other)
            || !uses(other, text, path, files)
        {
            continue;
        }
        let Some(relation) = user_relation(other) else {
            continue;
        };
        seen.push(other.clone());
        let (status, level, source) = status_of(other, files, ran);
        out.push(Link {
            path: other.clone(),
            relation,
            via: via.map(str::to_owned),
            status,
            level,
            source,
        });
    }
}

pub(super) fn links(kind: DesignKind, path: &str, files: &Files, ran: &mut Ran) -> Vec<Link> {
    let mut out = Vec::new();
    let mut seen = vec![path.to_owned()];
    // What the file uses: a board's schematic, an assembly's parts, a drawing's models.
    let text = files.get(path).map(String::as_str).unwrap_or("");
    for dep in dependency_files(path, text, files) {
        if is_lock(&dep) || !files.contains_key(&dep) || seen.contains(&dep) {
            continue;
        }
        let relation = match (kind, DesignKind::of(&dep)) {
            (DesignKind::Layout, Some(DesignKind::Circuit)) => Relation::Schematic,
            // A schematic's board is a dependency of the schematic too; it is listed below,
            // as what uses the schematic.
            (DesignKind::Circuit, Some(DesignKind::Layout)) => continue,
            (_, Some(k)) if k.is_model() => Relation::Uses,
            _ => continue,
        };
        seen.push(dep.clone());
        let (status, level, source) = status_of(&dep, files, ran);
        out.push(Link {
            path: dep,
            relation,
            via: None,
            status,
            level,
            source,
        });
    }
    users(path, files, None, &mut seen, &mut out, ran);
    // A schematic's board brings the board's studies and enclosure; a board's schematic is
    // already its link.
    let boards: Vec<String> = out
        .iter()
        .filter(|l| l.relation == Relation::Board)
        .map(|l| l.path.clone())
        .collect();
    for board in boards {
        users(&board, files, Some(&board), &mut seen, &mut out, ran);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_study_names_its_geometry_files_relative_to_itself() {
        let text = "geometry bracket {\n    file = \"../parts/bracket.part\"\n    units = mm\n}\n";
        assert_eq!(
            study_geometry_files("/studies/load.fea", text),
            ["/parts/bracket.part"]
        );
        let (status, level) = study_status("study s { }\n");
        assert_eq!(level, Level::Ok, "{status}");
        assert!(status.contains("1 study: s"), "{status}");
    }

    #[test]
    fn candidates_are_design_files_and_studies_but_not_locks() {
        let project: Vec<String> = [
            "/a.part",
            "/a.circuit.lock",
            "/b.fea",
            "/c.m",
            "/d.circuit.layout",
            "/a.part",
        ]
        .map(String::from)
        .to_vec();
        assert_eq!(
            related_candidates("/a.part", &project),
            ["/b.fea", "/d.circuit.layout"]
        );
    }
}
