//! A KiCad import's net classes and design rules (PCB-03): the project file's (`.kicad_pro`)
//! classes and their net assignments, and the custom rules file's (`.kicad_dru`) rules.
//!
//! - The `Default` class sets the board's limits (clearance, track width, via size and drill):
//!   KiCad applies it to every net no other class takes.
//! - Every other class becomes a circuit net class and the layout's `net_class` parameters
//!   (clearance, track width, via diameter and drill, pair width and gap), and each net takes
//!   its class by the project's assignments and patterns, first match first.
//! - Custom rules become layout rules in file order; rules with conditions RunMat doesn't read
//!   are listed as not carried.

use crate::board_rules::{props, RuleValue};
use crate::geom::Coord;
use crate::import::Report;
use crate::ir::{Design, NetClass, NetDecl, Param, Tolerance};
use crate::kicad::project::{KiNetClass, NetSettings, Project};
use crate::layout::{Board, NetClassRules};
use runmat_rules::Quantity;
use std::collections::BTreeMap;

/// The project's net-class settings and custom rules, read from `files`.
#[derive(Default)]
pub(crate) struct ProjectRules {
    pub settings: Option<NetSettings>,
    pub rules: Vec<crate::board_rules::RuleSpec>,
}

const DEFAULT: &str = "Default";

/// The class fields RunMat carries, with the rule property each sets.
fn fields(c: &KiNetClass) -> Vec<(&'static str, Coord)> {
    [
        (props::CLEARANCE, c.clearance),
        (props::TRACK_WIDTH, c.track_width),
        (props::VIA_DIAMETER, c.via_diameter),
        (props::VIA_DRILL, c.via_drill),
        (props::DIFF_WIDTH, c.diff_pair_width),
        (props::DIFF_GAP, c.diff_pair_gap),
    ]
    .into_iter()
    .filter_map(|(k, v)| v.filter(|v| *v > 0).map(|v| (k, v)))
    .collect()
}

/// Reads the `.kicad_pro` and `.kicad_dru` of `files`, reporting what isn't carried.
pub(crate) fn read(files: &BTreeMap<String, String>, report: &mut Report) -> Result<ProjectRules, String> {
    let mut out = ProjectRules::default();
    if let Some((name, text)) = files.iter().find(|(k, _)| k.ends_with(".kicad_pro")) {
        let settings = Project::read(text).map_err(|e| format!("{name}: {e}"))?.net_settings();
        for c in &settings.classes {
            if c.microvia_diameter.is_some() || c.microvia_drill.is_some() {
                report.not_carried(format!("net class {}: its microvia size", c.name));
            }
            if c.diff_pair_via_gap.is_some() {
                report.not_carried(format!("net class {}: its pair via gap", c.name));
            }
        }
        report.count("net classes", settings.classes.iter().filter(|c| c.name != DEFAULT).count());
        out.settings = Some(settings);
    }
    for (name, text) in files.iter().filter(|(k, _)| k.ends_with(".kicad_dru")) {
        let imp = crate::kicad::dru::read(text, name, 2).map_err(|e| format!("{name}: {e}"))?;
        for (rule, why) in &imp.unsupported {
            report.not_carried(format!("design rule {rule}: {why}"));
        }
        let before = out.rules.len();
        for spec in imp.rules {
            add_rule(&mut out.rules, spec);
        }
        report.count("design rules", out.rules.len() - before);
    }
    Ok(out)
}

/// Adds one constraint of a custom rule. A KiCad rule holds several constraints, read as one
/// spec each; layout rules are named uniquely, so constraints on the same items join one rule
/// and a constraint on other items (a pair constraint beside a single one) takes the rule's
/// name with its property (`usb pairs skew_max`).
fn add_rule(rules: &mut Vec<crate::board_rules::RuleSpec>, mut spec: crate::board_rules::RuleSpec) {
    if let Some(same) = rules.iter_mut().find(|r| r.id == spec.id && r.selector == spec.selector && r.other == spec.other) {
        same.decls.extend(spec.decls);
        return;
    }
    if rules.iter().any(|r| r.id == spec.id) {
        let props: Vec<&str> = spec.decls.iter().map(|(k, _)| k.as_str()).collect();
        spec.id = format!("{} {}", spec.id, props.join(" "));
    }
    rules.push(spec);
}

/// The board's limits from the `Default` class, its other classes as `net_class` parameters,
/// each net's class, and the custom rules.
pub(crate) fn apply_to_board(rules: &ProjectRules, board: &mut Board, report: &mut Report) {
    if let Some(settings) = &rules.settings {
        for c in &settings.classes {
            if c.name == DEFAULT {
                for (k, v) in fields(c) {
                    match k {
                        props::CLEARANCE => board.limits.min_clearance = v,
                        props::TRACK_WIDTH => board.limits.min_track = v,
                        props::VIA_DIAMETER => board.limits.min_via_diameter = v,
                        props::VIA_DRILL => board.limits.min_drill = v,
                        _ => report.not_carried(format!("net class Default: its {k}")),
                    }
                }
                continue;
            }
            let decls: Vec<_> = fields(c).into_iter().map(|(k, v)| (k.to_owned(), RuleValue::Len(v))).collect();
            board.net_class_rules.insert(c.name.clone(), NetClassRules { decls, ..Default::default() });
        }
        let mut assigned = 0;
        for n in &mut board.nets {
            let class = settings.class_of(&n.name);
            if class != DEFAULT {
                n.class = Some(class.to_owned());
                assigned += 1;
            }
        }
        report.count("nets with a class", assigned);
    }
    board.rules.extend(rules.rules.iter().cloned());
}

/// The circuit's net classes and each net's class. A net's KiCad name is `/<name>` on the
/// root sheet and its own name when global; a net on a lower sheet takes the class its board
/// namesake (`/<sheet path>/<name>`) has, when the board gives one.
pub(crate) fn apply_to_design(rules: &ProjectRules, design: &mut Design, board: Option<&Board>, report: &mut Report) {
    let Some(settings) = &rules.settings else { return };
    for c in settings.classes.iter().filter(|c| c.name != DEFAULT) {
        let params: BTreeMap<String, Param> = fields(c)
            .into_iter()
            .map(|(k, v)| (k.to_owned(), Param::quantity(Quantity::length_m(v as f64 * 1e-9), Tolerance::Exact)))
            .collect();
        design.net_classes.insert(c.name.clone(), NetClass { name: c.name.clone(), params, source: Default::default() });
    }
    let root = design.root.clone();
    let board_class = |name: &str| -> Option<String> {
        let b = board?;
        let mut classes = b.nets.iter().filter(|n| n.name.rsplit('/').next() == Some(name) && n.name.contains('/')).map(|n| n.class.clone());
        let first = classes.next()?;
        classes.all(|c| c == first).then_some(first).flatten()
    };
    let mut unresolved = 0;
    for (file, sheet) in &mut design.sheets {
        let mut names: Vec<(String, bool)> = sheet.nets.iter().map(|n| (n.name.clone(), n.global)).collect();
        for c in &sheet.connections {
            for e in &c.ends {
                if let crate::ir::Endpoint::Net(n) = e {
                    if !names.iter().any(|(m, _)| m == n) {
                        names.push((n.clone(), false));
                    }
                }
            }
        }
        for (name, global) in names {
            let class = if global {
                Some(settings.class_of(&name).to_owned())
            } else if *file == root {
                Some(settings.class_of(&format!("/{name}")).to_owned())
            } else {
                let c = board_class(&name);
                if c.is_none() && board.is_none() && settings.classes.iter().any(|c| c.name != DEFAULT) {
                    unresolved += 1;
                }
                c
            };
            let Some(class) = class.filter(|c| c != DEFAULT) else { continue };
            match sheet.nets.iter_mut().find(|n| n.name == name) {
                Some(n) => n.class = Some(class),
                None => sheet.nets.push(NetDecl {
                    name,
                    class: Some(class),
                    global: false,
                    params: Default::default(),
                    renamed_from: None,
                    source: Default::default(),
                }),
            }
        }
    }
    if unresolved > 0 {
        report.approximated(format!("{unresolved} nets on lower sheets keep the Default class: without a board their KiCad names aren't known"));
    }
}
