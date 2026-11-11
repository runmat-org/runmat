//! The overview's next steps: two or three that fit the file's state, each a guide or a catalog
//! command, most pressing first. Each step names the fact or link that makes it fit.

use super::{name_of, plural, Fact, Level, Link, NextStep, OpenDecision, Relation, StepAction};
use crate::guides::GuideCatalog;
use crate::DesignKind;

/// The most steps an overview offers.
pub const MAX_STEPS: usize = 3;

fn guide(label: impl Into<String>, because: &str, id: &str) -> NextStep {
    NextStep {
        label: label.into(),
        because: because.into(),
        action: StepAction::Guide { id: id.into() },
    }
}

fn command(label: impl Into<String>, because: &str, id: &str) -> NextStep {
    NextStep {
        label: label.into(),
        because: because.into(),
        action: StepAction::Command { id: id.into() },
    }
}

/// The leading count of a fact's value (`3 changes from …` → 3).
fn leading_count(f: &Fact) -> usize {
    f.value
        .split_whitespace()
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

pub(super) fn next_steps(
    kind: DesignKind,
    state: &[Fact],
    links: &[Link],
    open: &[OpenDecision],
    catalog: &GuideCatalog,
) -> Vec<NextStep> {
    let fact = |id: &str| state.iter().find(|f| f.id == id);
    let failing = |id: &str| fact(id).is_some_and(|f| f.level == Level::Error);
    let warning = |id: &str| fact(id).is_some_and(|f| f.level == Level::Warning);
    let has = |r: Relation| links.iter().any(|l| l.relation == r);
    let open_in = |g: &str| open.iter().any(|o| o.guide == g);
    let mut steps: Vec<NextStep> = Vec::new();
    match kind {
        DesignKind::Circuit | DesignKind::Layout => {
            if failing("erc") {
                steps.push(if open_in("schematic_capture") {
                    guide(
                        "Settle where the power rails come from",
                        "erc",
                        "schematic_capture",
                    )
                } else {
                    command(
                        "Review the electrical rule findings",
                        "erc",
                        "schematic.erc",
                    )
                });
            }
            if let Some(f) = fact("board_sync").filter(|f| f.level == Level::Warning) {
                let n = leading_count(f);
                steps.push(guide(
                    format!(
                        "Review the {} for the board",
                        plural(n, "schematic change", "schematic changes")
                    ),
                    "board_sync",
                    "engineering_change",
                ));
            }
            if let Some(f) = fact("placement").filter(|f| f.level == Level::Warning) {
                steps.push(guide(
                    format!(
                        "Place the {} not on the board",
                        plural(leading_count(f), "part", "parts")
                    ),
                    "placement",
                    "board_bring_up",
                ));
            }
            if fact("board_sync").is_some_and(|f| f.level == Level::Info) {
                steps.push(guide("Lay out the board", "board_sync", "board_bring_up"));
            }
            if failing("drc") {
                steps.push(command("Review the design rule errors", "drc", "board.drc"));
            }
            if fact("simulation").is_some_and(|f| f.value != "No simulation section") {
                steps.push(command("Run the simulation", "simulation", "simulate.run"));
            }
            let board = kind == DesignKind::Layout || has(Relation::Board);
            if board && !has(Relation::Study) {
                steps.push(guide(
                    "Set up a thermal study of the board",
                    "links",
                    "coupled_study",
                ));
            }
        }
        DesignKind::Part | DesignKind::Assembly => {
            if failing("regeneration") || failing("features") {
                steps.push(command(
                    "Fix what fails to regenerate",
                    "regeneration",
                    "agent.fix",
                ));
            }
            match links.iter().find(|l| l.relation == Relation::Study) {
                Some(study) => steps.push(command(
                    format!("Open the study {}", name_of(&study.path)),
                    "links",
                    "simulate.studies",
                )),
                None => steps.push(guide(
                    if kind == DesignKind::Part {
                        "Set up a simulation of this part"
                    } else {
                        "Set up a simulation of this assembly"
                    },
                    "links",
                    "simulation_setup",
                )),
            }
            if !has(Relation::Drawing) {
                steps.push(guide(
                    "Make a drawing for release",
                    "links",
                    "drawing_for_release",
                ));
            }
            if kind == DesignKind::Part {
                steps.push(guide(
                    "Make it manufacturable",
                    "regeneration",
                    "manufacturable",
                ));
            } else if open_in("assembly_from_parts") {
                steps.push(guide(
                    "Finish how the parts mate",
                    "open_decisions",
                    "assembly_from_parts",
                ));
            }
        }
        DesignKind::Drawing => {
            if failing("regeneration") {
                steps.push(command(
                    "Fix what fails on the sheet",
                    "regeneration",
                    "agent.fix",
                ));
            }
            if warning("annotations") {
                steps.push(guide(
                    "Choose the critical dimensions",
                    "annotations",
                    "drawing_for_release",
                ));
            }
            steps.push(command(
                "Add the hole callouts",
                "annotations",
                "drawing.hole-callout",
            ));
        }
    }
    // A guide with decisions still open, when no step above starts it.
    for o in open {
        if !steps.iter().any(|s| {
            s.action
                == StepAction::Guide {
                    id: o.guide.clone(),
                }
        }) {
            steps.push(guide(
                format!("Continue {}", o.guide_title.to_lowercase()),
                "open_decisions",
                &o.guide,
            ));
        }
    }
    let mut out: Vec<NextStep> = Vec::new();
    for s in steps {
        let known = match &s.action {
            StepAction::Guide { id } => catalog.guide(id).is_some(),
            StepAction::Command { .. } => true,
        };
        if known && !out.iter().any(|o| o.action == s.action) {
            out.push(s);
        }
    }
    out.truncate(MAX_STEPS);
    out
}
