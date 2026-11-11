//! Guides: declared, resumable flows the agent runs in the panel (research doc 18, §2.2).
//!
//! A guide has a goal, the decisions it needs, the file it writes and a completion check
//! through the real compilers. The agent picks the next question from the context and the
//! decisions still open; nothing here scripts the order. The Desktop renders guides, their
//! question cards and the composer's suggestions from [`catalog`], written to
//! `desktop/src/generated/design-guides.json` the way the command catalog is.
//!
//! [`status`] runs a guide's completion check over the workspace files: [`crate::check`] on the
//! target, then the counts each requirement names. The open decisions it returns are the
//! questions the guide still has to ask.

use crate::check::{check, CheckInput};
use crate::inspect::{inspect, InspectInput};
use crate::{DesignKind, Files, ToolError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod suggestions;
pub use suggestions::{FileState, Suggestion, SuggestionNames, MAX_PANEL_SUGGESTIONS};

/// The card controls every guide and the conversation share (doc 18, §2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
/// The names are the card contract's (`InteractionControlKind` in the agent harness protocol),
/// so a guide's declared controls are the controls its cards use.
pub enum ControlKind {
    /// One choice, with Other in your own words.
    SingleChoice,
    /// Several choices, with Other in your own words.
    MultipleChoice,
    /// A value with a unit and tolerance: a literal, an expression, a link or a published value.
    Quantity,
    /// A direction in a coordinate frame.
    Direction,
    /// Entities picked in the editor; the decision's `pick` filter says which kinds.
    EditorPick,
    /// A file or region from the project.
    ProjectFile,
    /// A proposed change shown as its diff, with Keep and Undo.
    ChangeReview,
}

impl ControlKind {
    pub const ALL: [ControlKind; 7] = [
        Self::SingleChoice,
        Self::MultipleChoice,
        Self::Quantity,
        Self::Direction,
        Self::EditorPick,
        Self::ProjectFile,
        Self::ChangeReview,
    ];
}

/// What an editor pick may select.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PickEntity {
    Face,
    Edge,
    Body,
    SketchEntity,
    Occurrence,
    View,
    Dimension,
    Sheet,
    Component,
    Pin,
    Pad,
    Net,
    Room,
}

/// How the agent settles a decision before it asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DefaultPolicy {
    /// Always asked: the agent has no basis to choose.
    Ask,
    /// The agent proposes a value with its reason; the card confirms or changes it.
    Propose,
    /// Read from the file or the selection; asked only when it can't be read.
    Infer,
}

/// One decision a guide needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Decision {
    pub id: String,
    /// The question as the card asks it.
    pub question: String,
    /// The controls the card may use, in order of preference.
    pub controls: Vec<ControlKind>,
    /// For an editor pick: the entity kinds the pick filter allows.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pick: Vec<PickEntity>,
    pub required: bool,
    pub default_policy: DefaultPolicy,
    /// What Help me decide offers: concrete alternatives, in one sentence.
    pub help: String,
}

/// Where a guide opens from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EntryPoint {
    /// Opening a file with one of these extensions.
    OpenFile { extensions: Vec<String> },
    /// Creating a new file of this extension.
    NewFile { extension: String },
    /// A catalog command (an id in `design-commands.json`).
    Command { id: String },
    /// The agent proposes it from the conversation.
    Agent,
}

/// A count from `design.check` (or `design.inspect`'s totals) the finished file must reach, or
/// stay under, and the decision that settles it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Requirement {
    /// A key of `design.check`'s `counts` (features, bodies, published, occurrences, mates,
    /// views, annotations, components, nets, footprints, tracks) or of `design.inspect`'s totals
    /// (`erc power errors`, `eco differences`, `unplaced components`).
    pub count: String,
    pub at_least: usize,
    /// The most the finished file may have (`Some(0)`: none left).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at_most: Option<usize>,
    /// The decision that is still open while the count is short, or over.
    pub decision: String,
}

/// Project files a guide works from. The Guide tab offers a guide only when every need is met,
/// counting every file in the project, not just the open tabs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Need {
    /// A file counts when its name ends with one of these extensions.
    pub extensions: Vec<String>,
    /// How many such files the project must hold.
    pub at_least: usize,
}

/// How a guide knows it is done.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Completion {
    /// The compiler check: `design.check` for design files, `runmat.check` for `.fea`.
    pub tool: String,
    /// What the checked file must contain beyond compiling cleanly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<Requirement>,
}

/// One guide.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Guide {
    pub id: String,
    pub title: String,
    /// What it gets done, in one line, for the guide list.
    pub summary: String,
    /// The dialect it writes.
    pub target: String,
    /// The file extension it writes.
    pub extension: String,
    pub opens_from: Vec<EntryPoint>,
    /// Inputs the project must already hold; empty means the guide is always offered.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub needs: Vec<Need>,
    pub decisions: Vec<Decision>,
    pub completion: Completion,
    /// The model the guide's turns use unless Settings overrides it.
    pub default_model: String,
    /// The harness turn kind: `fea` for guides that write `.fea`, `general` otherwise.
    pub turn_usage: String,
    /// The composer text that starts it from the guide list.
    pub start_prompt: String,
}

/// The guides and the composer suggestions, as the Desktop loads them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GuideCatalog {
    pub format: String,
    pub format_version: u32,
    pub controls: Vec<ControlKind>,
    pub guides: Vec<Guide>,
    pub suggestions: Vec<Suggestion>,
}

impl Guide {
    /// Offer this guide only when the project holds at least `at_least` files with one of
    /// `extensions`.
    fn needing(mut self, extensions: &[&str], at_least: usize) -> Self {
        self.needs.push(Need {
            extensions: extensions.iter().map(|e| (*e).to_owned()).collect(),
            at_least,
        });
        self
    }
}

impl GuideCatalog {
    pub fn guide(&self, id: &str) -> Option<&Guide> {
        self.guides.iter().find(|g| g.id == id)
    }

    /// Deterministic JSON, two-space indented, with a trailing newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("guide catalog serializes");
        s.push('\n');
        s
    }

    /// Problems in the declarations: duplicate ids, requirements naming unknown decisions,
    /// picks without an editor-pick control, commands missing from `command_ids`, more suggestions
    /// for an editor and file state than a placeholder and [`MAX_PANEL_SUGGESTIONS`], and
    /// suggestion texts whose `{1}`, `{2}`, … and `names` disagree.
    pub fn lint(&self, command_ids: &[&str]) -> Vec<String> {
        let mut problems = Vec::new();
        let mut ids: Vec<&str> = self.guides.iter().map(|g| g.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != self.guides.len() {
            problems.push("guide ids repeat".into());
        }
        for g in &self.guides {
            let mut d: Vec<&str> = g.decisions.iter().map(|d| d.id.as_str()).collect();
            d.sort_unstable();
            d.dedup();
            if d.len() != g.decisions.len() {
                problems.push(format!("{}: decision ids repeat", g.id));
            }
            for r in &g.completion.requires {
                if !g.decisions.iter().any(|d| d.id == r.decision) {
                    problems.push(format!("{}: requirement names {}", g.id, r.decision));
                }
            }
            for dec in &g.decisions {
                if dec.controls.is_empty() {
                    problems.push(format!("{}.{}: no control", g.id, dec.id));
                }
                if dec.pick.is_empty() == dec.controls.contains(&ControlKind::EditorPick) {
                    problems.push(format!(
                        "{}.{}: pick filter and editor pick disagree",
                        g.id, dec.id
                    ));
                }
            }
            for e in &g.opens_from {
                if let EntryPoint::Command { id } = e {
                    if !command_ids.contains(&id.as_str()) {
                        problems.push(format!("{}: command {id} isn't in the catalog", g.id));
                    }
                }
            }
        }
        for editor in self.suggestions.iter().map(|s| s.editor.as_str()) {
            for state in FileState::ALL {
                let shown = self
                    .suggestions
                    .iter()
                    .filter(|s| s.editor == editor && s.selection.is_none())
                    .filter(|s| s.state.is_none_or(|st| st == state))
                    .count();
                if shown > MAX_PANEL_SUGGESTIONS + 1 {
                    problems.push(format!(
                        "{editor} ({state:?}): {shown} suggestions, more than a placeholder and {MAX_PANEL_SUGGESTIONS}"
                    ));
                }
            }
        }
        for s in &self.suggestions {
            if s.text.contains("{1}") != s.names.is_some() {
                problems.push(format!(
                    "suggestion {:?}: names and {{1}}, {{2}}, … in the text disagree",
                    s.text
                ));
            }
            if let Some(id) = &s.command {
                if !command_ids.contains(&id.as_str()) {
                    problems.push(format!(
                        "suggestion {:?}: command {id} isn't in the catalog",
                        s.text
                    ));
                }
            }
        }
        problems
    }
}

/// Where a guide stands on a file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GuideStatus {
    pub guide: String,
    pub path: String,
    /// The file compiles with no errors.
    pub check_ok: bool,
    /// Compiles and meets every requirement.
    pub complete: bool,
    /// Decisions the file doesn't show yet, in declaration order.
    pub open: Vec<String>,
    /// Decisions settled, out of `total`.
    pub settled: usize,
    pub total: usize,
    pub summary: String,
}

/// Runs the completion check of `guide` on `path`. Guides that write `.fea` are checked by
/// `runmat check` in the runtime and return `unsupported` here.
pub fn status(guide: &Guide, path: &str, files: &Files) -> Result<GuideStatus, ToolError> {
    if guide.completion.tool != "design.check" {
        return Err(ToolError::new(
            "unsupported",
            format!("{} is checked by {}", guide.id, guide.completion.tool),
        ));
    }
    if DesignKind::of(path).is_none() {
        return Err(ToolError::invalid(format!("{path} isn't a design file")));
    }
    let out = check(
        &CheckInput {
            path: path.to_owned(),
        },
        files,
    )?;
    let check_ok = out.ok && out.errors == 0;
    // `design.check` counts what it compiles; a board's footprints and tracks come from
    // `design.inspect`, which reads the layout with its schematic.
    let mut totals = out.counts.clone();
    if guide
        .completion
        .requires
        .iter()
        .any(|r| !totals.contains_key(&r.count))
    {
        let input = InspectInput {
            path: path.to_owned(),
            sections: None,
            filter: None,
            density: None,
            set: None,
            limit: Some(1),
            changed: None,
            rays: None,
            contact: None,
            up: None,
        };
        if let Ok(seen) = inspect(&input, files) {
            for (k, v) in seen.totals {
                totals.entry(k).or_insert(v);
            }
        }
    }
    let mut open: Vec<String> = Vec::new();
    for r in &guide.completion.requires {
        let have = totals.get(&r.count).copied().unwrap_or(0);
        let over = r.at_most.is_some_and(|most| have > most);
        if (have < r.at_least || over) && !open.contains(&r.decision) {
            open.push(r.decision.clone());
        }
    }
    let total = guide.decisions.len();
    let settled = total - open.len();
    let complete = check_ok && open.is_empty();
    let summary = if complete {
        format!("{} is complete", guide.title)
    } else if !check_ok {
        format!("{}: {}", guide.title, out.summary)
    } else {
        format!("{}: {settled} of {total} decided", guide.title)
    };
    Ok(GuideStatus {
        guide: guide.id.clone(),
        path: path.to_owned(),
        check_ok,
        complete,
        open,
        settled,
        total,
        summary,
    })
}

fn decision(
    id: &str,
    question: &str,
    controls: &[ControlKind],
    pick: &[PickEntity],
    required: bool,
    default_policy: DefaultPolicy,
    help: &str,
) -> Decision {
    Decision {
        id: id.into(),
        question: question.into(),
        controls: controls.to_vec(),
        pick: pick.to_vec(),
        required,
        default_policy,
        help: help.into(),
    }
}

fn needs(count: &str, at_least: usize, decision: &str) -> Requirement {
    Requirement {
        count: count.into(),
        at_least,
        at_most: None,
        decision: decision.into(),
    }
}

/// None of `count` may remain: `decision` stays open while any does.
fn needs_none(count: &str, decision: &str) -> Requirement {
    Requirement {
        count: count.into(),
        at_least: 0,
        at_most: Some(0),
        decision: decision.into(),
    }
}

fn open_file(extensions: &[&str]) -> EntryPoint {
    EntryPoint::OpenFile {
        extensions: extensions.iter().map(|e| (*e).to_owned()).collect(),
    }
}

fn new_file(extension: &str) -> EntryPoint {
    EntryPoint::NewFile {
        extension: extension.into(),
    }
}

fn command(id: &str) -> EntryPoint {
    EntryPoint::Command { id: id.into() }
}

/// The model guide turns start on: fast and lower cost for one structured question at a time.
pub const GUIDE_DEFAULT_MODEL: &str = "gpt-6-luna";

/// Every guide, in the order of doc 18's table, and the composer suggestions.
pub fn catalog() -> GuideCatalog {
    use ControlKind::*;
    use DefaultPolicy::*;
    use PickEntity::*;
    let guide = |id: &str,
                 title: &str,
                 summary: &str,
                 target: &str,
                 extension: &str,
                 opens_from: Vec<EntryPoint>,
                 decisions: Vec<Decision>,
                 tool: &str,
                 requires: Vec<Requirement>,
                 turn_usage: &str,
                 start_prompt: &str| Guide {
        id: id.into(),
        title: title.into(),
        summary: summary.into(),
        target: target.into(),
        extension: extension.into(),
        opens_from,
        needs: Vec::new(),
        decisions,
        completion: Completion {
            tool: tool.into(),
            requires,
        },
        default_model: GUIDE_DEFAULT_MODEL.into(),
        turn_usage: turn_usage.into(),
        start_prompt: start_prompt.into(),
    };
    let guides = vec![
        guide(
            "simulation_setup",
            "Simulation setup",
            "Set up a study: goal, supports, loads, material and the acceptance limit",
            "fea",
            ".fea",
            vec![
                open_file(&[".step", ".stp", ".stl", ".part", ".assembly", ".fea"]),
                command("study.new"),
                EntryPoint::Agent,
            ],
            vec![
                decision("goal", "What would you like to simulate?", &[SingleChoice], &[], true, Ask,
                    "Strength under load, stiffness, vibration modes or temperature, with what each answers"),
                decision("supports", "Which faces hold the part?", &[EditorPick, SingleChoice], &[Face], true, Ask,
                    "Bolted faces, a clamped edge or a symmetry plane, with how each constrains the part"),
                decision("loads", "Where does the load act, and how large is it?", &[EditorPick, Quantity, Direction], &[Face], true, Ask,
                    "A force on a face, a pressure, or a remote load through a point, with typical magnitudes"),
                decision("material", "What material is it?", &[SingleChoice, Quantity], &[], true, Propose,
                    "Common alloys for the process, with yield strength and density"),
                decision("acceptance", "What result passes?", &[Quantity, SingleChoice], &[], true, Propose,
                    "A safety factor on yield, a deflection limit or a minimum frequency"),
                decision("mesh", "How fine should the mesh be?", &[SingleChoice, Quantity], &[], false, Propose,
                    "Coarse for a quick check, refined near holes and fillets for stress"),
                decision("run", "Run the study?", &[ChangeReview], &[], true, Propose,
                    "Run now on this machine, or keep the setup and run later"),
            ],
            "runmat.check",
            vec![],
            "fea",
            "Set up a simulation of this part",
        ),
        guide(
            "part_from_requirements",
            "New part from requirements",
            "Model a part from its function, envelope, interfaces, material and process",
            "part",
            ".part",
            vec![new_file(".part"), command("assembly.new-part"), EntryPoint::Agent],
            vec![
                decision("function", "What does the part do?", &[SingleChoice], &[], true, Ask,
                    "Mount, bracket, enclosure, shaft or plate, with the features each usually needs"),
                decision("envelope", "What space must it fit in?", &[Quantity], &[], true, Ask,
                    "Width, depth and height limits, from a mating part or a drawing"),
                decision("interfaces", "Which faces mount to the mating part?", &[EditorPick, ProjectFile], &[Face, Edge, SketchEntity], true, Ask,
                    "Faces of a mating part, a hole pattern, or a sketch on the base plane"),
                decision("material", "What material?", &[SingleChoice], &[], true, Propose,
                    "Aluminum 6061, steel 1018, stainless 304 or ABS, by strength, weight and cost"),
                decision("process", "How will it be made?", &[SingleChoice], &[], true, Propose,
                    "Machined, sheet metal, printed or cast, with the rules each brings"),
                decision("published", "Which values should other files use?", &[MultipleChoice], &[], false, Propose,
                    "Hole spacing, thickness and mounting faces, published for assemblies and drawings"),
            ],
            "design.check",
            vec![needs("features", 1, "envelope"), needs("bodies", 1, "function"), needs("regions", 1, "interfaces"), needs("published", 1, "published")],
            "general",
            "Make a new part from requirements",
        ),
        guide(
            "manufacturable",
            "Make it manufacturable",
            "Bring a part to its process: wall, draft, radii and tolerances",
            "part",
            ".part",
            vec![open_file(&[".part"]), command("evaluate.draft"), command("evaluate.check-geometry"), EntryPoint::Agent],
            vec![
                decision("process", "Which process will make it?", &[SingleChoice], &[], true, Infer,
                    "Machined, sheet metal, printed or molded, read from the part's features when it can be"),
                decision("min_wall", "Minimum wall thickness?", &[Quantity], &[], true, Propose,
                    "The process minimum for this material, with where the part is thinner"),
                decision("draft", "Which walls need draft, and how much?", &[EditorPick, Quantity], &[Face], false, Propose,
                    "1 to 3 degrees on walls parallel to the pull direction"),
                decision("radii", "Minimum inside bend radius?", &[Quantity], &[], true, Propose,
                    "1 × thickness for 5052-H32, 1.5 × for 6061-T6, with the cracking risk below it"),
                decision("tolerances", "Which features carry tight tolerances?", &[EditorPick, Quantity], &[Face, Edge], false, Ask,
                    "Bores, mating faces and hole positions; everything else at the general tolerance"),
            ],
            "design.check",
            vec![needs("features", 1, "process")],
            "general",
            "Make this part manufacturable",
        ).needing(&[".part"], 1),
        guide(
            "assembly_from_parts",
            "Assembly from parts",
            "Place parts, mate them, fix one and set the motion",
            "assembly",
            ".assembly",
            vec![new_file(".assembly"), command("assembly.mate"), EntryPoint::Agent],
            vec![
                decision("parts", "Which parts go in?", &[ProjectFile, MultipleChoice], &[], true, Ask,
                    "Parts in this project, with the ones other assemblies already use"),
                decision("fixed", "Which part stays fixed?", &[EditorPick, SingleChoice], &[Occurrence], true, Propose,
                    "The base or frame, usually the largest part"),
                decision("mates", "How do the parts attach?", &[SingleChoice, EditorPick], &[Face, Edge], true, Ask,
                    "Bolted, welded or snap fit, with the mates each one adds"),
                decision("motion", "Should anything move?", &[SingleChoice], &[], false, Ask,
                    "Nothing, a hinge, a slider, or a mate controller over a range"),
            ],
            "design.check",
            vec![needs("occurrences", 2, "parts"), needs("mates", 1, "mates")],
            "general",
            "Build an assembly from these parts",
        ).needing(&[".part"], 1),
        guide(
            "drawing_for_release",
            "Drawing for release",
            "A drawing ready to release: sheet, views, critical dimensions, tolerances and notes",
            "drawing",
            ".drawing",
            vec![new_file(".drawing"), command("drawing.auto"), EntryPoint::Agent],
            vec![
                decision("sheet", "Which sheet size and standard?", &[SingleChoice], &[], true, Propose,
                    "A3 or ANSI B, ISO 128 or ASME Y14.5, from the part's size and the project"),
                decision("views", "Which views?", &[MultipleChoice], &[], true, Propose,
                    "Front, top, right and isometric, plus a section through the holes"),
                decision("critical", "Which dimensions are critical?", &[EditorPick, MultipleChoice], &[Dimension, Edge, Face], true, Ask,
                    "Hole positions, mating faces and overall size, dimensioned from the datums"),
                decision("tolerances", "What tolerances apply?", &[Quantity, SingleChoice], &[], true, Propose,
                    "ISO 2768-m generally, with GD&T on the critical features"),
                decision("notes", "Which notes go on the sheet?", &[MultipleChoice], &[], false, Propose,
                    "Material, finish, break sharp edges and the general tolerance"),
            ],
            "design.check",
            vec![needs("sheets", 1, "sheet"), needs("views", 1, "views"), needs("annotations", 1, "critical")],
            "general",
            "Make a release drawing of this part",
        ).needing(&[".part", ".assembly"], 1),
        guide(
            "schematic_capture",
            "Schematic capture",
            "Purpose, blocks, interfaces, power rails, parts and alternates",
            "circuit",
            ".circuit",
            vec![new_file(".circuit"), EntryPoint::Agent],
            vec![
                decision("purpose", "What must the circuit do?", &[SingleChoice], &[], true, Ask,
                    "A sensor node, motor driver, power supply, USB device or radio module, with the blocks each needs"),
                decision("blocks", "Which blocks does the circuit have?", &[MultipleChoice], &[], true, Ask,
                    "Power, MCU, radio, sensors and connectors, with a part for each"),
                decision("interfaces", "How do the blocks connect?", &[SingleChoice, EditorPick], &[Net, Pin], true, Ask,
                    "I2C, SPI, UART or GPIO, with pull-ups and level shifting where needed"),
                decision("rails", "Which rail powers each block?", &[SingleChoice, EditorPick], &[Net], true, Ask,
                    "The rails on the sheet, with each one's voltage and the radio's supply range"),
                decision("parts", "Which parts, and which alternates?", &[SingleChoice, ProjectFile, EditorPick], &[Component], true, Propose,
                    "Parts in the library with stock and a second source each"),
            ],
            "design.check",
            vec![needs("components", 1, "purpose"), needs("components", 1, "blocks"), needs("nets", 1, "interfaces"), needs_none("erc power errors", "rails")],
            "general",
            "Capture a schematic from a block diagram",
        ),
        guide(
            "board_bring_up",
            "Board bring-up",
            "Outline, stack-up, rules from net classes and placement",
            "circuit_layout",
            ".circuit.layout",
            vec![
                open_file(&[".circuit"]),
                new_file(".circuit.layout"),
                command("schematic.update-board"),
                command("board.room"),
                EntryPoint::Agent,
            ],
            vec![
                decision("outline", "What is the board outline?", &[Quantity, ProjectFile], &[], true, Ask,
                    "A rectangle with corner radii, or the enclosure's outline from its part"),
                decision("stackup", "How many layers, and which stack-up?", &[SingleChoice], &[], true, Propose,
                    "2 layers for simple boards, 4 with solid ground for RF and fast edges"),
                decision("rules", "Which rules come from the net classes?", &[Quantity, SingleChoice], &[], true, Propose,
                    "Track width and clearance per class, from current and voltage"),
                decision("placement", "How are repeated channels placed?", &[ChangeReview, EditorPick], &[Room, Component], true, Propose,
                    "Copy Room Formats from channel A to B, or place each by hand"),
            ],
            "design.check",
            vec![needs("footprints", 1, "placement"), needs_none("unplaced components", "placement")],
            "general",
            "Bring up a board from this schematic",
        ).needing(&[".circuit"], 1),
        guide(
            "engineering_change",
            "Engineering change (ECO)",
            "Which schematic differences reach the board, with re-links and rooms",
            "circuit_layout",
            ".circuit.layout",
            vec![command("board.update-schematic"), command("schematic.update-board"), EntryPoint::Agent],
            vec![
                decision("differences", "Apply the schematic's changes to the board?", &[ChangeReview, MultipleChoice, EditorPick], &[Component], true, Propose,
                    "Apply all, apply the footprint changes only, or review each row"),
                decision("relinks", "Re-link the parts whose designators changed?", &[SingleChoice], &[], false, Propose,
                    "Keep placement by unique id, or place them again"),
                decision("rooms", "Update the rooms for the changed parts?", &[SingleChoice], &[], false, Propose,
                    "Move changed parts into their rooms, or leave them where they are"),
            ],
            "design.check",
            vec![needs_none("eco differences", "differences")],
            "general",
            "Update the board from the schematic",
        ).needing(&[".circuit.layout"], 1),
        guide(
            "coupled_study",
            "Coupled study",
            "A board's thermal or vibration study: heat sources, enclosure and environment",
            "fea",
            ".fea",
            vec![open_file(&[".circuit.layout"]), command("simulate.studies"), EntryPoint::Agent],
            vec![
                decision("sources", "Which parts give off heat, and how much?", &[EditorPick, Quantity], &[Component], true, Propose,
                    "Power from each part's datasheet at the operating point"),
                decision("enclosure", "Which enclosure holds the board?", &[ProjectFile, EditorPick], &[Occurrence], true, Ask,
                    "An assembly in the project, or the board alone in free air"),
                decision("environment", "What environment does it see?", &[Quantity, SingleChoice], &[], true, Propose,
                    "Ambient temperature and airflow, or a vibration profile"),
            ],
            "runmat.check",
            vec![],
            "fea",
            "Set up a thermal study of this board",
        ).needing(&[".circuit.layout"], 1),
    ];
    let suggestions = suggestions::catalog();
    GuideCatalog {
        format: "runmat.design-guides".into(),
        format_version: 1,
        controls: ControlKind::ALL.to_vec(),
        guides,
        suggestions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guides_follow_the_table_and_declare_everything() {
        let c = catalog();
        let ids: Vec<&str> = c.guides.iter().map(|g| g.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "simulation_setup",
                "part_from_requirements",
                "manufacturable",
                "assembly_from_parts",
                "drawing_for_release",
                "schematic_capture",
                "board_bring_up",
                "engineering_change",
                "coupled_study"
            ]
        );
        for g in &c.guides {
            assert!(!g.decisions.is_empty(), "{}", g.id);
            assert!(!g.default_model.is_empty(), "{}", g.id);
            assert!(g.extension.starts_with('.'), "{}", g.id);
            assert_eq!(g.turn_usage == "fea", g.extension == ".fea", "{}", g.id);
        }
        // Every control kind of §2.3 is used by some guide.
        for k in ControlKind::ALL {
            assert!(
                c.guides
                    .iter()
                    .flat_map(|g| &g.decisions)
                    .any(|d| d.controls.contains(&k)),
                "{k:?} unused"
            );
        }
        let structural: Vec<String> = c
            .lint(&[])
            .into_iter()
            .filter(|p| !p.contains("isn't in the catalog"))
            .collect();
        assert!(structural.is_empty(), "{structural:#?}");
    }
}
