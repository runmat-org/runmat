//! Engineering change orders: schematic ↔ board synchronization as one typed change list.
//!
//! [`compare`] lines the elaborated schematic (IR, netlist and lock) up with the board and
//! returns an [`EcoPlan`]: typed [`Change`]s, each with a stable id, from → to, the cause (the
//! schematic sheet and source span that produced it) and its board impact (the copper touching
//! the affected nets and pads, and where new parts go). [`back_annotate`] does the reverse
//! (board edits the schematic should take: designator renumbering, pin and gate swaps,
//! footprint changes) as the same change list, and [`agent_change_set`] turns an agent's model
//! edits into a plan, so agent ECOs are reviewed the same way.
//!
//! The flow is an engineering change order: validate the listed changes, then execute them.
//!
//! - **Validate, then execute.** [`validate`] gives each selected change ok, warning or error with
//!   a reason (a footprint missing from the library, pads the new footprint lacks, a change that
//!   needs another one that isn't selected, a designator already taken). [`apply`] executes a
//!   selected subset *atomically* as one undoable edit: nothing changes if any selected change
//!   fails validation. The valid lines never run alone while the rest are skipped.
//! - **Identity, not names.** Footprints link to components by path through the lock section, so
//!   a renamed component is a designator change, never delete plus add; nets with the same pins
//!   under a new name are a rename, so their routing stays.
//! - **Three-way.** The board records what it last took from the schematic ([`BoardSync`]). A
//!   difference the board made (a swap, a renumber) is flagged instead of silently reverted, and
//!   a field both sides changed is a conflict to resolve, never an overwrite.
//! - **Placement and routing survive.** Only what a change touches moves; new parts land in their
//!   channel's room or a staging area beside the board, grouped by sheet. Copper a change orphans
//!   is kept with no net unless [`ApplyOptions::remove_orphaned_copper`] is set.
//! - **Checks follow.** [`apply_and_check`] refills pours and runs DRC (and ERC on the schematic).
//!
//! Everything is deterministic: ordered maps and a fixed sort of the change list.

use crate::elab::Netlist;
use crate::geom::poly::Polygon;
use crate::geom::Point;
use crate::ids::{LockSection, Uuid};
use crate::ir::Design;
use crate::library::Pool;
use std::collections::{BTreeMap, BTreeSet};

mod support;
mod compare;
mod back;
mod apply;
mod apply_back;
mod report;

pub use support::*;
pub use compare::*;
pub use back::*;
pub use apply::*;
pub use apply_back::*;
pub use report::*;

/* ------------------------------------------------------------------------------------------------
 * The board's record of the last sync (the ECO base)
 * --------------------------------------------------------------------------------------------- */

/// What the board last took from the schematic: the base of the three-way compare. Updated per
/// applied change, so a partial ECO leaves the rest pending. A `.circuit.layout` file keeps it
/// in its `sync { }` section ([`crate::syntax::layout`]), with the footprints' component links.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoardSync {
    /// The comparator settings the board was last updated with; [`compare_saved`] and
    /// [`back_annotate_saved`] compare with them, and [`apply`] saves the plan's.
    pub options: ComparatorOptions,
    /// By component path.
    pub components: BTreeMap<String, SyncedComponent>,
    /// Net class → rule parameter → display value.
    pub net_classes: BTreeMap<String, BTreeMap<String, String>>,
    /// Differential pair → (positive net, negative net).
    pub diff_pairs: BTreeMap<String, (String, String)>,
    /// Component class (one per channel, plus authored classes) → member designators.
    pub component_classes: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SyncedComponent {
    pub designator: String,
    pub package: String,
    /// Pad → net name, as last synced.
    pub pads: BTreeMap<String, String>,
}

/* ------------------------------------------------------------------------------------------------
 * The change model
 * --------------------------------------------------------------------------------------------- */

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Direction {
    /// Schematic → board (Update PCB).
    Forward,
    /// Board → schematic (back-annotation).
    Back,
}

/// Change types, in the order the review groups them and [`apply`] executes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChangeKind {
    RemoveComponent,
    AddComponent,
    ChangeDesignator,
    ChangeFootprint,
    ChangeValue,
    ChangeParameter,
    ChangeFitted,
    RenameNet,
    AddNet,
    ChangePinNet,
    SwapPins,
    SwapGates,
    RemoveNet,
    AddNetClass,
    RemoveNetClass,
    ChangeNetClass,
    AddDiffPair,
    RemoveDiffPair,
    AddComponentClass,
    RemoveComponentClass,
    ChangeComponentClass,
    AddRoom,
    ChangeRoom,
    RemoveRoom,
    ChangeRule,
}

impl ChangeKind {
    pub fn as_str(self) -> &'static str {
        use ChangeKind::*;
        match self {
            RemoveComponent => "remove_component",
            AddComponent => "add_component",
            ChangeDesignator => "change_designator",
            ChangeFootprint => "change_footprint",
            ChangeValue => "change_value",
            ChangeParameter => "change_parameter",
            ChangeFitted => "change_fitted",
            RenameNet => "rename_net",
            AddNet => "add_net",
            ChangePinNet => "change_pin_net",
            SwapPins => "swap_pins",
            SwapGates => "swap_gates",
            RemoveNet => "remove_net",
            AddNetClass => "add_net_class",
            RemoveNetClass => "remove_net_class",
            ChangeNetClass => "change_net_class",
            AddDiffPair => "add_diff_pair",
            RemoveDiffPair => "remove_diff_pair",
            AddComponentClass => "add_component_class",
            RemoveComponentClass => "remove_component_class",
            ChangeComponentClass => "change_component_class",
            AddRoom => "add_room",
            ChangeRoom => "change_room",
            RemoveRoom => "remove_room",
            ChangeRule => "change_rule",
        }
    }
    pub fn group(self) -> ChangeGroup {
        use ChangeKind::*;
        match self {
            RemoveComponent | AddComponent | ChangeDesignator | ChangeFitted => {
                ChangeGroup::Components
            }
            ChangeFootprint => ChangeGroup::Footprints,
            ChangeValue | ChangeParameter => ChangeGroup::Parameters,
            RenameNet | AddNet | RemoveNet => ChangeGroup::Nets,
            ChangePinNet | SwapPins | SwapGates => ChangeGroup::Connections,
            AddNetClass | RemoveNetClass | ChangeNetClass | AddDiffPair | RemoveDiffPair
            | AddComponentClass | RemoveComponentClass | ChangeComponentClass => {
                ChangeGroup::Classes
            }
            AddRoom | ChangeRoom | RemoveRoom => ChangeGroup::Rooms,
            ChangeRule => ChangeGroup::Rules,
        }
    }
    /// The difference types whose ignore setting hides this kind of change: any one of them
    /// does. Rules are net-class parameters, so ignoring net classes ignores rules too.
    pub fn ignored_by(self) -> &'static [Difference] {
        use ChangeKind::*;
        use Difference as D;
        match self {
            ChangeFootprint => &[D::Footprints],
            ChangeValue => &[D::Values],
            ChangeParameter => &[D::Parameters],
            ChangeFitted => &[D::Fitting],
            AddNetClass | RemoveNetClass | ChangeNetClass => &[D::NetClasses],
            ChangeRule => &[D::Rules, D::NetClasses],
            AddDiffPair | RemoveDiffPair => &[D::DiffPairs],
            AddComponentClass | RemoveComponentClass | ChangeComponentClass => {
                &[D::ComponentClasses]
            }
            AddRoom | ChangeRoom | RemoveRoom => &[D::Rooms],
            RemoveComponent | AddComponent | ChangeDesignator | RenameNet | AddNet
            | ChangePinNet | SwapPins | SwapGates | RemoveNet => &[],
        }
    }
    /// The verb the review shows (`Add`, `Rename`…).
    pub fn label(self) -> &'static str {
        use ChangeKind::*;
        match self {
            RemoveComponent => "Remove component",
            AddComponent => "Add component",
            ChangeDesignator => "Change designator",
            ChangeFootprint => "Change footprint",
            ChangeValue => "Change value",
            ChangeParameter => "Change parameter",
            ChangeFitted => "Change fitting",
            RenameNet => "Rename net",
            AddNet => "Add net",
            ChangePinNet => "Change pin net",
            SwapPins => "Swap pins",
            SwapGates => "Swap gates",
            RemoveNet => "Remove net",
            AddNetClass => "Add net class",
            RemoveNetClass => "Remove net class",
            ChangeNetClass => "Change net class",
            AddDiffPair => "Add differential pair",
            RemoveDiffPair => "Remove differential pair",
            AddComponentClass => "Add component class",
            RemoveComponentClass => "Remove component class",
            ChangeComponentClass => "Change component class",
            AddRoom => "Add room",
            ChangeRoom => "Change room",
            RemoveRoom => "Remove room",
            ChangeRule => "Change rule",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChangeGroup {
    Components,
    Footprints,
    Parameters,
    Nets,
    Connections,
    Classes,
    Rooms,
    Rules,
}

impl ChangeGroup {
    pub fn as_str(self) -> &'static str {
        match self {
            ChangeGroup::Components => "components",
            ChangeGroup::Footprints => "footprints",
            ChangeGroup::Parameters => "parameters",
            ChangeGroup::Nets => "nets",
            ChangeGroup::Connections => "connections",
            ChangeGroup::Classes => "classes",
            ChangeGroup::Rooms => "rooms",
            ChangeGroup::Rules => "rules",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            ChangeGroup::Components => "Components",
            ChangeGroup::Footprints => "Footprints",
            ChangeGroup::Parameters => "Values and parameters",
            ChangeGroup::Nets => "Nets",
            ChangeGroup::Connections => "Pin connections",
            ChangeGroup::Classes => "Classes and pairs",
            ChangeGroup::Rooms => "Rooms",
            ChangeGroup::Rules => "Rules",
        }
    }
}

/// Who proposed the change.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Origin {
    Schematic,
    Board,
    Agent { title: String },
}

/// Why a change exists: the schematic sheet and declaration that produced it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cause {
    /// Sheet instance, for display and jumping (`ch[2]`, `main`).
    pub sheet: String,
    /// The sheet instance path (`/ch[2]`), for the Outline's hierarchy node.
    pub sheet_path: String,
    /// The declaration's source span (`io.circuit:120..168`), when known.
    pub source: Option<String>,
    /// One sentence: what differs.
    pub reason: String,
}

/// Where a new part goes.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    pub at: Point,
    /// The room it was placed in, when its channel has one.
    pub room: Option<String>,
    /// In the staging area beside the board (no room yet).
    pub staged: bool,
}

/// What a change touches on the board.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Impact {
    pub tracks: Vec<u64>,
    pub vias: Vec<u64>,
    pub zones: Vec<u64>,
    /// `R1.2`.
    pub pads: Vec<String>,
    pub footprints: Vec<String>,
    pub nets: Vec<String>,
    pub placement: Option<Placement>,
}

impl Impact {
    /// Canvas region ids (`component:R1`, `pad:R1.2`, `net:GND`, `track:12`…), as the Outline uses.
    pub fn regions(&self) -> Vec<String> {
        let mut v: Vec<String> = Vec::new();
        v.extend(self.footprints.iter().map(|f| format!("component:{f}")));
        v.extend(self.pads.iter().map(|p| format!("pad:{p}")));
        v.extend(self.nets.iter().map(|n| format!("net:{n}")));
        v.extend(self.tracks.iter().map(|t| format!("track:{t}")));
        v.extend(self.vias.iter().map(|t| format!("via:{t}")));
        v.extend(self.zones.iter().map(|t| format!("zone:{t}")));
        if let Some(room) = self.placement.as_ref().and_then(|p| p.room.as_ref()) {
            v.push(format!("room:{room}"));
        }
        v
    }
    fn merge(&mut self, o: Impact) {
        self.tracks.extend(o.tracks);
        self.vias.extend(o.vias);
        self.zones.extend(o.zones);
        self.pads.extend(o.pads);
        self.nets.extend(o.nets);
        self.footprints.extend(o.footprints);
        self.tidy();
    }
    fn tidy(&mut self) {
        for v in [&mut self.tracks, &mut self.vias, &mut self.zones] {
            v.sort();
            v.dedup();
        }
        for v in [&mut self.pads, &mut self.nets, &mut self.footprints] {
            v.sort();
            v.dedup();
        }
    }
    fn copper(&self) -> usize {
        self.tracks.len() + self.vias.len()
    }
}

/// The typed payload [`apply`] executes.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    AddComponent {
        path: String,
        designator: String,
        package: Option<Uuid>,
        value: String,
        params: BTreeMap<String, String>,
        /// Pad → net name.
        pads: BTreeMap<String, String>,
        fitted: bool,
        at: Point,
    },
    RemoveComponent {
        reference: String,
    },
    SetDesignator {
        reference: String,
        to: String,
        /// The component's current path: the footprint's link follows a rename.
        path: String,
    },
    SetFootprint {
        reference: String,
        package: Option<Uuid>,
    },
    SetValue {
        reference: String,
        value: String,
    },
    SetParam {
        reference: String,
        key: String,
        value: Option<String>,
    },
    SetFitted {
        reference: String,
        fitted: bool,
    },
    AddNet {
        name: String,
        class: Option<String>,
    },
    RemoveNet {
        name: String,
    },
    RenameNet {
        from: String,
        to: String,
    },
    SetPadNet {
        reference: String,
        pad: String,
        net: Option<String>,
    },
    SetNetClassDef {
        name: String,
        params: Option<BTreeMap<String, String>>,
    },
    SetNetClass {
        net: String,
        class: Option<String>,
    },
    SetDiffPair {
        name: String,
        pair: Option<(String, String)>,
    },
    SetComponentClass {
        name: String,
        members: Option<Vec<String>>,
    },
    SetRoom {
        name: String,
        outline: Option<Polygon>,
        members: Vec<String>,
    },
    SetRule {
        class: String,
        key: String,
        value: Option<String>,
    },
    /* Back-annotation: edits to the schematic. */
    SchDesignator {
        path: String,
        sheet: String,
        to: String,
    },
    SchSwapPins {
        path: String,
        sheet: String,
        component: String,
        a: String,
        b: String,
    },
    SchSwapGates {
        path: String,
        sheet: String,
        component: String,
        a: String,
        b: String,
    },
    SchFootprint {
        path: String,
        sheet: String,
        component: String,
        /// The replacement part (same entity, the board's package), or the package for an
        /// entity reference.
        part: Option<String>,
        package: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    Ok,
    Warning,
    Error,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Warning => "warning",
            Status::Error => "error",
        }
    }
}

/// One line of the ECO.
#[derive(Clone, Debug, PartialEq)]
pub struct Change {
    /// Stable across recompiles: kind and identity (`add_component:/ch[2]/U`).
    pub id: String,
    pub kind: ChangeKind,
    pub direction: Direction,
    pub origin: Origin,
    /// What the review shows: a designator, `designator.pad`, a net or class name.
    pub item: String,
    pub from: Option<String>,
    pub to: Option<String>,
    pub cause: Cause,
    pub impact: Impact,
    /// Ids of changes this one needs (the net a pin moves to, the part a room holds).
    pub requires: Vec<String>,
    /// Both sides changed the same field since the last sync.
    pub conflict: Option<String>,
    /// Problems known without a selection (missing footprint, pad mismatch, board edit).
    pub issues: Vec<(Status, String)>,
    pub op: Op,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EcoPlan {
    pub direction: Direction,
    pub changes: Vec<Change>,
    /// The comparator settings the plan was made with; [`apply`] saves them on the board.
    pub options: ComparatorOptions,
}

impl EcoPlan {
    pub fn get(&self, id: &str) -> Option<&Change> {
        self.changes.iter().find(|c| c.id == id)
    }
    pub fn ids(&self) -> BTreeSet<String> {
        self.changes.iter().map(|c| c.id.clone()).collect()
    }
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
    /// The selection closed over requirements: what a reviewer gets by checking these rows.
    pub fn with_requirements(&self, selected: &BTreeSet<String>) -> BTreeSet<String> {
        let mut out = selected.clone();
        let mut stack: Vec<String> = selected.iter().cloned().collect();
        while let Some(id) = stack.pop() {
            if let Some(c) = self.get(&id) {
                for r in &c.requires {
                    if out.insert(r.clone()) {
                        stack.push(r.clone());
                    }
                }
            }
        }
        out
    }
}

/// A type of difference the comparator can ignore (find or ignore differences, per type). An ignored type produces no ECO lines in
/// either direction, and [`apply`] leaves what it covers on the board as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Difference {
    Parameters,
    Values,
    Footprints,
    Fitting,
    NetClasses,
    Rules,
    DiffPairs,
    ComponentClasses,
    Rooms,
}

impl Difference {
    pub const ALL: [Difference; 9] = [
        Difference::Parameters,
        Difference::Values,
        Difference::Footprints,
        Difference::Fitting,
        Difference::NetClasses,
        Difference::Rules,
        Difference::DiffPairs,
        Difference::ComponentClasses,
        Difference::Rooms,
    ];

    /// The written word (`.circuit.layout` sync section, projection, command inputs).
    pub fn as_str(self) -> &'static str {
        match self {
            Difference::Parameters => "parameters",
            Difference::Values => "values",
            Difference::Footprints => "footprints",
            Difference::Fitting => "fitting",
            Difference::NetClasses => "net_classes",
            Difference::Rules => "rules",
            Difference::DiffPairs => "diff_pairs",
            Difference::ComponentClasses => "component_classes",
            Difference::Rooms => "rooms",
        }
    }

    pub fn parse(word: &str) -> Option<Difference> {
        Difference::ALL.into_iter().find(|d| d.as_str() == word)
    }

    /// The command input's label in Properties.
    pub fn label(self) -> &'static str {
        match self {
            Difference::Parameters => "Ignore parameters",
            Difference::Values => "Ignore values",
            Difference::Footprints => "Ignore footprints",
            Difference::Fitting => "Ignore fitting",
            Difference::NetClasses => "Ignore net classes",
            Difference::Rules => "Ignore rules",
            Difference::DiffPairs => "Ignore differential pairs",
            Difference::ComponentClasses => "Ignore component classes",
            Difference::Rooms => "Ignore rooms",
        }
    }

    /// What ignoring it leaves alone, for the input's help text.
    pub fn help(self) -> &'static str {
        match self {
            Difference::Parameters => "Component parameters are not compared.",
            Difference::Values => "Component values are not compared.",
            Difference::Footprints => "Footprint changes are not sent; the board keeps its footprints.",
            Difference::Fitting => "Fitted and not-fitted changes from variants are not sent.",
            Difference::NetClasses => "Net classes, net class membership and their rules are not compared.",
            Difference::Rules => "Net class rule parameters are not compared.",
            Difference::DiffPairs => "Differential pairs are not compared.",
            Difference::ComponentClasses => "Component classes, including one per channel, are not compared.",
            Difference::Rooms => "Rooms are not added, changed or removed, and their outlines and members stay as they are.",
        }
    }

    /// The command input id (`ignore_rooms`).
    pub fn input_id(self) -> String {
        format!("ignore_{}", self.as_str())
    }
}

/// What the compare ignores and how it maps the schematic (ignored difference types, and the
/// classes and rooms it generates).
#[derive(Clone, Debug, PartialEq)]
pub struct ComparatorOptions {
    pub ignore_parameters: bool,
    /// Parameters never compared (`datasheet`, `stock`).
    pub ignored_parameters: BTreeSet<String>,
    pub ignore_values: bool,
    pub ignore_footprints: bool,
    pub ignore_fitting: bool,
    pub ignore_rooms: bool,
    pub ignore_component_classes: bool,
    pub ignore_net_classes: bool,
    pub ignore_diff_pairs: bool,
    pub ignore_rules: bool,
    /// Remove footprints no component links to (KiCad's "Delete footprints with no symbols").
    /// Footprints marked `board_only` (fiducials, mounting holes) are always kept.
    pub remove_unlinked_footprints: bool,
    /// Link footprints with no component link by designator (KiCad's re-link option).
    pub relink_by_designator: bool,
    /// A room per channel of a repeated sheet (room generation for multi-channel designs).
    pub rooms_per_channel: bool,
    /// A component class per channel.
    pub classes_per_channel: bool,
}

impl Default for ComparatorOptions {
    fn default() -> Self {
        ComparatorOptions {
            ignore_parameters: false,
            ignored_parameters: BTreeSet::new(),
            ignore_values: false,
            ignore_footprints: false,
            ignore_fitting: false,
            ignore_rooms: false,
            ignore_component_classes: false,
            ignore_net_classes: false,
            ignore_diff_pairs: false,
            ignore_rules: false,
            remove_unlinked_footprints: true,
            relink_by_designator: true,
            rooms_per_channel: true,
            classes_per_channel: true,
        }
    }
}

impl ComparatorOptions {
    /// The setting for one difference type, as stored.
    pub fn ignore(&self, d: Difference) -> bool {
        match d {
            Difference::Parameters => self.ignore_parameters,
            Difference::Values => self.ignore_values,
            Difference::Footprints => self.ignore_footprints,
            Difference::Fitting => self.ignore_fitting,
            Difference::NetClasses => self.ignore_net_classes,
            Difference::Rules => self.ignore_rules,
            Difference::DiffPairs => self.ignore_diff_pairs,
            Difference::ComponentClasses => self.ignore_component_classes,
            Difference::Rooms => self.ignore_rooms,
        }
    }

    pub fn set_ignore(&mut self, d: Difference, on: bool) {
        let field = match d {
            Difference::Parameters => &mut self.ignore_parameters,
            Difference::Values => &mut self.ignore_values,
            Difference::Footprints => &mut self.ignore_footprints,
            Difference::Fitting => &mut self.ignore_fitting,
            Difference::NetClasses => &mut self.ignore_net_classes,
            Difference::Rules => &mut self.ignore_rules,
            Difference::DiffPairs => &mut self.ignore_diff_pairs,
            Difference::ComponentClasses => &mut self.ignore_component_classes,
            Difference::Rooms => &mut self.ignore_rooms,
        };
        *field = on;
    }

    /// The ignored difference types, in [`Difference::ALL`] order.
    pub fn ignored(&self) -> Vec<Difference> {
        Difference::ALL
            .into_iter()
            .filter(|d| self.ignore(*d))
            .collect()
    }

    /// The same settings finding every difference type. The review compares this way and
    /// hides the rows the chosen settings ignore, so changing a setting in Properties needs no
    /// new compare; the hidden rows are exactly the ones a compare with the setting leaves out.
    pub fn finding_all(&self) -> ComparatorOptions {
        let mut out = self.clone();
        for d in Difference::ALL {
            out.set_ignore(d, false);
        }
        out
    }

    /// Whether these settings hide changes of `kind`.
    pub fn hides(&self, kind: ChangeKind) -> bool {
        kind.ignored_by().iter().any(|d| self.ignore(*d))
    }

    /// The settings with the difference types the command inputs set (`ignore_rooms = true`);
    /// inputs that are not given keep the stored setting.
    pub fn with_inputs(&self, inputs: &BTreeMap<String, bool>) -> ComparatorOptions {
        let mut out = self.clone();
        for d in Difference::ALL {
            if let Some(on) = inputs.get(&d.input_id()) {
                out.set_ignore(d, *on);
            }
        }
        out
    }
}

/// The schematic side of a compare.
pub struct Schematic<'a> {
    pub design: &'a Design,
    pub netlist: &'a Netlist,
    pub pool: &'a Pool,
    /// Reconciled with `netlist`, so its aliases carry renames.
    pub lock: &'a LockSection,
}
