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
//! The flow follows Altium's Engineering Change Order dialog, with what other tools do better:
//!
//! - **Validate, then execute.** [`validate`] gives each selected change ok, warning or error with
//!   a reason (a footprint missing from the library, pads the new footprint lacks, a change that
//!   needs another one that isn't selected, a designator already taken). [`apply`] executes a
//!   selected subset *atomically* as one undoable edit: nothing changes if any selected change
//!   fails validation (Altium executes the valid lines and skips the rest).
//! - **Identity, not names.** Footprints link to components by path through the lock section, so
//!   a renamed component is a designator change, never delete plus add; nets with the same pins
//!   under a new name are a rename, so their routing stays.
//! - **Three-way.** The board records what it last took from the schematic ([`BoardSync`]). A
//!   difference the board made (a swap, a renumber) is flagged instead of silently reverted, and
//!   a field both sides changed is a conflict to resolve (Xpedition and Allegro keep this state;
//!   Altium and KiCad overwrite).
//! - **Placement and routing survive.** Only what a change touches moves; new parts land in their
//!   channel's room or a staging area beside the board, grouped by sheet. Copper a change orphans
//!   is kept with no net unless [`ApplyOptions::remove_orphaned_copper`] is set.
//! - **Checks follow.** [`apply_and_check`] refills pours and runs DRC (and ERC on the schematic).
//!
//! Everything is deterministic: ordered maps and a fixed sort of the change list.

use crate::board_rules::BItem;
use crate::board_rules::BoardRules;
use crate::connectivity::{islands, net_nodes};
use crate::drc::Drc;
use crate::elab::{elaborate, CompInst, Diagnostic, ElabOptions, Netlist, Severity};
use crate::geom::poly::{rect_ring, Polygon};
use crate::geom::shape::Prim;
use crate::geom::{mm, Coord, Point, Rect};
use crate::ids::{DerivedUuids, LockSection, NamePath, Uuid};
use crate::index::{component_value, param_display};
use crate::ir::{Design, Endpoint, PartRef};
use crate::layout::{Board, ItemId, NetId, Room, Side};
use crate::library::Pool;
use crate::review::ModelEdit;
use std::collections::{BTreeMap, BTreeSet};

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

/// A type of difference the comparator can ignore (Altium's Comparator tab: *Find
/// Differences* or *Ignore Differences* per type). An ignored type produces no ECO lines in
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

/// What the compare ignores and how it maps the schematic (Altium's Comparator and Class
/// Generation tabs; KiCad's Update PCB options).
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
    /// A room per channel of a repeated sheet (Altium's room generation for multi-channel).
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

/* ------------------------------------------------------------------------------------------------
 * Matching
 * --------------------------------------------------------------------------------------------- */

fn follow(lock: &LockSection, p: &NamePath) -> NamePath {
    let mut cur = p.clone();
    let mut seen = BTreeSet::new();
    while let Some(n) = lock.aliases.get(&cur) {
        if !seen.insert(cur.clone()) {
            break;
        }
        cur = n.clone();
    }
    cur
}

/// Component index → footprint id, and the footprints nothing links to.
struct Links {
    by_comp: BTreeMap<usize, ItemId>,
    unlinked: Vec<ItemId>,
}

fn link(s: &Schematic, board: &Board, o: &ComparatorOptions) -> Links {
    let by_path: BTreeMap<NamePath, usize> = s
        .netlist
        .components
        .iter()
        .enumerate()
        .map(|(i, c)| (c.path.clone(), i))
        .collect();
    let mut by_comp = BTreeMap::new();
    let mut pending = Vec::new();
    for f in board.footprints.values() {
        let hit = f
            .component
            .as_deref()
            .map(|p| follow(s.lock, &NamePath::parse(p)))
            .and_then(|p| by_path.get(&p).copied());
        match hit {
            Some(ci) if !by_comp.contains_key(&ci) => {
                by_comp.insert(ci, f.id);
            }
            _ => pending.push(f.id),
        }
    }
    let mut unlinked = Vec::new();
    for id in pending {
        let f = &board.footprints[&id];
        let by_ref = (o.relink_by_designator && f.component.is_none())
            .then(|| s.netlist.component(&f.reference).map(|(i, _)| i))
            .flatten()
            .filter(|ci| !by_comp.contains_key(ci));
        match by_ref {
            Some(ci) => {
                by_comp.insert(ci, id);
            }
            None => unlinked.push(id),
        }
    }
    Links { by_comp, unlinked }
}

fn sheet_of(path: &NamePath, root: &str) -> (String, String) {
    let parent = path.parent().unwrap_or_default();
    if parent.depth() == 0 {
        (root.to_owned(), "/".to_owned())
    } else {
        (parent.0.join("/"), parent.to_string())
    }
}

fn comp_cause(c: &CompInst, root: &str, reason: String) -> Cause {
    let (sheet, sheet_path) = sheet_of(&c.path, root);
    Cause {
        sheet,
        sheet_path,
        source: (!c.source.file.is_empty()).then(|| c.source.to_string()),
        reason,
    }
}

fn root_cause(root: &str, source: Option<String>, reason: String) -> Cause {
    Cause {
        sheet: root.to_owned(),
        sheet_path: "/".to_owned(),
        source,
        reason,
    }
}

/// A net as the review shows it: a sheet-local net (`/ch[1]/TIE_B`) by its name and sheet
/// (`TIE_B (ch[1])`), never as a raw path.
pub fn net_label(name: &str) -> String {
    match name
        .strip_prefix('/')
        .and_then(|rest| rest.rsplit_once('/'))
    {
        Some((sheet, leaf)) => format!("{leaf} ({sheet})"),
        None => name.to_owned(),
    }
}

fn package_name(pool: &Pool, u: Option<Uuid>) -> Option<String> {
    u.and_then(|u| pool.packages.get(&u))
        .map(|p| p.name.clone())
}

/// The room a channel's components belong to (`ch[2]`): the sheet instance with an index.
fn channel_of(c: &CompInst) -> Option<String> {
    c.channel?;
    let parent = c.path.parent()?;
    // The shallowest segment carrying the channel index.
    let at = parent.0.iter().rposition(|seg| seg.ends_with(']'))?;
    Some(parent.0[..=at].join("/"))
}

/* ------------------------------------------------------------------------------------------------
 * Board impact
 * --------------------------------------------------------------------------------------------- */

fn copper_on_net(board: &Board, net: NetId) -> Impact {
    Impact {
        tracks: board
            .tracks
            .values()
            .filter(|t| t.net == Some(net))
            .map(|t| t.id.0)
            .collect(),
        vias: board
            .vias
            .values()
            .filter(|v| v.net == Some(net))
            .map(|v| v.id.0)
            .collect(),
        zones: board
            .zones
            .values()
            .filter(|z| z.net == Some(net))
            .map(|z| z.id.0)
            .collect(),
        nets: vec![board.net_name(net).to_owned()],
        ..Impact::default()
    }
}

/// Tracks and vias touching a pad on its net (the copper a pin change disconnects).
fn copper_at_pad(board: &Board, fid: ItemId, pad: &str) -> Impact {
    let mut i = Impact::default();
    let f = &board.footprints[&fid];
    i.pads.push(format!("{}.{}", f.reference, pad));
    let n = board.copper_count();
    let Some(p) = f.pads(n).into_iter().find(|p| p.name == pad) else {
        return i;
    };
    let Some(net) = p.net else { return i };
    let near = p.prim.bbox().inflate(mm(0.05));
    for t in board.tracks.values().filter(|t| t.net == Some(net)) {
        if !p.layers.contains(&t.layer) {
            continue;
        }
        let touches = t
            .prims()
            .iter()
            .any(|q| q.bbox().intersects(&near) && q.gap(&p.prim) <= 0.0);
        if touches {
            i.tracks.push(t.id.0);
        }
    }
    for v in board.vias.values().filter(|v| v.net == Some(net)) {
        let q = Prim::circle(v.at, v.diameter / 2);
        if q.bbox().intersects(&near) && q.gap(&p.prim) <= 0.0 {
            i.vias.push(v.id.0);
        }
    }
    // The pad and the copper on it, not its whole net (that would light up all of GND).
    i
}

fn footprint_impact(board: &Board, fid: ItemId) -> Impact {
    let f = &board.footprints[&fid];
    let mut i = Impact {
        footprints: vec![f.reference.clone()],
        ..Impact::default()
    };
    for pad in f
        .package
        .pads
        .iter()
        .map(|p| p.name.clone())
        .collect::<BTreeSet<_>>()
    {
        i.merge(copper_at_pad(board, fid, &pad));
    }
    i.pads.clear();
    i.tidy();
    i
}

/* ------------------------------------------------------------------------------------------------
 * Placement of new parts
 * --------------------------------------------------------------------------------------------- */

fn package_extent(pool: &Pool, u: Option<Uuid>) -> (Coord, Coord) {
    let Some(p) = u.and_then(|u| pool.packages.get(&u)) else {
        return (mm(2.0), mm(2.0));
    };
    let r = Rect::from_points(
        p.courtyard
            .iter()
            .flatten()
            .copied()
            .chain(p.pads.iter().map(|d| d.at)),
    );
    if r.is_empty() {
        (mm(2.0), mm(2.0))
    } else {
        (r.width().max(mm(1.0)), r.height().max(mm(1.0)))
    }
}

/// Lays new parts out: inside their channel's room when the board has it, else in a staging
/// area to the right of the board, one block per room or sheet.
struct Placer {
    staging: Point,
    next_y: Coord,
}

impl Placer {
    fn new(board: &Board) -> Self {
        let b = board.outline.bbox();
        let (x, y) = if b.is_empty() {
            (0, 0)
        } else {
            (b.max.x + mm(10.0), b.min.y)
        };
        Placer {
            staging: Point::new(x, y),
            next_y: y,
        }
    }
    /// Grid positions for `sizes` inside `area` (row-major, 1 mm gaps), avoiding `occupied`.
    fn grid(area: Rect, sizes: &[(Coord, Coord)], occupied: &[Rect]) -> Vec<Point> {
        let gap = mm(1.0);
        let cell_w = sizes.iter().map(|s| s.0).max().unwrap_or(mm(2.0)) + gap;
        let cell_h = sizes.iter().map(|s| s.1).max().unwrap_or(mm(2.0)) + gap;
        let cols = ((area.width() - gap) / cell_w).max(1);
        let mut out = Vec::new();
        let mut k: Coord = 0;
        while out.len() < sizes.len() && k < 100_000 {
            let (cx, cy) = (k % cols, k / cols);
            k += 1;
            let at = Point::new(
                area.min.x + gap + cx * cell_w + cell_w / 2,
                area.min.y + gap + cy * cell_h + cell_h / 2,
            );
            let cell = Rect::new(
                Point::new(at.x - cell_w / 2, at.y - cell_h / 2),
                Point::new(at.x + cell_w / 2, at.y + cell_h / 2),
            );
            if occupied.iter().any(|r| r.intersects(&cell)) {
                continue;
            }
            out.push(at);
        }
        out
    }
    /// A staging block for `sizes`; returns the positions and the block's outline.
    fn stage(&mut self, sizes: &[(Coord, Coord)]) -> (Vec<Point>, Rect) {
        let gap = mm(1.0);
        let cell_w = sizes.iter().map(|s| s.0).max().unwrap_or(mm(2.0)) + gap;
        let cell_h = sizes.iter().map(|s| s.1).max().unwrap_or(mm(2.0)) + gap;
        let cols = (sizes.len() as f64).sqrt().ceil().max(1.0) as Coord;
        let rows = (sizes.len() as Coord + cols - 1) / cols;
        let area = Rect::new(
            Point::new(self.staging.x, self.next_y),
            Point::new(
                self.staging.x + cols * cell_w + gap,
                self.next_y + rows.max(1) * cell_h + gap,
            ),
        );
        self.next_y = area.max.y + mm(4.0);
        (Self::grid(area, sizes, &[]), area)
    }
}

/* ------------------------------------------------------------------------------------------------
 * Forward compare
 * --------------------------------------------------------------------------------------------- */

struct Builder {
    changes: Vec<Change>,
    direction: Direction,
}

impl Builder {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        kind: ChangeKind,
        key: &str,
        item: impl Into<String>,
        from: Option<String>,
        to: Option<String>,
        cause: Cause,
        impact: Impact,
        op: Op,
    ) -> &mut Change {
        let mut impact = impact;
        impact.tidy();
        self.changes.push(Change {
            id: format!("{}:{key}", kind.as_str()),
            kind,
            direction: self.direction,
            origin: match self.direction {
                Direction::Forward => Origin::Schematic,
                Direction::Back => Origin::Board,
            },
            item: item.into(),
            from,
            to,
            cause,
            impact,
            requires: Vec::new(),
            conflict: None,
            issues: Vec::new(),
            op,
        });
        self.changes.last_mut().unwrap()
    }
}

/// Sort key that orders `R2` before `R10`.
fn natural(s: &str) -> (String, u64, String) {
    let digits_at = s.find(|c: char| c.is_ascii_digit()).unwrap_or(s.len());
    let (head, rest) = s.split_at(digits_at);
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    (
        head.to_owned(),
        rest[..end].parse().unwrap_or(0),
        rest[end..].to_owned(),
    )
}

fn finish(direction: Direction, mut changes: Vec<Change>, o: &ComparatorOptions) -> EcoPlan {
    // An ignored difference type produces no lines, whichever part of the compare found it.
    changes.retain(|c| !o.hides(c.kind));
    changes.sort_by(|a, b| {
        (a.kind.group(), a.kind, natural(&a.item), &a.id).cmp(&(
            b.kind.group(),
            b.kind,
            natural(&b.item),
            &b.id,
        ))
    });
    let ids: BTreeSet<String> = changes.iter().map(|c| c.id.clone()).collect();
    for c in &mut changes {
        c.requires.retain(|r| ids.contains(r) && *r != c.id);
        c.requires.sort();
        c.requires.dedup();
    }
    EcoPlan {
        direction,
        changes,
        options: o.clone(),
    }
}

/// The schematic's net names per component pad (`pad → net`).
fn sch_pads(nl: &Netlist, c: &CompInst) -> BTreeMap<String, Option<String>> {
    let mut out = BTreeMap::new();
    for p in &c.pins {
        for pad in &p.pads {
            out.insert(pad.clone(), p.net.map(|n| nl.nets[n].name.clone()));
        }
    }
    out
}

/// [`compare`] with the comparator settings saved on the board (its `sync { }` section).
pub fn compare_saved(s: &Schematic, board: &Board) -> EcoPlan {
    compare(s, board, &board.sync.options)
}

/// [`back_annotate`] with the comparator settings saved on the board.
pub fn back_annotate_saved(s: &Schematic, board: &Board) -> EcoPlan {
    back_annotate(s, board, &board.sync.options)
}

/// Schematic → board: everything the board would change to match the schematic.
pub fn compare(s: &Schematic, board: &Board, o: &ComparatorOptions) -> EcoPlan {
    let nl = s.netlist;
    let root = s.design.root.clone();
    let links = link(s, board, o);
    let mut b = Builder {
        changes: Vec::new(),
        direction: Direction::Forward,
    };
    let base = &board.sync;

    // ---- Nets: renames first (pins kept under a new name keep their routing).
    let sch_names: BTreeSet<&str> = nl.nets.iter().map(|n| n.name.as_str()).collect();
    let board_names: BTreeSet<&str> = board.nets.iter().map(|n| n.name.as_str()).collect();
    // Pin identity: (component path, pad).
    let comp_of_fp: BTreeMap<ItemId, usize> = links.by_comp.iter().map(|(c, f)| (*f, *c)).collect();
    let mut board_pins: BTreeMap<&str, BTreeSet<(String, String)>> = BTreeMap::new();
    for f in board.footprints.values() {
        let Some(ci) = comp_of_fp.get(&f.id) else {
            continue;
        };
        for (pad, net) in &f.pad_nets {
            board_pins
                .entry(board.net_name(*net))
                .or_default()
                .insert((nl.components[*ci].path.to_string(), pad.clone()));
        }
    }
    let mut sch_pins: BTreeMap<&str, BTreeSet<(String, String)>> = BTreeMap::new();
    for n in &nl.nets {
        let set = sch_pins.entry(n.name.as_str()).or_default();
        for (ci, pi) in &n.pins {
            let c = &nl.components[*ci];
            for pad in &c.pins[*pi].pads {
                set.insert((c.path.to_string(), pad.clone()));
            }
        }
    }
    let new_nets: Vec<&str> = sch_names.difference(&board_names).copied().collect();
    let gone_nets: Vec<&str> = board_names.difference(&sch_names).copied().collect();
    // board name → schematic name
    let mut renamed: BTreeMap<String, String> = BTreeMap::new();
    let mut taken: BTreeSet<&str> = BTreeSet::new();
    for old in &gone_nets {
        let Some(bp) = board_pins.get(old).filter(|p| !p.is_empty()) else {
            continue;
        };
        let mut best: Option<(&str, f64)> = None;
        for new in &new_nets {
            if taken.contains(new) {
                continue;
            }
            let explicit = nl
                .net(new)
                .and_then(|(_, n)| n.renamed_from.as_ref())
                .is_some_and(|r| r.leaf() == NamePath::parse(old).leaf());
            let sp = sch_pins.get(new).cloned().unwrap_or_default();
            let inter = bp.intersection(&sp).count() as f64;
            let uni = bp.union(&sp).count() as f64;
            let score = if explicit {
                2.0
            } else if uni > 0.0 {
                inter / uni
            } else {
                0.0
            };
            if score >= 0.5 && best.is_none_or(|(_, s)| score > s) {
                best = Some((new, score));
            }
        }
        if let Some((new, _)) = best {
            taken.insert(new);
            renamed.insert((*old).to_owned(), new.to_owned());
        }
    }
    for (old, new) in &renamed {
        let nid = board.net(old).unwrap();
        let (_, n) = nl.net(new).unwrap();
        let (sheet, sheet_path) = sheet_of(&n.path, &root);
        let mut impact = copper_on_net(board, nid);
        impact.nets.push(new.clone());
        b.push(
            ChangeKind::RenameNet,
            old,
            net_label(new),
            Some(net_label(old)),
            Some(net_label(new)),
            Cause {
                sheet,
                sheet_path,
                source: None,
                reason: if n.renamed_from.is_some() {
                    format!("net renamed from {old} in the schematic")
                } else {
                    format!("the pins of {old} are now named {new}")
                },
            },
            impact,
            Op::RenameNet {
                from: old.clone(),
                to: new.clone(),
            },
        );
    }
    // A board net name as the schematic now calls it.
    let board_as_sch = |name: &str| -> String {
        renamed
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_owned())
    };
    let renamed_to: BTreeSet<&str> = renamed.values().map(|s| s.as_str()).collect();
    for new in &new_nets {
        if renamed_to.contains(new) {
            continue;
        }
        let (_, n) = nl.net(new).unwrap();
        if n.pins.is_empty() {
            continue;
        }
        let (sheet, sheet_path) = sheet_of(&n.path, &root);
        b.push(
            ChangeKind::AddNet,
            new,
            net_label(new),
            None,
            n.class.as_ref().map(|c| format!("class {c}")),
            Cause {
                sheet,
                sheet_path,
                source: None,
                reason: format!(
                    "{} has {} pins in the schematic",
                    net_label(new),
                    n.pins.len()
                ),
            },
            Impact {
                nets: vec![(*new).to_owned()],
                ..Impact::default()
            },
            Op::AddNet {
                name: (*new).to_owned(),
                class: n.class.clone(),
            },
        );
    }
    let add_net_req = |name: &str| -> Option<String> {
        new_nets
            .contains(&name)
            .then(|| {
                if renamed_to.contains(name) {
                    renamed
                        .iter()
                        .find(|(_, v)| v.as_str() == name)
                        .map(|(k, _)| format!("rename_net:{k}"))
                } else {
                    Some(format!("add_net:{name}"))
                }
            })
            .flatten()
    };
    for old in &gone_nets {
        if renamed.contains_key(*old) {
            continue;
        }
        let nid = board.net(old).unwrap();
        let impact = copper_on_net(board, nid);
        let ch = b.push(
            ChangeKind::RemoveNet,
            old,
            net_label(old),
            Some(net_label(old)),
            None,
            root_cause(
                &root,
                None,
                format!("{} is no longer in the schematic", net_label(old)),
            ),
            impact,
            Op::RemoveNet {
                name: (*old).to_owned(),
            },
        );
        if ch.impact.copper() > 0 {
            let n = ch.impact.copper();
            ch.issues.push((
                Status::Warning,
                format!("{n} routed items on {} lose their net", net_label(old)),
            ));
        }
    }

    // ---- Components.
    let mut placer = Placer::new(board);
    let mut adds: BTreeMap<Option<String>, Vec<usize>> = BTreeMap::new();
    for (ci, c) in nl.components.iter().enumerate() {
        if !links.by_comp.contains_key(&ci) {
            adds.entry(if o.rooms_per_channel {
                channel_of(c)
            } else {
                None
            })
            .or_default()
            .push(ci);
        }
    }
    let room_rect = |name: &str| {
        board
            .rooms
            .iter()
            .find(|r| r.name == name)
            .map(|r| r.outline.bbox())
    };
    let mut new_rooms: BTreeMap<String, Rect> = BTreeMap::new();
    for (room, cis) in &adds {
        let sizes: Vec<(Coord, Coord)> = cis
            .iter()
            .map(|ci| package_extent(s.pool, nl.components[*ci].package))
            .collect();
        let (spots, placed_room, staged) = match room.as_deref().and_then(room_rect) {
            Some(area) => {
                let occupied: Vec<Rect> = board
                    .footprints
                    .values()
                    .filter(|f| area.intersects(&f.bbox()))
                    .map(|f| f.bbox())
                    .collect();
                (Placer::grid(area, &sizes, &occupied), room.clone(), false)
            }
            None => {
                let (spots, area) = placer.stage(&sizes);
                if let Some(r) = room {
                    new_rooms.insert(r.clone(), area);
                }
                (spots, room.clone(), true)
            }
        };
        for (k, ci) in cis.iter().enumerate() {
            let c = &nl.components[*ci];
            let at = spots.get(k).copied().unwrap_or(placer.staging);
            let pads: BTreeMap<String, String> = sch_pads(nl, c)
                .into_iter()
                .filter_map(|(p, n)| n.map(|n| (p, n)))
                .collect();
            let pkg = package_name(s.pool, c.package);
            let mut requires: Vec<String> = pads.values().filter_map(|n| add_net_req(n)).collect();
            requires.sort();
            requires.dedup();
            let ch = b.push(
                ChangeKind::AddComponent,
                &c.path.to_string(),
                c.designator.clone(),
                None,
                Some(match &pkg {
                    Some(p) => p.clone(),
                    None => "no footprint".to_owned(),
                }),
                comp_cause(
                    c,
                    &root,
                    format!("{} is in the schematic, not on the board", c.designator),
                ),
                Impact {
                    footprints: vec![c.designator.clone()],
                    nets: pads.values().cloned().collect(),
                    placement: Some(Placement {
                        at,
                        room: placed_room.clone(),
                        staged,
                    }),
                    ..Impact::default()
                },
                Op::AddComponent {
                    path: c.path.to_string(),
                    designator: c.designator.clone(),
                    package: c.package,
                    value: component_value(c).unwrap_or_default(),
                    params: c
                        .params
                        .iter()
                        .map(|(k, v)| (k.clone(), param_display(v)))
                        .collect(),
                    pads: pads.clone(),
                    fitted: c.fitted(),
                    at,
                },
            );
            ch.requires = requires;
            match c.package.map(|u| s.pool.packages.get(&u)) {
                None => ch.issues.push((
                    Status::Error,
                    format!("{} has no footprint assigned", c.designator),
                )),
                Some(None) => ch.issues.push((
                    Status::Error,
                    format!("the footprint for {} is not in the library", c.designator),
                )),
                Some(Some(p)) => {
                    let missing: Vec<&str> = pads
                        .keys()
                        .filter(|pad| p.pad(pad).is_none())
                        .map(|s| s.as_str())
                        .collect();
                    if !missing.is_empty() {
                        ch.issues.push((
                            Status::Error,
                            format!("{} has no pad {}", p.name, missing.join(", ")),
                        ));
                    }
                }
            }
        }
    }

    for fid in &links.unlinked {
        let f = &board.footprints[fid];
        if !o.remove_unlinked_footprints
            || f.properties.get("board_only").is_some_and(|v| v == "true")
        {
            continue;
        }
        let impact = footprint_impact(board, *fid);
        let ch = b.push(
            ChangeKind::RemoveComponent,
            &f.reference,
            f.reference.clone(),
            Some(f.package.name.clone()),
            None,
            root_cause(
                &root,
                None,
                format!("no schematic component links to {}", f.reference),
            ),
            impact,
            Op::RemoveComponent {
                reference: f.reference.clone(),
            },
        );
        if ch.impact.copper() > 0 {
            ch.issues.push((
                Status::Warning,
                format!("{} routed items lose their pad", ch.impact.copper()),
            ));
        }
        if f.locked {
            ch.issues
                .push((Status::Warning, format!("{} is locked", f.reference)));
        }
    }

    for (ci, fid) in &links.by_comp {
        let c = &nl.components[*ci];
        let f = &board.footprints[fid];
        let key = c.path.to_string();
        let synced = base
            .components
            .get(&key)
            .or_else(|| f.component.as_ref().and_then(|p| base.components.get(p)));
        // Designator.
        if f.reference != c.designator {
            let board_edit = synced.is_some_and(|b| b.designator == c.designator);
            let both =
                synced.is_some_and(|b| b.designator != c.designator && b.designator != f.reference);
            let ch = b.push(
                ChangeKind::ChangeDesignator,
                &key,
                c.designator.clone(),
                Some(f.reference.clone()),
                Some(c.designator.clone()),
                comp_cause(
                    c,
                    &root,
                    if c.renamed_from.is_some() {
                        format!(
                            "renamed in the schematic; the board still calls it {}",
                            f.reference
                        )
                    } else {
                        format!("annotated {} in the schematic", c.designator)
                    },
                ),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SetDesignator {
                    reference: f.reference.clone(),
                    to: c.designator.clone(),
                    path: key.clone(),
                },
            );
            if board_edit {
                ch.issues.push((
                    Status::Warning,
                    format!(
                        "{} was renumbered on the board; back-annotate to keep it",
                        f.reference
                    ),
                ));
            }
            if both {
                ch.conflict = Some(format!(
                    "renamed on both sides since the last sync ({} in the schematic, {} on the board)",
                    c.designator, f.reference
                ));
            }
        }
        // Footprint.
        let pkg = package_name(s.pool, c.package);
        // A package the library doesn't have still differs (and fails validation).
        let differs = match (&c.package, &pkg) {
            (Some(_), Some(p)) => *p != f.package.name,
            (Some(_), None) => true,
            (None, _) => false,
        };
        if !o.ignore_footprints && differs {
            let to = pkg
                .clone()
                .unwrap_or_else(|| "a footprint missing from the library".to_owned());
            let board_edit = synced.is_some_and(|b| b.package == to);
            let both = synced.is_some_and(|b| b.package != to && b.package != f.package.name);
            let mut impact = footprint_impact(board, *fid);
            impact.footprints = vec![f.reference.clone()];
            let ch = b.push(
                ChangeKind::ChangeFootprint,
                &key,
                c.designator.clone(),
                Some(f.package.name.clone()),
                Some(to.clone()),
                comp_cause(c, &root, format!("the part of {} uses {to}", c.designator)),
                impact,
                Op::SetFootprint {
                    reference: f.reference.clone(),
                    package: c.package,
                },
            );
            match c.package.and_then(|u| s.pool.packages.get(&u)) {
                None => ch.issues.push((
                    Status::Error,
                    format!("the footprint for {} is not in the library", c.designator),
                )),
                Some(p) => {
                    let sp = sch_pads(nl, c);
                    let used: BTreeSet<&String> = f
                        .pad_nets
                        .keys()
                        .chain(sp.iter().filter(|(_, n)| n.is_some()).map(|(p, _)| p))
                        .collect();
                    let missing: Vec<&str> = used
                        .iter()
                        .filter(|pad| p.pad(pad).is_none())
                        .map(|s| s.as_str())
                        .collect();
                    if !missing.is_empty() {
                        ch.issues.push((
                            Status::Error,
                            format!(
                                "pads don't match: {} has {} pads, {} has {}; no pad {} for its nets",
                                f.package.name,
                                f.package.pads.len(),
                                p.name,
                                p.pads.len(),
                                missing.join(", ")
                            ),
                        ));
                    } else if p.pads.len() != f.package.pads.len() {
                        ch.issues.push((
                            Status::Warning,
                            format!(
                                "pad count changes from {} to {}",
                                f.package.pads.len(),
                                p.pads.len()
                            ),
                        ));
                    }
                }
            }
            if f.locked {
                ch.issues.push((
                    Status::Warning,
                    format!("{} is locked; its position is kept", f.reference),
                ));
            }
            if board_edit {
                ch.issues.push((
                    Status::Warning,
                    format!(
                        "the footprint was changed on the board; back-annotate to keep {}",
                        f.package.name
                    ),
                ));
            }
            if both {
                ch.conflict = Some(format!(
                    "footprint changed on both sides since the last sync ({to} and {})",
                    f.package.name
                ));
            }
        }
        // Value and parameters.
        let value = component_value(c).unwrap_or_default();
        if !o.ignore_values && value != f.value {
            b.push(
                ChangeKind::ChangeValue,
                &key,
                c.designator.clone(),
                Some(f.value.clone()).filter(|v| !v.is_empty()),
                Some(value.clone()).filter(|v| !v.is_empty()),
                comp_cause(
                    c,
                    &root,
                    format!(
                        "{} is {} in the schematic",
                        c.designator,
                        if value.is_empty() { "unvalued" } else { &value }
                    ),
                ),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SetValue {
                    reference: f.reference.clone(),
                    value,
                },
            );
        }
        if !o.ignore_parameters {
            let want: BTreeMap<String, String> = c
                .params
                .iter()
                .filter(|(k, _)| !o.ignored_parameters.contains(*k))
                .map(|(k, v)| (k.clone(), param_display(v)))
                .collect();
            let mut keys: BTreeSet<String> = want.keys().cloned().collect();
            // Parameters the ECO wrote before and the schematic dropped.
            keys.extend(
                f.properties
                    .keys()
                    .filter(|k| k.starts_with("param:"))
                    .map(|k| k["param:".len()..].to_owned())
                    .filter(|k| !o.ignored_parameters.contains(k)),
            );
            for k in keys {
                let have = f.properties.get(&format!("param:{k}")).cloned();
                let to = want.get(&k).cloned();
                if have == to {
                    continue;
                }
                b.push(
                    ChangeKind::ChangeParameter,
                    &format!("{key}#{k}"),
                    format!("{} {}", c.designator, k.replace('_', " ")),
                    have,
                    to.clone(),
                    comp_cause(
                        c,
                        &root,
                        match &to {
                            Some(v) => format!(
                                "{} {} is {v} in the schematic",
                                c.designator,
                                k.replace('_', " ")
                            ),
                            None => {
                                format!("{} no longer sets {}", c.designator, k.replace('_', " "))
                            }
                        },
                    ),
                    Impact {
                        footprints: vec![f.reference.clone()],
                        ..Impact::default()
                    },
                    Op::SetParam {
                        reference: f.reference.clone(),
                        key: k.clone(),
                        value: to,
                    },
                );
            }
        }
        if !o.ignore_fitting && c.fitted() != f.fitted {
            let fit = |x: bool| if x { "fitted" } else { "not fitted" }.to_owned();
            b.push(
                ChangeKind::ChangeFitted,
                &key,
                c.designator.clone(),
                Some(fit(f.fitted)),
                Some(fit(c.fitted())),
                comp_cause(
                    c,
                    &root,
                    format!(
                        "{} is {} in variant {}",
                        c.designator,
                        fit(c.fitted()),
                        nl.variant.as_deref().unwrap_or("default")
                    ),
                ),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SetFitted {
                    reference: f.reference.clone(),
                    fitted: c.fitted(),
                },
            );
        }
        // Pin connections.
        let want = sch_pads(nl, c);
        let mut pads: BTreeSet<String> = want.keys().cloned().collect();
        pads.extend(f.pad_nets.keys().cloned());
        for pad in pads {
            let have_board = f.pad_nets.get(&pad).map(|n| board.net_name(*n).to_owned());
            let have = have_board.as_deref().map(board_as_sch);
            let to = want.get(&pad).cloned().flatten();
            if have == to {
                continue;
            }
            if f.package.pad(&pad).is_none() && have.is_none() {
                // The footprint change reports pads the board footprint doesn't have.
                continue;
            }
            let synced_net = synced.and_then(|s| s.pads.get(&pad)).cloned();
            let board_edit = synced.is_some() && synced_net == to;
            let both =
                synced.is_some() && synced_net != to && synced_net.as_ref() != have_board.as_ref();
            let impact = copper_at_pad(board, *fid, &pad);
            let mut requires = Vec::new();
            if let Some(r) = to.as_deref().and_then(add_net_req) {
                requires.push(r);
            }
            let ch = b.push(
                ChangeKind::ChangePinNet,
                &format!("{key}#{pad}"),
                format!("{}.{}", c.designator, pad),
                have_board.as_deref().map(net_label),
                to.as_deref().map(net_label),
                comp_cause(
                    c,
                    &root,
                    match &to {
                        Some(n) => {
                            format!(
                                "pin {}.{} is on {} in the schematic",
                                c.designator,
                                pad,
                                net_label(n)
                            )
                        }
                        None => format!(
                            "pin {}.{} is unconnected in the schematic",
                            c.designator, pad
                        ),
                    },
                ),
                impact,
                Op::SetPadNet {
                    reference: f.reference.clone(),
                    pad: pad.clone(),
                    net: to,
                },
            );
            ch.requires = requires;
            if ch.impact.copper() > 0 {
                ch.issues.push((
                    Status::Warning,
                    format!(
                        "{} routed items on {} touch this pad",
                        ch.impact.copper(),
                        have_board
                            .as_deref()
                            .map(net_label)
                            .unwrap_or_else(|| "no net".into())
                    ),
                ));
            }
            if board_edit {
                ch.issues.push((
                    Status::Warning,
                    "the pin was swapped on the board; back-annotate to keep the swap".to_owned(),
                ));
            }
            if both {
                ch.conflict =
                    Some("the pin's net changed on both sides since the last sync".to_owned());
            }
        }
    }

    // ---- Net classes, membership, rules.
    if !o.ignore_net_classes {
        for (name, nc) in &s.design.net_classes {
            let params: BTreeMap<String, String> = nc
                .params
                .iter()
                .map(|(k, v)| (k.clone(), param_display(v)))
                .collect();
            let src = (!nc.source.file.is_empty()).then(|| nc.source.to_string());
            match base.net_classes.get(name) {
                None => {
                    b.push(
                        ChangeKind::AddNetClass,
                        name,
                        name.clone(),
                        None,
                        Some(format!("{} rules", params.len())),
                        root_cause(
                            &root,
                            src.clone(),
                            format!("net class {name} is declared in the schematic"),
                        ),
                        Impact::default(),
                        Op::SetNetClassDef {
                            name: name.clone(),
                            params: Some(params.clone()),
                        },
                    );
                }
                Some(have) if !o.ignore_rules => {
                    let keys: BTreeSet<&String> = have.keys().chain(params.keys()).collect();
                    for k in keys {
                        let (from, to) = (have.get(k).cloned(), params.get(k).cloned());
                        if from == to {
                            continue;
                        }
                        let nets: Vec<String> = board
                            .nets
                            .iter()
                            .filter(|n| n.class.as_deref() == Some(name))
                            .map(|n| n.name.clone())
                            .collect();
                        b.push(
                            ChangeKind::ChangeRule,
                            &format!("{name}#{k}"),
                            format!("{name} {}", k.replace('_', " ")),
                            from,
                            to,
                            root_cause(
                                &root,
                                src.clone(),
                                format!(
                                    "net class {name} sets {} in the schematic",
                                    k.replace('_', " ")
                                ),
                            ),
                            Impact {
                                nets,
                                ..Impact::default()
                            },
                            Op::SetRule {
                                class: name.clone(),
                                key: k.clone(),
                                value: params.get(k).cloned(),
                            },
                        );
                    }
                }
                _ => {}
            }
        }
        for name in base
            .net_classes
            .keys()
            .filter(|k| !s.design.net_classes.contains_key(*k))
        {
            b.push(
                ChangeKind::RemoveNetClass,
                name,
                name.clone(),
                Some(name.clone()),
                None,
                root_cause(
                    &root,
                    None,
                    format!("net class {name} is no longer declared"),
                ),
                Impact::default(),
                Op::SetNetClassDef {
                    name: name.clone(),
                    params: None,
                },
            );
        }
        for n in &nl.nets {
            let Some(bn) = board.nets.iter().find(|x| board_as_sch(&x.name) == n.name) else {
                continue;
            };
            if bn.class == n.class {
                continue;
            }
            let (sheet, sheet_path) = sheet_of(&n.path, &root);
            let mut requires = Vec::new();
            if let Some(cls) = &n.class {
                if !base.net_classes.contains_key(cls) {
                    requires.push(format!("add_net_class:{cls}"));
                }
            }
            if let Some(r) = add_net_req(&n.name) {
                requires.push(r);
            }
            let ch = b.push(
                ChangeKind::ChangeNetClass,
                &n.name,
                net_label(&n.name),
                bn.class.clone(),
                n.class.clone(),
                Cause {
                    sheet,
                    sheet_path,
                    source: None,
                    reason: match &n.class {
                        Some(c) => format!("{} is in net class {c}", net_label(&n.name)),
                        None => format!("{} has no net class", net_label(&n.name)),
                    },
                },
                Impact {
                    nets: vec![n.name.clone()],
                    ..Impact::default()
                },
                Op::SetNetClass {
                    net: n.name.clone(),
                    class: n.class.clone(),
                },
            );
            ch.requires = requires;
        }
    }
    // Nets added with a class carry it; their class must exist.
    for ch in b.changes.iter_mut() {
        if let Op::AddNet {
            class: Some(cls), ..
        } = &ch.op
        {
            if !o.ignore_net_classes
                && !base.net_classes.contains_key(cls)
                && s.design.net_classes.contains_key(cls)
            {
                ch.requires.push(format!("add_net_class:{cls}"));
            }
        }
    }

    // ---- Differential pairs.
    if !o.ignore_diff_pairs {
        let mut want: BTreeMap<String, (String, String)> = BTreeMap::new();
        for d in &nl.diff_pairs {
            want.insert(
                d.name.clone(),
                (nl.nets[d.p].name.clone(), nl.nets[d.n].name.clone()),
            );
        }
        for (name, pair) in &want {
            if base.diff_pairs.get(name) == Some(pair) {
                continue;
            }
            let mut requires: Vec<String> = [&pair.0, &pair.1]
                .iter()
                .filter_map(|n| add_net_req(n))
                .collect();
            requires.sort();
            let ch = b.push(
                ChangeKind::AddDiffPair,
                name,
                name.clone(),
                base.diff_pairs.get(name).map(|(p, n)| format!("{p} / {n}")),
                Some(format!("{} / {}", pair.0, pair.1)),
                root_cause(
                    &root,
                    None,
                    format!("differential pair {name} is declared in the schematic"),
                ),
                Impact {
                    nets: vec![pair.0.clone(), pair.1.clone()],
                    ..Impact::default()
                },
                Op::SetDiffPair {
                    name: name.clone(),
                    pair: Some(pair.clone()),
                },
            );
            ch.requires = requires;
        }
        for (name, pair) in base
            .diff_pairs
            .iter()
            .filter(|(k, _)| !want.contains_key(*k))
        {
            b.push(
                ChangeKind::RemoveDiffPair,
                name,
                name.clone(),
                Some(format!("{} / {}", pair.0, pair.1)),
                None,
                root_cause(
                    &root,
                    None,
                    format!("differential pair {name} is no longer declared"),
                ),
                Impact {
                    nets: vec![pair.0.clone(), pair.1.clone()],
                    ..Impact::default()
                },
                Op::SetDiffPair {
                    name: name.clone(),
                    pair: None,
                },
            );
        }
    }

    // ---- Component classes and rooms (multi-channel).
    let mut channels: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut authored: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (ci, c) in nl.components.iter().enumerate() {
        if let Some(ch) = channel_of(c) {
            channels.entry(ch).or_default().push(ci);
        }
        for cls in &c.classes {
            authored.entry(cls.clone()).or_default().push(ci);
        }
    }
    let designators = |cis: &[usize]| -> Vec<String> {
        let mut v: Vec<String> = cis
            .iter()
            .map(|ci| nl.components[*ci].designator.clone())
            .collect();
        v.sort_by_key(|d| natural(d));
        v
    };
    let add_comp_req = |cis: &[usize]| -> Vec<String> {
        cis.iter()
            .filter(|ci| !links.by_comp.contains_key(ci))
            .map(|ci| format!("add_component:{}", nl.components[*ci].path))
            .collect()
    };
    let first_comp = |cis: &[usize], why: String| -> Cause {
        cis.first()
            .map(|ci| {
                let c = &nl.components[*ci];
                let (sheet, sheet_path) = sheet_of(&c.path, &root);
                Cause {
                    sheet,
                    sheet_path,
                    source: None,
                    reason: why.clone(),
                }
            })
            .unwrap_or_else(|| root_cause(&root, None, why))
    };
    if !o.ignore_component_classes {
        let mut want: BTreeMap<String, Vec<usize>> = authored.clone();
        if o.classes_per_channel {
            for (k, v) in &channels {
                want.insert(k.clone(), v.clone());
            }
        }
        for (name, cis) in &want {
            let members = designators(cis);
            let have = base.component_classes.get(name);
            if have == Some(&members) {
                continue;
            }
            let kind = if have.is_some() {
                ChangeKind::ChangeComponentClass
            } else {
                ChangeKind::AddComponentClass
            };
            let ch = b.push(
                kind,
                name,
                name.clone(),
                have.map(|m| format!("{} members", m.len())),
                Some(format!("{} members", members.len())),
                first_comp(
                    cis,
                    if channels.contains_key(name) {
                        format!("channel {name} of a repeated sheet")
                    } else {
                        format!("components tagged {name}")
                    },
                ),
                Impact {
                    footprints: members.clone(),
                    ..Impact::default()
                },
                Op::SetComponentClass {
                    name: name.clone(),
                    members: Some(members),
                },
            );
            ch.requires = add_comp_req(cis);
        }
        for name in base
            .component_classes
            .keys()
            .filter(|k| !want.contains_key(*k))
        {
            b.push(
                ChangeKind::RemoveComponentClass,
                name,
                name.clone(),
                Some(name.clone()),
                None,
                root_cause(
                    &root,
                    None,
                    format!("component class {name} is gone from the schematic"),
                ),
                Impact::default(),
                Op::SetComponentClass {
                    name: name.clone(),
                    members: None,
                },
            );
        }
    }
    if !o.ignore_rooms && o.rooms_per_channel {
        let fp_ref_of = |ci: usize| links.by_comp.get(&ci).map(|f| board.footprints[f].bbox());
        for (name, cis) in &channels {
            let members = designators(cis);
            let existing = board.rooms.iter().find(|r| &r.name == name);
            match existing {
                Some(r) => {
                    let mut have = r.members.clone();
                    have.sort_by_key(|d| natural(d));
                    if have == members {
                        continue;
                    }
                    let ch = b.push(
                        ChangeKind::ChangeRoom,
                        name,
                        name.clone(),
                        Some(format!("{} parts", have.len())),
                        Some(format!("{} parts", members.len())),
                        first_comp(cis, format!("channel {name} has {} parts", members.len())),
                        Impact {
                            footprints: members.clone(),
                            ..Impact::default()
                        },
                        Op::SetRoom {
                            name: name.clone(),
                            outline: None,
                            members,
                        },
                    );
                    ch.requires = add_comp_req(cis);
                }
                None => {
                    // Around the channel's placed parts, else its staging block.
                    let placed = cis
                        .iter()
                        .filter_map(|ci| fp_ref_of(*ci))
                        .fold(Rect::EMPTY, |a, r| a.union(r));
                    let area = if placed.is_empty() {
                        new_rooms.get(name).copied().unwrap_or(Rect::EMPTY)
                    } else {
                        placed.inflate(mm(1.0))
                    };
                    if area.is_empty() {
                        continue;
                    }
                    let staged = placed.is_empty();
                    let ch = b.push(
                        ChangeKind::AddRoom,
                        name,
                        name.clone(),
                        None,
                        Some(format!("{} parts", members.len())),
                        first_comp(
                            cis,
                            format!("channel {name} of a repeated sheet gets a room"),
                        ),
                        Impact {
                            footprints: members.clone(),
                            placement: Some(Placement {
                                at: area.center(),
                                room: Some(name.clone()),
                                staged,
                            }),
                            ..Impact::default()
                        },
                        Op::SetRoom {
                            name: name.clone(),
                            outline: Some(Polygon::from_ring(rect_ring(area))),
                            members,
                        },
                    );
                    ch.requires = add_comp_req(cis);
                }
            }
        }
        for r in board
            .rooms
            .iter()
            .filter(|r| r.name.ends_with(']') && !channels.contains_key(&r.name))
        {
            b.push(
                ChangeKind::RemoveRoom,
                &r.name,
                r.name.clone(),
                Some(format!("{} parts", r.members.len())),
                None,
                root_cause(
                    &root,
                    None,
                    format!("channel {} is no longer in the schematic", r.name),
                ),
                Impact::default(),
                Op::SetRoom {
                    name: r.name.clone(),
                    outline: None,
                    members: Vec::new(),
                },
            );
        }
    }

    finish(Direction::Forward, b.changes, o)
}

/* ------------------------------------------------------------------------------------------------
 * Back-annotation
 * --------------------------------------------------------------------------------------------- */

/// Board → schematic: designator renumbering, pin and gate swaps and footprint changes made on
/// the board, as changes to the schematic.
pub fn back_annotate(s: &Schematic, board: &Board, o: &ComparatorOptions) -> EcoPlan {
    let nl = s.netlist;
    let root = s.design.root.clone();
    let links = link(s, board, o);
    let mut b = Builder {
        changes: Vec::new(),
        direction: Direction::Back,
    };
    for (ci, fid) in &links.by_comp {
        let c = &nl.components[*ci];
        let f = &board.footprints[fid];
        let key = c.path.to_string();
        let synced = board.sync.components.get(&key).or_else(|| {
            f.component
                .as_ref()
                .and_then(|p| board.sync.components.get(p))
        });
        let (sheet_name, _) = sheet_of(&c.path, &root);
        let sheet_def = c.sheet.clone();
        if f.reference != c.designator && synced.is_none_or(|b| b.designator != f.reference) {
            let both =
                synced.is_some_and(|b| b.designator != c.designator && b.designator != f.reference);
            let ch = b.push(
                ChangeKind::ChangeDesignator,
                &key,
                f.reference.clone(),
                Some(c.designator.clone()),
                Some(f.reference.clone()),
                comp_cause(c, &root, format!("renumbered {} on the board", f.reference)),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SchDesignator {
                    path: key.clone(),
                    sheet: sheet_def.clone(),
                    to: f.reference.clone(),
                },
            );
            if both {
                ch.conflict = Some(format!(
                    "renamed on both sides since the last sync ({} in the schematic, {} on the board)",
                    c.designator, f.reference
                ));
            }
        }
        let pkg = package_name(s.pool, c.package);
        if !o.ignore_footprints
            && pkg.as_deref().is_some_and(|p| p != f.package.name)
            && synced.is_none_or(|b| b.package != f.package.name)
        {
            let both = synced
                .is_some_and(|b| Some(&b.package) != pkg.as_ref() && b.package != f.package.name);
            // A part of the same entity on the board's package, or the package itself for an
            // entity reference.
            let decl_part = s
                .design
                .sheets
                .get(&sheet_def)
                .and_then(|sd| sd.components.iter().find(|d| d.name == c.path.leaf()))
                .map(|d| d.part.clone());
            let alt = c.entity.and_then(|e| {
                let pkg_uuid = s
                    .pool
                    .packages
                    .values()
                    .find(|p| p.name == f.package.name)?
                    .uuid;
                s.pool
                    .parts
                    .values()
                    .filter(|p| p.entity == e && p.package == pkg_uuid)
                    .map(|p| p.mpn.clone())
                    .min()
            });
            let ch = b.push(
                ChangeKind::ChangeFootprint,
                &key,
                f.reference.clone(),
                pkg.clone(),
                Some(f.package.name.clone()),
                comp_cause(
                    c,
                    &root,
                    format!("{} uses {} on the board", f.reference, f.package.name),
                ),
                Impact {
                    footprints: vec![f.reference.clone()],
                    ..Impact::default()
                },
                Op::SchFootprint {
                    path: key.clone(),
                    sheet: sheet_def.clone(),
                    component: c.path.leaf().to_owned(),
                    part: alt.clone(),
                    package: f.package.name.clone(),
                },
            );
            if both {
                ch.conflict = Some(format!(
                    "footprint changed on both sides since the last sync ({} and {})",
                    pkg.clone().unwrap_or_default(),
                    f.package.name
                ));
            }
            match (&decl_part, &alt) {
                (Some(PartRef::Entity { .. }), _) => {}
                (_, Some(mpn)) => ch
                    .issues
                    .push((Status::Warning, format!("the part changes to {mpn}"))),
                _ => ch.issues.push((
                    Status::Error,
                    format!(
                        "no part in the library puts this component on {}",
                        f.package.name
                    ),
                )),
            }
        }
        // Swaps: pin pairs whose board nets are each other's schematic nets.
        let want = sch_pads(nl, c);
        let board_net = |pad: &str| f.pad_nets.get(pad).map(|n| board.net_name(*n).to_owned());
        let pin_of = |pad: &str| c.pins.iter().find(|p| p.pads.iter().any(|x| x == pad));
        let net_of_pin = |p: &crate::elab::PinInst, board_side: bool| -> Option<String> {
            let pad = p.pads.first()?;
            if board_side {
                board_net(pad)
            } else {
                want.get(pad).cloned().flatten()
            }
        };
        // Whole-gate swaps first: every pin of gate A carries gate B's schematic net and back.
        let gates: BTreeSet<&str> = c
            .pins
            .iter()
            .map(|p| p.gate.as_str())
            .filter(|g| !g.is_empty())
            .collect();
        let gates: Vec<&str> = gates.into_iter().collect();
        let mut gate_swapped: BTreeSet<String> = BTreeSet::new();
        for (i, ga) in gates.iter().enumerate() {
            for gb in gates.iter().skip(i + 1) {
                if gate_swapped.contains(*ga) || gate_swapped.contains(*gb) {
                    continue;
                }
                let pins_a: Vec<&crate::elab::PinInst> =
                    c.pins.iter().filter(|p| p.gate == *ga).collect();
                let partner = |pa: &crate::elab::PinInst| {
                    c.pins.iter().find(|p| p.gate == *gb && p.pin == pa.pin)
                };
                let same_shape = pins_a.len() == c.pins.iter().filter(|p| p.gate == *gb).count()
                    && pins_a.iter().all(|pa| partner(pa).is_some());
                if !same_shape {
                    continue;
                }
                let differs = pins_a
                    .iter()
                    .any(|pa| net_of_pin(pa, true) != net_of_pin(pa, false));
                let swapped = pins_a.iter().all(|pa| {
                    let pb = partner(pa).unwrap();
                    net_of_pin(pa, true) == net_of_pin(pb, false)
                        && net_of_pin(pb, true) == net_of_pin(pa, false)
                });
                if !(differs && swapped) {
                    continue;
                }
                gate_swapped.insert((*ga).to_owned());
                gate_swapped.insert((*gb).to_owned());
                let mut impact = Impact::default();
                for pa in &pins_a {
                    let pb = partner(pa).unwrap();
                    for pad in pa.pads.iter().chain(pb.pads.iter()) {
                        impact.merge(copper_at_pad(board, *fid, pad));
                    }
                }
                let ch = b.push(
                    ChangeKind::SwapGates,
                    &format!("{key}#{ga}#{gb}"),
                    format!("{} gates {ga} \u{2194} {gb}", f.reference),
                    Some(format!("gate {ga}")),
                    Some(format!("gate {gb}")),
                    comp_cause(
                        c,
                        &root,
                        format!(
                            "gates {ga} and {gb} of {} were swapped on the board",
                            f.reference
                        ),
                    ),
                    impact,
                    Op::SchSwapGates {
                        path: key.clone(),
                        sheet: sheet_def.clone(),
                        component: c.path.leaf().to_owned(),
                        a: (*ga).to_owned(),
                        b: (*gb).to_owned(),
                    },
                );
                if c.channel.is_some() {
                    ch.issues.push((Status::Warning, format!("sheet {} repeats; the swap changes every channel, not only {sheet_name}", c.sheet)));
                }
            }
        }
        // Single pin swaps among the rest, preferring partners of the same pin name.
        let pads: Vec<&String> = want
            .keys()
            .filter(|p| pin_of(p).is_none_or(|x| !gate_swapped.contains(&x.gate)))
            .collect();
        let mut used: BTreeSet<&String> = BTreeSet::new();
        for a in &pads {
            if used.contains(*a) {
                continue;
            }
            let (wa, ba) = (want[*a].clone(), board_net(a));
            if wa == ba || wa.is_none() {
                continue;
            }
            let fits = |bp: &&&String| {
                !used.contains(**bp)
                    && *bp != a
                    && want[**bp] != board_net(bp)
                    && wa == board_net(bp)
                    && want[**bp] == ba
            };
            let same_name = pads
                .iter()
                .filter(fits)
                .find(|bp| pin_of(bp).map(|p| &p.pin) == pin_of(a).map(|p| &p.pin));
            let Some(bp) = same_name.or_else(|| pads.iter().find(fits)) else {
                continue;
            };
            used.insert(a);
            used.insert(bp);
            let (Some(pa), Some(pb)) = (pin_of(a), pin_of(bp)) else {
                continue;
            };
            let (qa, qb) = (qualified(pa), qualified(pb));
            let mut impact = copper_at_pad(board, *fid, a);
            impact.merge(copper_at_pad(board, *fid, bp));
            let ch = b.push(
                ChangeKind::SwapPins,
                &format!("{key}#{a}#{bp}"),
                format!("{}.{a} \u{2194} {}.{bp}", f.reference, f.reference),
                Some(format!(
                    "{qa} on {}",
                    net_label(&wa.clone().unwrap_or_default())
                )),
                Some(format!(
                    "{qa} on {}",
                    net_label(&want[*bp].clone().unwrap_or_default())
                )),
                comp_cause(
                    c,
                    &root,
                    format!(
                        "pins {a} and {bp} of {} were swapped on the board",
                        f.reference
                    ),
                ),
                impact,
                Op::SchSwapPins {
                    path: key.clone(),
                    sheet: sheet_def.clone(),
                    component: c.path.leaf().to_owned(),
                    a: qa.clone(),
                    b: qb.clone(),
                },
            );
            if pa.kind != pb.kind || pa.gate != pb.gate {
                ch.issues.push((
                    Status::Warning,
                    format!("{qa} and {qb} are not equivalent pins"),
                ));
            }
            if c.channel.is_some() {
                ch.issues.push((
                    Status::Warning,
                    format!("sheet {} repeats; the swap changes every channel", c.sheet),
                ));
            }
        }
    }
    finish(Direction::Back, b.changes, o)
}

fn qualified(p: &crate::elab::PinInst) -> String {
    if p.gate.is_empty() {
        p.pin.clone()
    } else {
        format!("{}.{}", p.gate, p.pin)
    }
}

/* ------------------------------------------------------------------------------------------------
 * Agent change sets
 * --------------------------------------------------------------------------------------------- */

/// An agent's model edits as an ECO: applied to a copy of the design, elaborated with the lock,
/// compared with the board. Every change carries the agent as its origin.
#[allow(clippy::too_many_arguments)]
pub fn agent_change_set(
    design: &Design,
    edits: &[ModelEdit],
    title: &str,
    pool: &Pool,
    lock: &LockSection,
    board: &Board,
    elab: &ElabOptions,
    o: &ComparatorOptions,
) -> Result<(Design, EcoPlan), String> {
    let mut next = design.clone();
    for e in edits {
        crate::review::apply(&mut next, e)?;
    }
    let mut lock = lock.clone();
    let nl = elaborate(&next, pool, elab, Some(&lock));
    lock.reconcile(&nl.lock_items(), &mut DerivedUuids { namespace: 0x5eed });
    let s = Schematic {
        design: &next,
        netlist: &nl,
        pool,
        lock: &lock,
    };
    let mut plan = compare(&s, board, o);
    for c in &mut plan.changes {
        c.origin = Origin::Agent {
            title: title.to_owned(),
        };
    }
    Ok((next, plan))
}

/* ------------------------------------------------------------------------------------------------
 * Validation
 * --------------------------------------------------------------------------------------------- */

#[derive(Clone, Debug, PartialEq)]
pub struct Check {
    pub status: Status,
    pub reasons: Vec<String>,
}

impl Check {
    pub fn reason(&self) -> Option<String> {
        (!self.reasons.is_empty()).then(|| self.reasons.join("; "))
    }
}

/// Status of each selected change, given the whole selection (Altium's Validate Changes).
pub fn validate(
    plan: &EcoPlan,
    board: &Board,
    selected: &BTreeSet<String>,
) -> BTreeMap<String, Check> {
    // Designators the selection frees and claims.
    let mut freed: BTreeSet<&str> = BTreeSet::new();
    let mut claimed: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for c in plan.changes.iter().filter(|c| selected.contains(&c.id)) {
        match &c.op {
            Op::RemoveComponent { reference } => {
                freed.insert(reference);
            }
            Op::SetDesignator { reference, to, .. } => {
                freed.insert(reference);
                claimed.entry(to).or_default().push(&c.id);
            }
            Op::AddComponent { designator, .. } => {
                claimed.entry(designator).or_default().push(&c.id)
            }
            _ => {}
        }
    }
    let mut out = BTreeMap::new();
    for c in plan.changes.iter().filter(|c| selected.contains(&c.id)) {
        let mut status = Status::Ok;
        let mut reasons = Vec::new();
        let mut raise = |s: Status, r: String, status: &mut Status| {
            *status = (*status).max(s);
            reasons.push(r);
        };
        if let Some(conflict) = &c.conflict {
            raise(Status::Error, format!("conflict: {conflict}"), &mut status);
        }
        for (s, r) in &c.issues {
            raise(*s, r.clone(), &mut status);
        }
        for r in &c.requires {
            if !selected.contains(r) {
                let what = plan
                    .get(r)
                    .map(|x| format!("{} {}", x.kind.label().to_lowercase(), x.item))
                    .unwrap_or_else(|| r.clone());
                raise(
                    Status::Error,
                    format!("needs {what} (not selected)"),
                    &mut status,
                );
            }
        }
        let claim = match &c.op {
            Op::SetDesignator { to, .. } => Some(to.as_str()),
            Op::AddComponent { designator, .. } => Some(designator.as_str()),
            _ => None,
        };
        if let Some(d) = claim {
            if claimed.get(d).is_some_and(|v| v.len() > 1) {
                raise(
                    Status::Error,
                    format!("{d} is claimed by more than one change"),
                    &mut status,
                );
            } else if board.footprint_by_ref(d).is_some() && !freed.contains(d) {
                raise(
                    Status::Error,
                    format!("{d} is already used on the board"),
                    &mut status,
                );
            }
        }
        out.insert(c.id.clone(), Check { status, reasons });
    }
    out
}

/* ------------------------------------------------------------------------------------------------
 * Apply
 * --------------------------------------------------------------------------------------------- */

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ApplyOptions {
    /// Delete the tracks and vias a change leaves without a net or pad (else they keep no net).
    pub remove_orphaned_copper: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApplyReport {
    pub applied: Vec<String>,
    pub removed_tracks: Vec<u64>,
    pub removed_vias: Vec<u64>,
    /// Copper left on no net (kept because orphan removal is off).
    pub unnetted: Vec<u64>,
    /// The board before the edit: the one undo step.
    pub undo: Box<Board>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ApplyError {
    /// Each selected change that failed validation, with why.
    pub failures: Vec<(String, String)>,
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} changes failed validation", self.failures.len())
    }
}

fn net_or_add(board: &mut Board, name: &str) -> NetId {
    board.add_net(name, None)
}

/// Executes the selected forward changes as one edit: validated first, all or nothing.
/// Placement and routing nothing touches stay as they were.
pub fn apply(
    board: &mut Board,
    plan: &EcoPlan,
    selected: &BTreeSet<String>,
    pool: &Pool,
    o: &ApplyOptions,
) -> Result<ApplyReport, ApplyError> {
    let checks = validate(plan, board, selected);
    let failures: Vec<(String, String)> = checks
        .iter()
        .filter(|(_, c)| c.status == Status::Error)
        .map(|(id, c)| (id.clone(), c.reason().unwrap_or_default()))
        .collect();
    if !failures.is_empty() {
        return Err(ApplyError { failures });
    }
    let before = board.clone();
    let mut next = board.clone();
    // The settings this update was compared with are the board's from now on.
    next.sync.options = plan.options.clone();
    let mut changes: Vec<&Change> = plan
        .changes
        .iter()
        .filter(|c| selected.contains(&c.id) && c.direction == Direction::Forward)
        .collect();
    // Net renames run first: an added part's pads look their nets up by the new names, and
    // would otherwise create a second net under a name a rename is about to take.
    changes.sort_by_key(|c| (c.kind != ChangeKind::RenameNet, c.kind));
    let mut removed_nets: BTreeSet<NetId> = BTreeSet::new();
    let mut touched_nets: BTreeSet<NetId> = BTreeSet::new();
    let mut moved_pads: Vec<(String, String)> = Vec::new();
    let by_ref = |b: &Board, r: &str| {
        b.footprints
            .values()
            .find(|f| f.reference == r)
            .map(|f| f.id)
    };
    for c in &changes {
        match &c.op {
            Op::RemoveComponent { reference } => {
                if let Some(id) = by_ref(&next, reference) {
                    let f = next.footprints.remove(&id).unwrap();
                    touched_nets.extend(f.pad_nets.values().copied());
                    if let Some(p) = &f.component {
                        next.sync.components.remove(p);
                    }
                    // Ignore rooms leaves every room as it is, members included.
                    if !plan.options.ignore_rooms {
                        for r in next.rooms.iter_mut() {
                            r.members.retain(|m| m != reference);
                        }
                    }
                }
            }
            Op::AddComponent {
                path,
                designator,
                package,
                value,
                params,
                pads,
                fitted,
                at,
            } => {
                let Some(pkg) = package.and_then(|u| pool.packages.get(&u)) else {
                    continue;
                };
                let id = next.place(designator, pkg.clone(), *at, 0.0, Side::Top);
                let mut nets = BTreeMap::new();
                for (pad, n) in pads {
                    nets.insert(pad.clone(), net_or_add(&mut next, n));
                }
                let f = next.footprints.get_mut(&id).unwrap();
                f.component = Some(path.clone());
                f.value = value.clone();
                f.fitted = *fitted;
                f.pad_nets = nets;
                for (k, v) in params {
                    f.properties.insert(format!("param:{k}"), v.clone());
                }
                next.sync.components.insert(
                    path.clone(),
                    SyncedComponent {
                        designator: designator.clone(),
                        package: pkg.name.clone(),
                        pads: pads.clone(),
                    },
                );
            }
            Op::SetDesignator {
                reference,
                to,
                path,
            } => {
                if let Some(id) = by_ref(&next, reference) {
                    let f = next.footprints.get_mut(&id).unwrap();
                    f.reference = to.clone();
                    let old_link = f.component.replace(path.clone());
                    let mut rec = old_link
                        .and_then(|p| next.sync.components.remove(&p))
                        .or_else(|| next.sync.components.remove(path));
                    if let Some(r) = rec.as_mut() {
                        r.designator = to.clone();
                    }
                    if let Some(r) = rec {
                        next.sync.components.insert(path.clone(), r);
                    }
                    for r in next.rooms.iter_mut() {
                        for m in r.members.iter_mut().filter(|m| *m == reference) {
                            *m = to.clone();
                        }
                    }
                }
            }
            Op::SetFootprint { reference, package } => {
                let Some(pkg) = package.and_then(|u| pool.packages.get(&u)) else {
                    continue;
                };
                if let Some(id) = by_ref(&next, reference) {
                    let f = next.footprints.get_mut(&id).unwrap();
                    f.package = pkg.clone();
                    f.pad_nets.retain(|pad, _| pkg.pad(pad).is_some());
                    if let Some(s) = f
                        .component
                        .clone()
                        .and_then(|p| next.sync.components.get_mut(&p))
                    {
                        s.package = pkg.name.clone();
                    }
                }
            }
            Op::SetValue { reference, value } => {
                if let Some(id) = by_ref(&next, reference) {
                    next.footprints.get_mut(&id).unwrap().value = value.clone();
                }
            }
            Op::SetParam {
                reference,
                key,
                value,
            } => {
                if let Some(id) = by_ref(&next, reference) {
                    let props = &mut next.footprints.get_mut(&id).unwrap().properties;
                    match value {
                        Some(v) => props.insert(format!("param:{key}"), v.clone()),
                        None => props.remove(&format!("param:{key}")),
                    };
                }
            }
            Op::SetFitted { reference, fitted } => {
                if let Some(id) = by_ref(&next, reference) {
                    next.footprints.get_mut(&id).unwrap().fitted = *fitted;
                }
            }
            Op::RenameNet { from, to } => {
                if let Some(n) = next.net(from) {
                    next.nets[n.0 as usize].name = to.clone();
                    for s in next.sync.components.values_mut() {
                        for v in s.pads.values_mut().filter(|v| *v == from) {
                            *v = to.clone();
                        }
                    }
                }
            }
            Op::AddNet { name, class } => {
                let n = next.add_net(name, class.as_deref());
                next.nets[n.0 as usize].class = class.clone();
            }
            Op::SetPadNet {
                reference,
                pad,
                net,
            } => {
                if let Some(id) = by_ref(&next, reference) {
                    let new = net.as_deref().map(|n| net_or_add(&mut next, n));
                    let f = next.footprints.get_mut(&id).unwrap();
                    let old = match new {
                        Some(n) => f.pad_nets.insert(pad.clone(), n),
                        None => f.pad_nets.remove(pad),
                    };
                    touched_nets.extend(old);
                    moved_pads.push((reference.clone(), pad.clone()));
                    let path = f.component.clone();
                    if let Some(s) = path.and_then(|p| next.sync.components.get_mut(&p)) {
                        match net {
                            Some(n) => s.pads.insert(pad.clone(), n.clone()),
                            None => s.pads.remove(pad),
                        };
                    }
                }
            }
            Op::RemoveNet { name } => {
                if let Some(n) = next.net(name) {
                    removed_nets.insert(n);
                }
            }
            Op::SetNetClassDef { name, params } => match params {
                Some(p) => {
                    next.sync.net_classes.insert(name.clone(), p.clone());
                }
                None => {
                    next.sync.net_classes.remove(name);
                    for n in next
                        .nets
                        .iter_mut()
                        .filter(|n| n.class.as_deref() == Some(name))
                    {
                        n.class = None;
                    }
                }
            },
            Op::SetNetClass { net, class } => {
                if let Some(n) = next.net(net) {
                    next.nets[n.0 as usize].class = class.clone();
                }
            }
            Op::SetDiffPair { name, pair } => match pair {
                Some(p) => {
                    next.sync.diff_pairs.insert(name.clone(), p.clone());
                }
                None => {
                    next.sync.diff_pairs.remove(name);
                }
            },
            Op::SetComponentClass { name, members } => match members {
                Some(m) => {
                    next.sync.component_classes.insert(name.clone(), m.clone());
                }
                None => {
                    next.sync.component_classes.remove(name);
                }
            },
            Op::SetRoom {
                name,
                outline,
                members,
            } => {
                if let Some(r) = next.rooms.iter_mut().find(|r| &r.name == name) {
                    if members.is_empty() && outline.is_none() {
                        next.rooms.retain(|r| &r.name != name);
                    } else {
                        r.members = members.clone();
                        if let Some(o) = outline {
                            r.outline = o.clone();
                        }
                    }
                } else if let Some(o) = outline {
                    next.rooms.push(Room {
                        name: name.clone(),
                        outline: o.clone(),
                        members: members.clone(),
                    });
                    next.rooms
                        .sort_by(|a, b| natural(&a.name).cmp(&natural(&b.name)));
                }
            }
            Op::SetRule { class, key, value } => {
                let e = next.sync.net_classes.entry(class.clone()).or_default();
                match value {
                    Some(v) => e.insert(key.clone(), v.clone()),
                    None => e.remove(key),
                };
            }
            Op::SchDesignator { .. }
            | Op::SchSwapPins { .. }
            | Op::SchSwapGates { .. }
            | Op::SchFootprint { .. } => {}
        }
    }
    // Copper the changes orphaned.
    let mut removed_tracks = Vec::new();
    let mut removed_vias = Vec::new();
    let mut unnetted = Vec::new();
    for n in &removed_nets {
        for t in next.tracks.values_mut().filter(|t| t.net == Some(*n)) {
            t.net = None;
            unnetted.push(t.id.0);
        }
        for v in next.vias.values_mut().filter(|v| v.net == Some(*n)) {
            v.net = None;
            unnetted.push(v.id.0);
        }
        for z in next.zones.values_mut().filter(|z| z.net == Some(*n)) {
            z.net = None;
        }
    }
    if o.remove_orphaned_copper {
        // Unnetted copper from removed nets, plus copper of touched nets left with no pad.
        let unnetted_set: BTreeSet<u64> = unnetted.iter().copied().collect();
        next.tracks.retain(|id, _| {
            let keep = !unnetted_set.contains(&id.0);
            if !keep {
                removed_tracks.push(id.0);
            }
            keep
        });
        next.vias.retain(|id, _| {
            let keep = !unnetted_set.contains(&id.0);
            if !keep {
                removed_vias.push(id.0);
            }
            keep
        });
        unnetted.clear();
        // Copper still on an old net that touches a pad now on another net shorts it.
        for (r, pad) in &moved_pads {
            let Some(fid) = by_ref(&before, r) else {
                continue;
            };
            let imp = copper_at_pad(&before, fid, pad);
            for t in imp.tracks {
                if next.tracks.remove(&ItemId(t)).is_some() {
                    removed_tracks.push(t);
                }
            }
            for v in imp.vias {
                if next.vias.remove(&ItemId(v)).is_some() {
                    removed_vias.push(v);
                }
            }
        }
        let nodes = net_nodes(&next);
        for net in &touched_nets {
            let Some(ns) = nodes.get(net) else { continue };
            let conn = islands(*net, ns.clone());
            let mut with_pad = vec![false; conn.islands];
            for (k, node) in conn.nodes.iter().enumerate() {
                if matches!(node.item, BItem::Pad(..)) {
                    with_pad[conn.island[k]] = true;
                }
            }
            for (k, node) in conn.nodes.iter().enumerate() {
                if with_pad[conn.island[k]] {
                    continue;
                }
                match node.item {
                    BItem::Track(id) => {
                        if next.tracks.remove(&id).is_some() {
                            removed_tracks.push(id.0);
                        }
                    }
                    BItem::Via(id) => {
                        if next.vias.remove(&id).is_some() {
                            removed_vias.push(id.0);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if !removed_nets.is_empty() {
        compact_nets(&mut next, &removed_nets);
    }
    removed_tracks.sort();
    removed_tracks.dedup();
    removed_vias.sort();
    removed_vias.dedup();
    unnetted.sort();
    *board = next;
    Ok(ApplyReport {
        applied: changes.iter().map(|c| c.id.clone()).collect(),
        removed_tracks,
        removed_vias,
        unnetted,
        undo: Box::new(before),
    })
}

/// Drops nets and renumbers every net reference (net ids are indices).
fn compact_nets(board: &mut Board, removed: &BTreeSet<NetId>) {
    let mut map: BTreeMap<NetId, NetId> = BTreeMap::new();
    let mut nets = Vec::new();
    for (i, n) in board.nets.iter().enumerate() {
        let id = NetId(i as u32);
        if !removed.contains(&id) {
            map.insert(id, NetId(nets.len() as u32));
            nets.push(n.clone());
        }
    }
    board.nets = nets;
    let re = |n: Option<NetId>| n.and_then(|n| map.get(&n).copied());
    for f in board.footprints.values_mut() {
        f.pad_nets = f
            .pad_nets
            .iter()
            .filter_map(|(p, n)| map.get(n).map(|m| (p.clone(), *m)))
            .collect();
    }
    for t in board.tracks.values_mut() {
        t.net = re(t.net);
    }
    for v in board.vias.values_mut() {
        v.net = re(v.net);
    }
    for z in board.zones.values_mut() {
        z.net = re(z.net);
    }
}

/// Reverts an applied ECO (the one undo step).
pub fn undo(board: &mut Board, report: ApplyReport) {
    *board = *report.undo;
}

/// What the checks found after an ECO.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckSummary {
    pub drc_errors: usize,
    pub drc_warnings: usize,
    /// DRC finding messages, sorted.
    pub drc: Vec<String>,
    pub erc_errors: usize,
    pub erc_warnings: usize,
}

/// Applies, refills pours and runs DRC on the board and ERC on the schematic.
pub fn apply_and_check(
    board: &mut Board,
    plan: &EcoPlan,
    selected: &BTreeSet<String>,
    pool: &Pool,
    o: &ApplyOptions,
    rules: &mut BoardRules,
    erc: &[Diagnostic],
) -> Result<(ApplyReport, CheckSummary), ApplyError> {
    let report = apply(board, plan, selected, pool, o)?;
    Ok((report, check(board, rules, erc)))
}

/// DRC (after a pour refill) and ERC counts.
pub fn check(board: &mut Board, rules: &mut BoardRules, erc: &[Diagnostic]) -> CheckSummary {
    let pairs: Vec<(NetId, NetId)> = board
        .sync
        .diff_pairs
        .values()
        .filter_map(|(p, n)| Some((board.net(p)?, board.net(n)?)))
        .collect();
    rules.rebuild(&crate::board_rules::BoardGraph::new(board, &pairs));
    let fills = crate::zone::fill_zones(board, rules, &pairs);
    let drc = Drc::run(board, rules, &pairs, &fills);
    let found = drc.findings();
    let mut msgs: Vec<String> = found.iter().map(|f| f.message.clone()).collect();
    msgs.sort();
    CheckSummary {
        drc_errors: found
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .count(),
        drc_warnings: found
            .iter()
            .filter(|f| f.severity == Severity::Warning)
            .count(),
        drc: msgs,
        erc_errors: erc.iter().filter(|d| d.severity == Severity::Error).count(),
        erc_warnings: erc
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count(),
    }
}

/* ------------------------------------------------------------------------------------------------
 * Back-annotation apply
 * --------------------------------------------------------------------------------------------- */

/// Executes the selected back-annotation changes on the schematic (and the lock, which holds
/// auto-annotated designators), all or nothing. The board's sync record takes the new state so
/// the next forward compare is clean for these items.
pub fn apply_back(
    design: &mut Design,
    lock: &mut LockSection,
    board: &mut Board,
    plan: &EcoPlan,
    selected: &BTreeSet<String>,
) -> Result<Vec<String>, ApplyError> {
    let checks = validate(plan, board, selected);
    let failures: Vec<(String, String)> = checks
        .iter()
        .filter(|(_, c)| c.status == Status::Error)
        .map(|(id, c)| (id.clone(), c.reason().unwrap_or_default()))
        .collect();
    if !failures.is_empty() {
        return Err(ApplyError { failures });
    }
    let mut next = design.clone();
    let mut next_lock = lock.clone();
    let mut applied = Vec::new();
    let mut errors = Vec::new();
    for c in plan.changes.iter().filter(|c| selected.contains(&c.id)) {
        let r = match &c.op {
            Op::SchDesignator { path, sheet, to } => {
                back_designator(&mut next, &mut next_lock, path, sheet, to)
            }
            Op::SchSwapPins {
                sheet,
                component,
                a,
                b,
                ..
            } => swap_endpoints(&mut next, sheet, component, |p| {
                if p == a {
                    Some(b.clone())
                } else if p == b {
                    Some(a.clone())
                } else {
                    None
                }
            }),
            Op::SchSwapGates {
                sheet,
                component,
                a,
                b,
                ..
            } => {
                let (pa, pb) = (format!("{a}."), format!("{b}."));
                swap_endpoints(&mut next, sheet, component, |p| {
                    p.strip_prefix(&pa)
                        .map(|r| format!("{pb}{r}"))
                        .or_else(|| p.strip_prefix(&pb).map(|r| format!("{pa}{r}")))
                })
            }
            Op::SchFootprint {
                sheet,
                component,
                part,
                package,
                ..
            } => {
                match next
                    .sheets
                    .get_mut(sheet)
                    .and_then(|s| s.components.iter_mut().find(|d| &d.name == component))
                {
                    Some(d) => match (&mut d.part, part) {
                        (PartRef::Entity { package: p, .. }, _) => {
                            *p = Some(package.clone());
                            Ok(())
                        }
                        (r, Some(mpn)) => {
                            *r = PartRef::Part(mpn.clone());
                            Ok(())
                        }
                        _ => Err(format!("no part puts {component} on {package}")),
                    },
                    None => Err(format!("{component} is not declared on sheet {sheet}")),
                }
            }
            _ => Ok(()),
        };
        match r {
            Ok(()) => applied.push(c.id.clone()),
            Err(e) => errors.push((c.id.clone(), e)),
        }
    }
    if !errors.is_empty() {
        return Err(ApplyError { failures: errors });
    }
    // The board already holds these states: record them as synced.
    for c in plan.changes.iter().filter(|c| selected.contains(&c.id)) {
        let path = match &c.op {
            Op::SchDesignator { path, .. }
            | Op::SchSwapPins { path, .. }
            | Op::SchSwapGates { path, .. }
            | Op::SchFootprint { path, .. } => path,
            _ => continue,
        };
        let Some(f) = board
            .footprints
            .values()
            .find(|f| f.component.as_deref() == Some(path))
        else {
            continue;
        };
        let pads = f
            .pad_nets
            .iter()
            .map(|(p, n)| (p.clone(), board.net_name(*n).to_owned()))
            .collect();
        let rec = SyncedComponent {
            designator: f.reference.clone(),
            package: f.package.name.clone(),
            pads,
        };
        board.sync.components.insert(path.clone(), rec);
    }
    *design = next;
    *lock = next_lock;
    Ok(applied)
}

fn back_designator(
    design: &mut Design,
    lock: &mut LockSection,
    path: &str,
    sheet: &str,
    to: &str,
) -> Result<(), String> {
    let p = NamePath::parse(path);
    let repeated = p.0.iter().any(|s| s.ends_with(']'));
    let decl_name = p.leaf().to_owned();
    let auto = repeated
        || design
            .sheets
            .get(sheet)
            .and_then(|s| s.components.iter().find(|d| d.name == decl_name))
            .is_none_or(|d| d.name.contains('?'));
    if auto {
        // Auto-annotated (`R?`) and repeated-channel designators live in the lock.
        let e = lock
            .entries
            .get_mut(&p)
            .ok_or_else(|| format!("{path} is not in the lock"))?;
        e.designator = Some(to.to_owned());
        return Ok(());
    }
    // An explicitly named component: rename it, keeping its identity (`renamed_from`).
    let s = design
        .sheets
        .get_mut(sheet)
        .ok_or_else(|| format!("no sheet {sheet}"))?;
    let d = s
        .components
        .iter_mut()
        .find(|d| d.name == decl_name)
        .ok_or_else(|| format!("{path} is not declared"))?;
    let old = std::mem::replace(&mut d.name, to.to_owned());
    d.renamed_from = Some(old.clone());
    for conn in s.connections.iter_mut() {
        for e in conn.ends.iter_mut() {
            if let Endpoint::Pin { component, .. } = e {
                if *component == old {
                    *component = to.to_owned();
                }
            }
        }
    }
    Ok(())
}

fn swap_endpoints(
    design: &mut Design,
    sheet: &str,
    component: &str,
    map: impl Fn(&str) -> Option<String>,
) -> Result<(), String> {
    let s = design
        .sheets
        .get_mut(sheet)
        .ok_or_else(|| format!("no sheet {sheet}"))?;
    let mut hits = 0;
    for conn in s.connections.iter_mut() {
        for e in conn.ends.iter_mut() {
            if let Endpoint::Pin { component: c, pin } = e {
                if c == component {
                    if let Some(n) = map(pin) {
                        *pin = n;
                        hits += 1;
                    }
                }
            }
        }
    }
    if hits == 0 {
        return Err(format!("no connections to swap on {component}"));
    }
    Ok(())
}

/// Records the whole schematic as synced (the state after a full ECO), for boards made before
/// the sync record existed.
pub fn record_sync(board: &mut Board, s: &Schematic) {
    let o = ComparatorOptions::default();
    let links = link(s, board, &o);
    for (ci, fid) in &links.by_comp {
        let c = &s.netlist.components[*ci];
        let f = &board.footprints[fid];
        board.sync.components.insert(
            c.path.to_string(),
            SyncedComponent {
                designator: f.reference.clone(),
                package: f.package.name.clone(),
                pads: f
                    .pad_nets
                    .iter()
                    .map(|(p, n)| (p.clone(), board.net_name(*n).to_owned()))
                    .collect(),
            },
        );
    }
}

/* ------------------------------------------------------------------------------------------------
 * Report
 * --------------------------------------------------------------------------------------------- */

/// A plain-text ECO report (Altium's Report Changes): every line grouped by type, with status.
pub fn report(plan: &EcoPlan, checks: &BTreeMap<String, Check>, design: &str) -> String {
    let mut out = String::new();
    let title = match plan.direction {
        Direction::Forward => "Update board",
        Direction::Back => "Update schematic",
    };
    out.push_str(&format!(
        "{title} (ECO) for {design}: {} changes\n",
        plan.changes.len()
    ));
    let mut group = None;
    for c in &plan.changes {
        if group != Some(c.kind.group()) {
            group = Some(c.kind.group());
            let n = plan
                .changes
                .iter()
                .filter(|x| x.kind.group() == c.kind.group())
                .count();
            out.push_str(&format!("\n{} ({n})\n", c.kind.group().label()));
        }
        let status = checks
            .get(&c.id)
            .map(|k| k.status.as_str())
            .unwrap_or("not selected");
        let ft = match (&c.from, &c.to) {
            (Some(f), Some(t)) => format!("{f} → {t}"),
            (None, Some(t)) => t.clone(),
            (Some(f), None) => format!("{f} → removed"),
            (None, None) => String::new(),
        };
        out.push_str(&format!(
            "  [{status}] {} {}: {ft}  ({}",
            c.kind.label(),
            c.item,
            c.cause.sheet
        ));
        if let Some(src) = &c.cause.source {
            out.push_str(&format!(", {src}"));
        }
        out.push_str(")\n");
        if let Some(r) = checks.get(&c.id).and_then(|k| k.reason()) {
            out.push_str(&format!("      {r}\n"));
        }
    }
    out
}
