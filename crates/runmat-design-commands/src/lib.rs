//! Design command descriptors: one schema for every CAD and PCB command the Desktop editors,
//! keymaps, command search and agent tools run. Neither `runmat-cad` nor `runmat-circuit` owns
//! it; each describes its commands with these types, and the Desktop loads the merged catalog.
//!
//! Each [`CommandDescriptor`] carries its id, label, icon key, typed input schema, an
//! enablement predicate (with the reason shown when it fails) and the model edit it produces.
//! [`EditorRows`] lay the commands out per editor: domain tabs, groups, `▾` menus with preset
//! values, the view cluster and the S box. [`ItemSchema`]s give the editable properties of a
//! selected item (feature, body, occurrence, mate, component, pad…) for the Properties panel.
//! [`common`] holds the commands every design editor shares (rebuild, measure, fit, section…).
//!
//! The catalog serializes to a versioned JSON file (`design-commands.json`) that the Desktop
//! checks in and loads; nothing in TypeScript restates a schema. The merged CAD and circuit
//! catalog is written and checked by this crate's `design_catalog` test.

mod build;
mod common;
mod reference;
mod withdraw;

pub use build::*;
pub use common::common;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A physical dimension on the wire: exponents over the eight SI-style bases the unit authority
/// uses (length, mass, time, current, temperature, amount, luminous intensity, information) plus
/// a separate angle power, the layout of `runmat-types::physical::PhysicalDimension`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Dimension {
    pub base: [i8; 8],
    pub angle: i8,
}

impl Dimension {
    pub const DIMENSIONLESS: Dimension = Dimension {
        base: [0; 8],
        angle: 0,
    };
    pub const LENGTH: Dimension = Dimension::base_unit(0);
    pub const MASS: Dimension = Dimension::base_unit(1);
    pub const TIME: Dimension = Dimension::base_unit(2);
    pub const CURRENT: Dimension = Dimension::base_unit(3);
    pub const TEMPERATURE: Dimension = Dimension::base_unit(4);
    pub const ANGLE: Dimension = Dimension {
        base: [0; 8],
        angle: 1,
    };
    pub const AREA: Dimension = Dimension {
        base: [2, 0, 0, 0, 0, 0, 0, 0],
        angle: 0,
    };

    const fn base_unit(index: usize) -> Dimension {
        let mut base = [0i8; 8];
        base[index] = 1;
        Dimension { base, angle: 0 }
    }

    pub fn multiply(self, other: Dimension) -> Dimension {
        self.combine(other, 1)
    }

    pub fn divide(self, other: Dimension) -> Dimension {
        self.combine(other, -1)
    }

    pub fn power(self, exponent: i8) -> Dimension {
        let mut base = self.base;
        base.iter_mut().for_each(|b| *b *= exponent);
        Dimension {
            base,
            angle: self.angle * exponent,
        }
    }

    fn combine(self, other: Dimension, sign: i8) -> Dimension {
        let mut base = self.base;
        for (b, o) in base.iter_mut().zip(other.base) {
            *b += sign * o;
        }
        Dimension {
            base,
            angle: self.angle + sign * other.angle,
        }
    }
}

/// Format name written into the JSON file.
pub const FORMAT: &str = "runmat.design-commands";
/// Bumped on any change to the JSON shape (not on new commands).
pub const FORMAT_VERSION: u32 = 2;

/// File kinds commands target: the extension without the dot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EditorKind {
    #[serde(rename = "part")]
    Part,
    #[serde(rename = "assembly")]
    Assembly,
    #[serde(rename = "drawing")]
    Drawing,
    #[serde(rename = "circuit")]
    Circuit,
    #[serde(rename = "circuit.layout")]
    CircuitLayout,
    /// The .fea study view: only its viewport toolbar (the 3D view cluster) comes from here.
    #[serde(rename = "fea")]
    Fea,
}

impl EditorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EditorKind::Part => "part",
            EditorKind::Assembly => "assembly",
            EditorKind::Drawing => "drawing",
            EditorKind::Circuit => "circuit",
            EditorKind::CircuitLayout => "circuit.layout",
            EditorKind::Fea => "fea",
        }
    }
}

/// Editing modes that replace an editor's rows (sketch mode replaces both rows of a part).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditMode {
    /// The editor's normal rows.
    Model,
    /// Editing a 2D sketch (part, or a layout sketch in an assembly).
    Sketch,
    /// Editing a 3D sketch.
    Sketch3d,
    /// Editing an occurrence's definition in the context of its assembly.
    InContext,
    /// Schematic view of a circuit.
    Schematic,
    /// Board view of a circuit.
    Board,
    /// 3D view of a circuit board.
    Board3d,
}

/// What a selection field accepts. Matches the Outline node kinds and canvas pick kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectKind {
    // Part topology and features.
    Body,
    Face,
    Edge,
    Vertex,
    Feature,
    Sketch,
    SketchEntity,
    SketchPoint,
    SketchRegion,
    Plane,
    Axis,
    Point,
    CoordinateSystem,
    MateConnector,
    Curve,
    Region,
    Input,
    // Assembly.
    Occurrence,
    Mate,
    Part,
    Hole,
    ExplodeStep,
    // Drawing.
    Sheet,
    View,
    Annotation,
    DrawingDimension,
    Table,
    // Circuit.
    Component,
    Pin,
    Net,
    NetClass,
    Port,
    SheetSymbol,
    Wire,
    /// One line of a pending schematic-to-board change (ECO).
    EcoChange,
    /// One line of a pending board-to-schematic change (back-annotation).
    EcoBackChange,
    Pad,
    Track,
    Via,
    Zone,
    Keepout,
    Room,
    Layer,
    /// A design rule (clearance, width, via…), in the circuit's Rules folder.
    Rule,
}

impl SelectKind {
    /// "face", "sketch entity": for reasons and help text.
    pub fn noun(self) -> &'static str {
        use SelectKind::*;
        match self {
            Body => "body",
            Face => "face",
            Edge => "edge",
            Vertex => "vertex",
            Feature => "feature",
            Sketch => "sketch",
            SketchEntity => "sketch entity",
            SketchPoint => "sketch point",
            SketchRegion => "sketch region",
            Plane => "plane",
            Axis => "axis",
            Point => "point",
            CoordinateSystem => "coordinate system",
            MateConnector => "mate connector",
            Curve => "curve",
            Region => "region",
            Input => "input",
            Occurrence => "component",
            Mate => "mate",
            Part => "part",
            Hole => "hole",
            ExplodeStep => "explode step",
            Sheet => "sheet",
            View => "view",
            Annotation => "annotation",
            DrawingDimension => "dimension",
            Table => "table",
            Component => "component",
            Pin => "pin",
            Net => "net",
            NetClass => "net class",
            Port => "port",
            SheetSymbol => "sheet symbol",
            Wire => "wire",
            EcoChange => "change",
            EcoBackChange => "change",
            Pad => "pad",
            Track => "track",
            Via => "via",
            Zone => "pour",
            Keepout => "keepout",
            Room => "room",
            Layer => "layer",
            Rule => "rule",
        }
    }
}

/// A named physical dimension with the unit the field shows by default. `dimension` is the
/// unit authority's exponent vector, so the Desktop and the compiler agree on what a value is.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuantitySpec {
    pub name: String,
    pub dimension: Dimension,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChoiceOption {
    pub value: String,
    pub label: String,
}

/// Where a text field's suggestions come from, filled by the editor from the open file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Suggest {
    /// The circuit's global nets (power and ground rails), with their pin counts.
    Rails,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChoiceDisplay {
    Select,
    Radio,
}

/// A field's type and constraints.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FieldKind {
    /// A physical value: the number in `unit`. `quantity` names a row of [`Catalog::quantities`].
    Quantity {
        quantity: String,
        unit: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<f64>,
    },
    /// A whole number (instance counts, sides).
    Count {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<i64>,
    },
    Choice {
        options: Vec<ChoiceOption>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<String>,
        display: ChoiceDisplay,
    },
    /// Picks from the tree or the canvas, as persistent ids.
    Selection {
        accepts: Vec<SelectKind>,
        multiple: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<u32>,
        /// Offer every item of the accepted kinds in the file as a checklist, all checked at
        /// the start (the ECO review accepted line by line), instead of picking.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        checklist: bool,
    },
    Boolean {
        default: bool,
    },
    Text {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        placeholder: Option<String>,
        /// Names the editor offers as the field is typed (a menu on focus); any other text is
        /// still accepted.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        suggest: Option<Suggest>,
    },
    /// Several values of one kind (hole positions, variable fillet radii, table rows).
    List {
        item: Box<FieldKind>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        min: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<u32>,
    },
}

/// Shows a field only while another field has one of the listed values (`true`/`false` for
/// booleans), or, with `filled`, while it has any value at all (a Bus typed in).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldCondition {
    pub field: String,
    pub any_of: Vec<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub filled: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Field {
    pub id: String,
    pub label: String,
    #[serde(flatten)]
    pub kind: FieldKind,
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub help: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible_when: Option<FieldCondition>,
    /// Canvas manipulators that drive this field (extrude arrow, radius drag, triad).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manipulator: Option<Manipulator>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Manipulator {
    /// A linear arrow along a direction (extrude depth, offset).
    Arrow,
    /// A radius or diameter drag handle.
    Radius,
    /// A rotation ring (revolve angle, draft angle).
    Angle,
    /// A move/rotate triad.
    Triad,
    /// A point picked or dragged on the canvas.
    Point,
}

/// Facts about the open file the enablement predicates test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Facet {
    /// The part has at least one solid body.
    Bodies,
    /// The part has at least one sketch.
    Sketches,
    /// The part or assembly has more than one configuration.
    Configurations,
    /// The part is sheet metal (has a base flange or was converted).
    SheetMetal,
    /// The part has structural members.
    Weldments,
    /// The assembly has components.
    Occurrences,
    /// The assembly has mates.
    Mates,
    /// The assembly has exploded views.
    ExplodedViews,
    /// The drawing has at least one view.
    Views,
    /// The circuit has a board.
    Board,
    /// The circuit board has an outline.
    BoardOutline,
    /// There are schematic changes the board hasn't taken yet.
    PendingEco,
    /// There are board changes the schematic hasn't taken yet (back-annotation).
    PendingBackAnnotation,
    /// The file compiled without errors.
    Compiled,
    /// The runtime that rebuilds and previews is ready.
    Runtime,
}

impl Facet {
    fn reason(self) -> &'static str {
        match self {
            Facet::Bodies => "Create a body first",
            Facet::Sketches => "Create a sketch first",
            Facet::Configurations => "This file has one configuration",
            Facet::SheetMetal => "Start with a base flange or convert the part to sheet metal",
            Facet::Weldments => "Add a structural member first",
            Facet::Occurrences => "Insert a component first",
            Facet::Mates => "Add a mate first",
            Facet::ExplodedViews => "Create an exploded view first",
            Facet::Views => "Add a view first",
            Facet::Board => "This circuit has no board yet",
            Facet::BoardOutline => "Draw the board outline first",
            Facet::PendingEco => "The board is up to date with the schematic",
            Facet::PendingBackAnnotation => "The schematic is up to date with the board",
            Facet::Compiled => "Fix the errors in this file first",
            Facet::Runtime => "Waiting for the runtime",
        }
    }
}

/// When a command can run. Every leaf carries the sentence shown when it fails.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "when", rename_all = "snake_case")]
pub enum Predicate {
    Always,
    /// The editor is in one of these modes.
    Mode {
        modes: Vec<EditMode>,
        reason: String,
    },
    /// The selection holds between `min` and `max` items, each of one of `kinds`.
    Selection {
        kinds: Vec<SelectKind>,
        min: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max: Option<u32>,
        reason: String,
    },
    /// The file has this facet.
    Has {
        facet: Facet,
        reason: String,
    },
    All {
        all: Vec<Predicate>,
    },
    Any {
        any: Vec<Predicate>,
    },
}

/// The part of the document an edit changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Section {
    Inputs,
    Features,
    Sketch,
    Regions,
    Publishes,
    Materials,
    Appearances,
    Properties,
    Configurations,
    Settings,
    Pmi,
    Occurrences,
    Mates,
    Fasteners,
    Explode,
    Motion,
    Options,
    Sheets,
    Views,
    Annotations,
    Tables,
    Components,
    Wires,
    Nets,
    Directives,
    Rules,
    Stackup,
    Placement,
    Copper,
    Outline,
    Studies,
    /// No document change (analysis, display, checks, export).
    None,
}

/// What a command does to the model when confirmed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditOp {
    /// Append a new item (feature, entity, mate, view, wire…).
    Add,
    /// Change the selected item's definition.
    Edit,
    /// Set one property on the selected items.
    Set,
    Remove,
    Rename,
    Reorder,
    Suppress,
    Unsuppress,
    /// Move the rollback bar.
    Rollback,
    /// Enter an editing mode (sketch, in-context).
    EnterMode,
    /// Leave the mode, keeping (`Exit`) or discarding changes.
    ExitMode,
    DiscardMode,
    /// Compute and show something; no document change.
    Analyze,
    /// Run a check pack (ERC, DRC, drawing standards); results go to the status dot.
    Check,
    Export,
    Import,
    /// Create another file (a `.fea` study, split bodies to parts, a drawing from a part).
    CreateFile,
    /// Display only (view orientation, section, visibility).
    View,
    /// Synchronise two views of a design (schematic to board).
    Sync,
    /// Ask the agent about the selection (explain, fix, change); its answer is a reviewed change set.
    Agent,
}

/// What distinguishes the edited item within its kind: the model's serde tag for a sub-kind
/// (`{"feature": "surface", "kind": "fill"}` is a fill surface) or a mode field of the item
/// (`kind: union` on a boolean).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Variant {
    pub tag: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelEdit {
    pub op: EditOp,
    pub section: Section,
    /// The model's tag for the item added or edited: a `FeatureKind` tag (`extrude`), a sketch
    /// entity, a mate type, a view kind, a circuit item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variant: Option<Variant>,
    /// For export and import: the file format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// Where a command sits in an editor's rows (filled in from the rows).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Placement {
    pub editor: EditorKind,
    pub mode: EditMode,
    pub tab: String,
    pub group: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommandDescriptor {
    /// Dotted and stable: `part.extrude`, `sketch.line`, `board.route`.
    pub id: String,
    pub label: String,
    /// One sentence for search results and tooltips.
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
    pub editors: Vec<EditorKind>,
    /// A product icon key; the Desktop maps keys to its icon set.
    pub icon: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<Field>,
    pub enabled: Predicate,
    pub edit: ModelEdit,
    /// The command previews live while its inputs change.
    pub preview: bool,
    /// Outline node kinds whose right-click menu offers this command.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub context: Vec<SelectKind>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placements: Vec<Placement>,
}

/// One entry of a row: a button, or a `▾` menu of commands (with preset values).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RowItem {
    Command {
        command: String,
    },
    Menu {
        id: String,
        label: String,
        icon: String,
        entries: Vec<MenuEntry>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MenuEntry {
    pub command: String,
    /// The entry's own label when it presets values (`Extruded cut`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Field values the entry starts the command with.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub values: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RowGroup {
    pub id: String,
    pub label: String,
    pub items: Vec<RowItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DomainTab {
    pub id: String,
    pub label: String,
    pub groups: Vec<RowGroup>,
}

/// An editor's rows in one mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditorRows {
    pub editor: EditorKind,
    pub mode: EditMode,
    pub tabs: Vec<DomainTab>,
    pub default_tab: String,
    /// Row 1, right side.
    pub view_cluster: Vec<RowItem>,
    /// The bottom strip: document context (sheets, standard and layer; configuration; options,
    /// representation and working set; variant). Replaces the Command Window in design editors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub document_bar: Vec<RowItem>,
    /// Commands the S box offers first.
    pub shortcut_box: Vec<String>,
    /// Row 1 status text the mode shows (sketch definition state and DOF).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mode_actions: Vec<String>,
}

/// The editable properties of a selected item in the Properties panel's selection mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemSchema {
    pub kind: SelectKind,
    pub editors: Vec<EditorKind>,
    pub title: String,
    pub sections: Vec<ItemSection>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemSection {
    pub title: String,
    pub fields: Vec<ItemField>,
}

/// One editable property: its field, and the edit a change produces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemField {
    #[serde(flatten)]
    pub field: Field,
    pub edit: ModelEdit,
    /// Shown but not editable here (computed values, values from rules).
    pub read_only: bool,
}

/// The whole catalog, as written to `design-commands.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub format: String,
    pub format_version: u32,
    /// Every quantity a field uses, with the unit authority's dimension.
    pub quantities: Vec<QuantitySpec>,
    pub commands: Vec<CommandDescriptor>,
    pub rows: Vec<EditorRows>,
    pub items: Vec<ItemSchema>,
    /// Right-click menus per item kind: the single source of what each item offers.
    pub context_menus: Vec<ContextMenu>,
}

/// The right-click menu of one item kind in some editors (tree rows and canvas picks alike):
/// a row of quick icon actions, grouped commands (separated in the menu; `▸` submenus are
/// [`RowItem::Menu`]s) and the Agent submenu. Commands that don't apply to the selection stay in
/// the menu, disabled, with their predicate's reason.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextMenu {
    pub kind: SelectKind,
    pub editors: Vec<EditorKind>,
    /// Four to six commands drawn as icons across the top of the menu.
    pub quick: Vec<String>,
    pub groups: Vec<Vec<RowItem>>,
    /// The Agent submenu's commands (explain, fix, change).
    pub agent: Vec<String>,
}

impl ContextMenu {
    /// Every command the menu names, in menu order, once each.
    pub fn command_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = Vec::new();
        let groups = self.groups.iter().flatten().flat_map(|item| item.command_ids());
        for id in self
            .quick
            .iter()
            .map(String::as_str)
            .chain(groups)
            .chain(self.agent.iter().map(String::as_str))
        {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }
}

impl Catalog {
    pub fn empty() -> Self {
        Catalog {
            format: FORMAT.to_owned(),
            format_version: FORMAT_VERSION,
            quantities: Vec::new(),
            commands: Vec::new(),
            rows: Vec::new(),
            items: Vec::new(),
            context_menus: Vec::new(),
        }
    }

    pub fn command(&self, id: &str) -> Option<&CommandDescriptor> {
        self.commands.iter().find(|c| c.id == id)
    }

    /// Adds commands; an id that is already present is a programming error.
    pub fn add_commands(&mut self, commands: impl IntoIterator<Item = CommandDescriptor>) {
        for command in commands {
            assert!(
                self.command(&command.id).is_none(),
                "duplicate command id {}",
                command.id
            );
            self.commands.push(command);
        }
    }

    /// Merges another catalog into this one. A command both catalogs carry (the shared commands
    /// of [`common`]) must be the same descriptor in each; it is kept once. Placements and the
    /// kinds it is offered on come from each catalog's rows and menus, so those are combined.
    pub fn merge(&mut self, other: Catalog) {
        for mut command in other.commands {
            if let Some(existing) = self.commands.iter_mut().find(|c| c.id == command.id) {
                let mut mine = existing.clone();
                mine.placements.clear();
                mine.context.clear();
                let kinds = std::mem::take(&mut command.context);
                command.placements.clear();
                assert!(
                    mine == command,
                    "command {} differs between the merged catalogs",
                    command.id
                );
                for kind in kinds {
                    if !existing.context.contains(&kind) {
                        existing.context.push(kind);
                    }
                }
                continue;
            }
            self.commands.push(command);
        }
        self.rows.extend(other.rows);
        self.items.extend(other.items);
        self.context_menus.extend(other.context_menus);
        self.finish();
    }

    /// Fills each command's placements from the rows and sorts everything into a stable order.
    pub fn finish(&mut self) {
        let mut placements: BTreeMap<String, Vec<Placement>> = BTreeMap::new();
        for rows in &self.rows {
            for tab in &rows.tabs {
                for group in &tab.groups {
                    for item in &group.items {
                        for id in item.command_ids() {
                            let p = Placement {
                                editor: rows.editor,
                                mode: rows.mode,
                                tab: tab.id.clone(),
                                group: group.id.clone(),
                            };
                            let list = placements.entry(id.to_owned()).or_default();
                            if !list.contains(&p) {
                                list.push(p);
                            }
                        }
                    }
                }
            }
        }
        // A command a menu names is offered on that menu's kind: `context` follows the menus.
        let mut offered: BTreeMap<String, Vec<SelectKind>> = BTreeMap::new();
        for menu in &self.context_menus {
            for id in menu.command_ids() {
                let kinds = offered.entry(id.to_owned()).or_default();
                if !kinds.contains(&menu.kind) {
                    kinds.push(menu.kind);
                }
            }
        }
        for command in &mut self.commands {
            let mut list = placements.remove(&command.id).unwrap_or_default();
            list.sort();
            command.placements = list;
            for kind in offered.remove(&command.id).unwrap_or_default() {
                if !command.context.contains(&kind) {
                    command.context.push(kind);
                }
            }
        }
        self.commands.sort_by(|a, b| a.id.cmp(&b.id));
        let mut names: Vec<String> = Vec::new();
        let mut collect = |kind: &FieldKind| {
            let mut kind = kind;
            while let FieldKind::List { item, .. } = kind {
                kind = item;
            }
            if let FieldKind::Quantity { quantity, .. } = kind {
                if !names.contains(quantity) {
                    names.push(quantity.clone());
                }
            }
        };
        for command in &self.commands {
            command.inputs.iter().for_each(|f| collect(&f.kind));
        }
        for schema in &self.items {
            for section in &schema.sections {
                section.fields.iter().for_each(|f| collect(&f.field.kind));
            }
        }
        names.sort();
        self.quantities = names.iter().map(|n| quantity_spec(n)).collect();
        self.rows.sort_by_key(|r| (r.editor, r.mode));
        self.items
            .sort_by(|a, b| (a.kind, &a.editors).cmp(&(b.kind, &b.editors)));
        self.context_menus
            .sort_by(|a, b| (a.kind, &a.editors).cmp(&(b.kind, &b.editors)));
    }

    /// Problems with the catalog: rows naming unknown commands, unknown preset fields, labels
    /// breaking the copy rules. Empty when the catalog is sound.
    pub fn lint(&self) -> Vec<String> {
        let mut problems = Vec::new();
        for rows in &self.rows {
            let mut items: Vec<&RowItem> = rows
                .view_cluster
                .iter()
                .chain(rows.document_bar.iter())
                .collect();
            for tab in &rows.tabs {
                for group in &tab.groups {
                    items.extend(group.items.iter());
                }
            }
            for item in items {
                if let RowItem::Menu { entries, id, .. } = item {
                    for entry in entries {
                        match self.command(&entry.command) {
                            None => {
                                problems.push(format!("menu {id} names unknown {}", entry.command))
                            }
                            Some(command) => {
                                for key in entry.values.keys() {
                                    if !command.inputs.iter().any(|f| &f.id == key) {
                                        problems.push(format!(
                                            "menu {id} presets unknown field {key} of {}",
                                            command.id
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                for id in item.command_ids() {
                    let Some(command) = self.command(id) else {
                        problems.push(format!(
                            "{} rows name unknown command {id}",
                            rows.editor.as_str()
                        ));
                        continue;
                    };
                    if !command.editors.contains(&rows.editor) {
                        problems.push(format!(
                            "{id} is in the {} rows but not declared for that editor",
                            rows.editor.as_str()
                        ));
                    }
                }
            }
            for id in rows.shortcut_box.iter().chain(rows.mode_actions.iter()) {
                if self.command(id).is_none() {
                    problems.push(format!(
                        "{} names unknown command {id}",
                        rows.editor.as_str()
                    ));
                }
            }
            // A view-only row (the .fea study view's toolbar) has no tabs.
            if !rows.tabs.is_empty() && !rows.tabs.iter().any(|t| t.id == rows.default_tab) {
                problems.push(format!(
                    "{} default tab {} is missing",
                    rows.editor.as_str(),
                    rows.default_tab
                ));
            }
        }
        for command in &self.commands {
            for text in [&command.label, &command.description] {
                if let Some(problem) = copy_problem(text) {
                    problems.push(format!("{}: {problem}", command.id));
                }
            }
            for field in &command.inputs {
                if let Some(condition) = &field.visible_when {
                    if !command.inputs.iter().any(|f| f.id == condition.field) {
                        problems.push(format!(
                            "{}.{} depends on unknown field {}",
                            command.id, field.id, condition.field
                        ));
                    }
                }
                if let Some(problem) = copy_problem(&field.label) {
                    problems.push(format!("{}.{}: {problem}", command.id, field.id));
                }
            }
            if command.edit.op == EditOp::Agent && !command.id.starts_with("agent.") {
                problems.push(format!("{} asks the agent but is not an agent.* command", command.id));
            }
            let ids: Vec<&str> = command.inputs.iter().map(|f| f.id.as_str()).collect();
            let mut unique = ids.clone();
            unique.sort();
            unique.dedup();
            if unique.len() != ids.len() {
                problems.push(format!("{} has duplicate field ids", command.id));
            }
        }
        for menu in &self.context_menus {
            let name = format!("{} menu", menu.kind.noun());
            if !(4..=6).contains(&menu.quick.len()) {
                problems.push(format!("{name} has {} quick actions, not 4 to 6", menu.quick.len()));
            }
            if menu.agent.is_empty() {
                problems.push(format!("{name} has no Agent submenu"));
            }
            for item in menu.groups.iter().flatten() {
                if let RowItem::Menu { entries, id, .. } = item {
                    for entry in entries {
                        if let Some(command) = self.command(&entry.command) {
                            for key in entry.values.keys() {
                                if !command.inputs.iter().any(|f| &f.id == key) {
                                    problems.push(format!("{name} submenu {id} presets unknown field {key} of {}", command.id));
                                }
                            }
                        }
                    }
                }
            }
            for id in menu.command_ids() {
                let Some(command) = self.command(id) else {
                    problems.push(format!("{name} names unknown command {id}"));
                    continue;
                };
                for editor in &menu.editors {
                    if !command.editors.contains(editor) {
                        problems.push(format!("{name} offers {id} in {}, where it is not declared", editor.as_str()));
                    }
                }
            }
            if menu.agent.iter().any(|id| self.command(id).is_some_and(|c| c.edit.op != EditOp::Agent)) {
                problems.push(format!("{name} Agent submenu names a command that does not ask the agent"));
            }
        }
        let mut keys: Vec<(SelectKind, &EditorKind)> = self
            .context_menus
            .iter()
            .flat_map(|m| m.editors.iter().map(move |e| (m.kind, e)))
            .collect();
        let before = keys.len();
        keys.sort();
        keys.dedup();
        if keys.len() != before {
            problems.push("two context menus share an item kind and editor".to_owned());
        }
        problems
    }

    /// The menu for an item of `kind` in `editor`, if one is declared.
    pub fn context_menu(&self, kind: SelectKind, editor: EditorKind) -> Option<&ContextMenu> {
        self.context_menus
            .iter()
            .find(|m| m.kind == kind && m.editors.contains(&editor))
    }

    pub fn to_json(&self) -> String {
        let mut json = serde_json::to_string_pretty(self).expect("catalog serializes");
        json.push('\n');
        json
    }
}

impl RowItem {
    pub fn command_ids(&self) -> Vec<&str> {
        match self {
            RowItem::Command { command } => vec![command.as_str()],
            RowItem::Menu { entries, .. } => {
                let mut ids: Vec<&str> = Vec::new();
                for entry in entries {
                    if !ids.contains(&entry.command.as_str()) {
                        ids.push(entry.command.as_str());
                    }
                }
                ids
            }
        }
    }
}

/// The copy rules for UI strings: US spelling for the words we use, no em dashes or arrows.
fn copy_problem(text: &str) -> Option<String> {
    if text.contains('\u{2014}') {
        return Some("em dash in UI text".to_owned());
    }
    if text.chars().any(|c| ('\u{2190}'..='\u{21ff}').contains(&c)) {
        return Some("arrow character in UI text".to_owned());
    }
    for british in [
        "colour",
        "centre",
        "behaviour",
        "optimis",
        "analys",
        "metre",
        "favour",
        "modell",
        "labell",
        "cancell",
    ] {
        if text.to_lowercase().contains(british) && !text.to_lowercase().contains("analysis") {
            return Some(format!("British spelling ({british})"));
        }
    }
    None
}
