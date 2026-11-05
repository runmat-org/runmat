//! The `.assembly` document: one level of product structure.
//!
//! An assembly lists its direct children only (one level per file). Each child is an
//! [`Occurrence`] of a part output, a sub-assembly or a circuit, with a configuration selection,
//! an optional `when` condition (the 150% structure), a placement, and typed overrides on its own
//! descendants. Mate connectors are frames on occurrences; mates join connectors and are solved
//! through `runmat-constraint`. Patterns and mirrors repeat occurrences with semantic instance
//! keys. Fastener sets are rules ([`crate::assembly::fasteners`]). Nothing here parses text: the
//! syntax layer (M1) builds these values from the lossless tree.

use crate::assembly::explode::ExplodedView;
use crate::assembly::fasteners::FastenerSet;
use crate::assembly::options::{Condition, OptionsBlock};
use crate::ids::{ConfigurationName, FilePath, InputName, MaterialRef, PartName, PublishName};
use crate::model::items::{Link, Property};
use crate::model::VersionHeader;
use crate::refs::{InstanceKey, XrefContext};
use crate::value::{InputDecl, LetDecl, ValueExpr};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefinitionKind {
    Part,
    Assembly,
    Circuit,
}

/// What an occurrence instantiates: `"flange.part#gasket"`, `"wing/box.assembly"`,
/// `"systems/fcc.circuit"`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DefinitionRef {
    pub kind: DefinitionKind,
    pub file: FilePath,
    /// The part output within a multi-part `.part` file. `None`: the file's single output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<PartName>,
}

impl DefinitionRef {
    pub fn part(file: &str, part: Option<&str>) -> Self {
        Self {
            kind: DefinitionKind::Part,
            file: FilePath::new(file),
            part: part.map(PartName::new),
        }
    }
    pub fn assembly(file: &str) -> Self {
        Self {
            kind: DefinitionKind::Assembly,
            file: FilePath::new(file),
            part: None,
        }
    }
    pub fn circuit(file: &str) -> Self {
        Self {
            kind: DefinitionKind::Circuit,
            file: FilePath::new(file),
            part: None,
        }
    }
}

impl fmt::Display for DefinitionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.part {
            Some(p) => write!(f, "{}#{}", self.file, p),
            None => write!(f, "{}", self.file),
        }
    }
}

/// `use <kind> <name> from "<path>"`, live or pinned.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OccurrenceSource {
    pub definition: DefinitionRef,
    #[serde(default = "live")]
    pub context: XrefContext,
}

fn live() -> XrefContext {
    XrefContext::Live
}

impl OccurrenceSource {
    pub fn live(definition: DefinitionRef) -> Self {
        Self {
            definition,
            context: XrefContext::Live,
        }
    }
}

/// Which configuration of the definition an occurrence uses. Typed: names are checked against
/// the definition's configurations and input values against its declared input kinds.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "select")]
pub enum ConfigurationSelection {
    #[default]
    Default,
    Named {
        name: ConfigurationName,
    },
    /// Input values on top of a base configuration (an anonymous configuration). Values may
    /// read the parent assembly's inputs and lets.
    Inputs {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base: Option<ConfigurationName>,
        values: BTreeMap<InputName, ValueExpr>,
    },
    /// The first case whose condition holds; the validator proves the cases are exclusive.
    Switch {
        cases: Vec<ConfigurationCase>,
        otherwise: Box<ConfigurationSelection>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationCase {
    pub when: Condition,
    pub select: ConfigurationSelection,
}

/// A rotation written in the source.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "rotation")]
pub enum RotationExpr {
    #[default]
    Identity,
    AxisAngle {
        axis: [f64; 3],
        angle: ValueExpr,
    },
    /// Intrinsic Z-Y-X (yaw, pitch, roll).
    Euler {
        yaw: ValueExpr,
        pitch: ValueExpr,
        roll: ValueExpr,
    },
    /// Local Z along `z`, local X towards `x` (orthogonalised).
    Basis {
        x: [f64; 3],
        z: [f64; 3],
    },
}

/// A rigid offset: translation and rotation (applied rotation first).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OffsetExpr {
    pub translation: [ValueExpr; 3],
    #[serde(default)]
    pub rotation: RotationExpr,
}

impl OffsetExpr {
    pub fn translation(t: [ValueExpr; 3]) -> Self {
        Self {
            translation: t,
            rotation: RotationExpr::Identity,
        }
    }
}

/// A frame inside a definition (a mate connector's anchor).
// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "anchor")]
pub enum AnchorRef {
    /// The definition's origin frame.
    Origin,
    /// A frame, plane, axis or point the definition publishes (its interface).
    Published { name: PublishName },
    /// An explicit frame in definition coordinates.
    Explicit { offset: OffsetExpr },
}

/// A frame in this assembly's coordinates.
// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "frame")]
pub enum FrameRef {
    Origin,
    /// A mate connector of this assembly.
    Connector {
        connector: String,
    },
    /// A frame published by another file (a skeleton or interface `.part`), live or pinned.
    Interface {
        file: FilePath,
        name: PublishName,
        #[serde(default = "live")]
        context: XrefContext,
    },
    Explicit {
        offset: OffsetExpr,
    },
}

/// How an occurrence is placed in its parent.
// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "placement")]
pub enum Placement {
    #[default]
    Origin,
    Offset {
        offset: OffsetExpr,
    },
    /// Put the occurrence's anchor `own` on the frame `to` (then `offset` in `to`'s axes).
    /// `flip` turns the occurrence half a turn about the target's X axis (face to face).
    Attach {
        own: AnchorRef,
        to: FrameRef,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<OffsetExpr>,
        #[serde(default)]
        flip: bool,
    },
}

/// One segment of an occurrence path: an authored name, or a pattern or mirror instance of one.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PathSegment {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<InstanceRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct InstanceRef {
    /// The pattern, mirror or fastener set that made the instance.
    pub by: String,
    pub key: InstanceKey,
}

impl PathSegment {
    pub fn named(name: &str) -> Self {
        Self {
            name: name.into(),
            instance: None,
        }
    }
    pub fn instance(name: &str, by: &str, key: InstanceKey) -> Self {
        Self {
            name: name.into(),
            instance: Some(InstanceRef { by: by.into(), key }),
        }
    }
}

impl fmt::Display for PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.instance {
            None => f.write_str(&self.name),
            Some(i) => write!(f, "{}@{}[{}]", self.name, i.by, i.key),
        }
    }
}

/// A path relative to an assembly: `box/rib@ribs[3]/clip`.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RelPath(pub Vec<PathSegment>);

impl RelPath {
    pub fn new(names: &[&str]) -> Self {
        RelPath(names.iter().map(|n| PathSegment::named(n)).collect())
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for RelPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, s) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str("/")?;
            }
            write!(f, "{}", s)?;
        }
        Ok(())
    }
}

/// Named representations. Each feeds lightweight display, analysis idealisation, or both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Representation {
    /// The exact B-rep.
    Full,
    /// A bounding or shrink-wrap solid (space allocation, far context).
    Envelope,
    /// Published geometry only: frames, faces and envelopes other teams consume.
    Interface,
    /// Tessellation only.
    Display,
    /// Small features removed (fillets, small holes) for analysis and light display.
    Defeatured,
    /// Thin-wall idealisation for shell analysis.
    Midsurface,
    /// Line idealisation for beam analysis.
    Beam,
}

impl Representation {
    pub const ALL: [Representation; 7] = [
        Representation::Full,
        Representation::Envelope,
        Representation::Interface,
        Representation::Display,
        Representation::Defeatured,
        Representation::Midsurface,
        Representation::Beam,
    ];
}

/// A typed override on a descendant occurrence. There is no free attribute override.
// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "set")]
pub enum OverrideKind {
    Configuration {
        select: ConfigurationSelection,
    },
    /// One input value on top of the target's configuration selection.
    Input {
        name: InputName,
        value: ValueExpr,
    },
    Suppress {
        suppressed: bool,
    },
    /// Re-place the target. Allowed only where the target is positioned by its placement
    /// (not by mates) or its parent is flexible.
    Position {
        placement: Placement,
    },
    Representation {
        representation: Representation,
    },
    Material {
        material: MaterialRef,
    },
    /// An occurrence-scope property.
    Property {
        property: Property,
    },
    /// Solve a sub-assembly flexibly (or rigidly) in this context.
    Flexible {
        flexible: bool,
    },
    /// Use another definition (or another revision) for the target in this context: the
    /// target keeps its name, placement and mates; its geometry, configurations and
    /// published items come from `source`.
    Replace {
        source: OccurrenceSource,
    },
}

impl OverrideKind {
    pub fn slot(&self) -> String {
        match self {
            OverrideKind::Configuration { .. } => "configuration".into(),
            OverrideKind::Input { name, .. } => format!("input.{}", name),
            OverrideKind::Suppress { .. } => "suppress".into(),
            OverrideKind::Position { .. } => "position".into(),
            OverrideKind::Representation { .. } => "representation".into(),
            OverrideKind::Material { .. } => "material".into(),
            OverrideKind::Property { property } => format!("property.{}", property.name),
            OverrideKind::Flexible { .. } => "flexible".into(),
            OverrideKind::Replace { .. } => "replace".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Override {
    /// The descendant, relative to the occurrence that carries the override.
    pub target: RelPath,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Condition>,
    pub set: OverrideKind,
}

/// `use part bracket from "./bracket.part" as left { … }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Occurrence {
    pub name: String,
    pub source: OccurrenceSource,
    #[serde(default)]
    pub configuration: ConfigurationSelection,
    /// The 150% condition. Alternatives share a name and have exclusive conditions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Condition>,
    #[serde(default)]
    pub placement: Placement,
    /// Fixed: the placement is final. Floating: the placement is the start and mates move it.
    #[serde(default = "yes")]
    pub fixed: bool,
    /// A sub-assembly whose own occurrences join the parent's mate solve.
    #[serde(default)]
    pub flexible: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation: Option<Representation>,
    #[serde(default)]
    pub suppressed: bool,
    /// Some alternative of this name must be active in every legal configuration.
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub overrides: Vec<Override>,
    #[serde(default)]
    pub properties: Vec<Property>,
    /// A typed material override of the definition's material.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialRef>,
}

fn yes() -> bool {
    true
}

impl Occurrence {
    pub fn new(name: &str, definition: DefinitionRef) -> Self {
        Self {
            name: name.into(),
            source: OccurrenceSource::live(definition),
            configuration: ConfigurationSelection::Default,
            when: None,
            placement: Placement::Origin,
            fixed: true,
            flexible: false,
            representation: None,
            suppressed: false,
            required: false,
            overrides: Vec::new(),
            properties: Vec::new(),
            material: None,
        }
    }
}

/// A frame on an occurrence (or on the assembly itself) that mates join.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MateConnector {
    pub name: String,
    /// The occurrence the connector is fixed in, relative to this assembly (through rigid or
    /// flexible sub-assemblies). Empty: the assembly's own frame (ground).
    pub occurrence: RelPath,
    pub anchor: AnchorRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<OffsetExpr>,
    /// Turn half a turn about the anchor's X axis (reverses Z).
    #[serde(default)]
    pub flip: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum TargetExpr {
    Exact { value: ValueExpr },
    Range { min: ValueExpr, max: ValueExpr },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMeasure {
    PointPoint,
    PlanePoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AngleMeasure {
    AboutZ,
    BetweenZ,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "tangent")]
pub enum TangentSpec {
    PlaneSphere { radius: ValueExpr },
    PlaneCylinder { radius: ValueExpr },
}

/// Mate kinds. Joint mates take optional limits (ranges) or drive values (exact).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mate")]
pub enum MateKind {
    Fasten,
    Revolute {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        angle: Option<TargetExpr>,
    },
    Slider {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        distance: Option<TargetExpr>,
    },
    Cylindrical {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        angle: Option<TargetExpr>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        distance: Option<TargetExpr>,
    },
    Planar,
    Ball,
    Parallel,
    Tangent {
        kind: TangentSpec,
    },
    Distance {
        measure: DistanceMeasure,
        target: TargetExpr,
    },
    Angle {
        measure: AngleMeasure,
        target: TargetExpr,
    },
}

impl MateKind {
    pub fn name(&self) -> &'static str {
        match self {
            MateKind::Fasten => "fasten",
            MateKind::Revolute { .. } => "revolute",
            MateKind::Slider { .. } => "slider",
            MateKind::Cylindrical { .. } => "cylindrical",
            MateKind::Planar => "planar",
            MateKind::Ball => "ball",
            MateKind::Parallel => "parallel",
            MateKind::Tangent { .. } => "tangent",
            MateKind::Distance { .. } => "distance",
            MateKind::Angle { .. } => "angle",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MateDef {
    pub name: String,
    pub kind: MateKind,
    pub a: String,
    pub b: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Condition>,
    #[serde(default)]
    pub suppressed: bool,
}

/// One explicit row of a table pattern.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccurrenceRow {
    pub key: String,
    pub offset: OffsetExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "layout")]
pub enum OccurrenceLayout {
    /// `count` instances in all (the seed is step 0), `spacing` apart along `direction`.
    Linear {
        direction: [f64; 3],
        spacing: ValueExpr,
        count: ValueExpr,
    },
    Grid {
        first: [f64; 3],
        first_spacing: ValueExpr,
        first_count: ValueExpr,
        second: [f64; 3],
        second_spacing: ValueExpr,
        second_count: ValueExpr,
    },
    /// `count` instances in all, equally spaced over `angle` about the axis.
    Circular {
        axis_point: [ValueExpr; 3],
        axis: [f64; 3],
        count: ValueExpr,
        angle: ValueExpr,
    },
    Table {
        rows: Vec<OccurrenceRow>,
    },
    /// Follow a keyed pattern a definition publishes (a hole pattern): one instance per key,
    /// the seed sitting on the pattern's first key.
    FollowPattern {
        occurrence: RelPath,
        publish: PublishName,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccurrencePattern {
    pub name: String,
    pub seeds: Vec<String>,
    pub layout: OccurrenceLayout,
    #[serde(default)]
    pub skip: Vec<InstanceKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Condition>,
}

/// A plane in assembly coordinates.
// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "plane")]
pub enum PlaneSpec {
    Explicit {
        point: [ValueExpr; 3],
        normal: [f64; 3],
    },
    /// The XY plane of a frame.
    Frame { frame: FrameRef },
}

// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "hand")]
pub enum MirrorHand {
    /// The seed's definition is symmetric about this plane (definition coordinates), so the
    /// image reuses it with a proper rigid placement.
    Symmetric {
        point: [ValueExpr; 3],
        normal: [f64; 3],
    },
    /// The image uses the opposite-hand definition, which is the seed's mirror about this
    /// plane (definition coordinates).
    Opposite {
        source: OccurrenceSource,
        #[serde(default)]
        configuration: ConfigurationSelection,
        point: [ValueExpr; 3],
        normal: [f64; 3],
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccurrenceMirror {
    pub name: String,
    pub seeds: Vec<String>,
    pub plane: PlaneSpec,
    pub hand: MirrorHand,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Condition>,
}

/// A space allocation declared by this assembly (a keep-out or reserved volume).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvelopeDecl {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    pub min: [ValueExpr; 3],
    pub max: [ValueExpr; 3],
    /// Nothing but its owner may intrude.
    #[serde(default)]
    pub keep_out: bool,
}

/// What an assembly exposes to the assemblies that use it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "publishes")]
pub enum AssemblyPublishTarget {
    Connector { connector: String },
    Envelope { envelope: String },
    Occurrence { path: RelPath },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssemblyPublish {
    pub name: PublishName,
    pub target: AssemblyPublishTarget,
}

/// An assembly-level configuration: input values and suppressed children.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AssemblyConfiguration {
    pub name: ConfigurationName,
    #[serde(default)]
    pub overrides: BTreeMap<InputName, ValueExpr>,
    #[serde(default)]
    pub suppress: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssemblyDocument {
    pub path: FilePath,
    #[serde(default)]
    pub header: VersionHeader,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part_number: Option<String>,
    #[serde(default)]
    pub inputs: Vec<InputDecl>,
    #[serde(default)]
    pub lets: Vec<LetDecl>,
    /// Option and effectivity axes: meaningful on the top assembly of a product.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<OptionsBlock>,
    #[serde(default)]
    pub occurrences: Vec<Occurrence>,
    #[serde(default)]
    pub connectors: Vec<MateConnector>,
    #[serde(default)]
    pub mates: Vec<MateDef>,
    #[serde(default)]
    pub patterns: Vec<OccurrencePattern>,
    #[serde(default)]
    pub mirrors: Vec<OccurrenceMirror>,
    #[serde(default)]
    pub fasteners: Vec<FastenerSet>,
    #[serde(default)]
    pub envelopes: Vec<EnvelopeDecl>,
    #[serde(default)]
    pub publishes: Vec<AssemblyPublish>,
    #[serde(default)]
    pub configurations: Vec<AssemblyConfiguration>,
    #[serde(default)]
    pub properties: Vec<Property>,
    #[serde(default)]
    pub links: Vec<Link>,
    /// Named exploded states ([`crate::assembly::explode`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exploded: Vec<ExplodedView>,
    /// Saved model views ([`crate::assembly::views`]).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub views: Vec<crate::assembly::views::ModelView>,
    #[serde(default)]
    pub closed: bool,
}

impl AssemblyDocument {
    pub fn new(path: &str, name: &str) -> Self {
        Self {
            path: FilePath::new(path),
            header: VersionHeader::default(),
            name: name.into(),
            part_number: None,
            inputs: Vec::new(),
            lets: Vec::new(),
            options: None,
            occurrences: Vec::new(),
            connectors: Vec::new(),
            mates: Vec::new(),
            patterns: Vec::new(),
            mirrors: Vec::new(),
            fasteners: Vec::new(),
            envelopes: Vec::new(),
            publishes: Vec::new(),
            configurations: Vec::new(),
            properties: Vec::new(),
            links: Vec::new(),
            exploded: Vec::new(),
            views: Vec::new(),
            closed: false,
        }
    }

    /// Every alternative of a child name, in document order.
    pub fn alternatives<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Occurrence> + 'a {
        self.occurrences.iter().filter(move |o| o.name == name)
    }

    /// Child names in first-appearance order.
    pub fn child_names(&self) -> Vec<&str> {
        let mut seen = BTreeSet::new();
        self.occurrences
            .iter()
            .filter(|o| seen.insert(o.name.as_str()))
            .map(|o| o.name.as_str())
            .collect()
    }

    pub fn connector(&self, name: &str) -> Option<&MateConnector> {
        self.connectors.iter().find(|c| c.name == name)
    }

    pub fn configuration(&self, name: &ConfigurationName) -> Option<&AssemblyConfiguration> {
        self.configurations.iter().find(|c| &c.name == name)
    }

    /// Every file this assembly uses directly.
    pub fn uses(&self) -> BTreeSet<DefinitionRef> {
        let mut out: BTreeSet<DefinitionRef> = self
            .occurrences
            .iter()
            .map(|o| o.source.definition.clone())
            .collect();
        for m in &self.mirrors {
            if let MirrorHand::Opposite { source, .. } = &m.hand {
                out.insert(source.definition.clone());
            }
        }
        for f in &self.fasteners {
            out.insert(f.fastener.family.clone());
        }
        out
    }
}
