//! Regions, publish entries, properties, materials, links and configurations.

use crate::ids::{
    ConfigurationName, FeatureId, InputName, LetName, MaterialRef, PartName, PropertyName,
    PublishName, RegionName,
};
use crate::model::datum::DatumRef;
use crate::refs::{Reference, TopoKind};
use crate::value::ValueExpr;
use serde::{Deserialize, Serialize};


/// How many entities a reference site accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cardinality {
    ExactlyOne,
    AtLeastOne,
    /// Zero is allowed (an optional selection).
    Any,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub name: RegionName,
    pub kind: TopoKind,
    pub select: Vec<Reference>,
    pub expect: Cardinality,
    /// The name the region's faces travel under in STEP, when it differs from the region's
    /// identifier: a simulation selects `select name "Port P"` by it (`label = "Port P"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Only the faces of this body (`body = feature.oil.body`): where two bodies share a face, as
    /// a fluid and the part it fills do, each body's copy of that face takes its own region.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Reference>,
}

/// What a `publish` entry exposes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "publishes")]
pub enum PublishTarget {
    /// Topology: faces, edges, vertices or bodies.
    Entities {
        kind: TopoKind,
        references: Vec<Reference>,
    },
    Datum {
        datum: DatumRef,
    },
    Region {
        region: RegionName,
    },
    Part {
        part: PartName,
    },
    /// A plain value: an input (a parameter, with its unit).
    Value {
        input: InputName,
    },
    /// A plain value derived by a `let`.
    Let {
        name: LetName,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PublishEntry {
    pub name: PublishName,
    pub target: PublishTarget,
    /// Written by auto-publish (a pick from another file), as opposed to by hand.
    #[serde(default)]
    pub auto: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "value")]
pub enum PropertyValue {
    Text(String),
    Quantity(ValueExpr),
    Boolean(bool),
}

/// The scope a property schema field applies at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyScope {
    #[default]
    Definition,
    Configuration,
    Occurrence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Property {
    pub name: PropertyName,
    pub value: PropertyValue,
    #[serde(default)]
    pub scope: PropertyScope,
}

/// A material set on a body. Bodies without an assignment take the part output's material.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialAssignment {
    pub body: Reference,
    pub material: MaterialRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    DerivedFrom,
    Verifies,
    Supersedes,
    SupplierDoc,
}

/// An item within this file that a link starts from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "item")]
pub enum LocalItem {
    Part { part: PartName },
    Feature { feature: FeatureId },
    Region { region: RegionName },
    Publish { publish: PublishName },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub kind: LinkKind,
    pub from: LocalItem,
    /// An item path in another file (`checks/mass.fea#check.mass_budget`) or an external URI.
    pub to: String,
}

/// A part output: one body promoted to a first-class part with its own identity, part number
/// and properties (`"flange.part#gasket"`). A `.part` may hold several.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartOutput {
    pub name: PartName,
    pub body: Reference,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part_number: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialRef>,
    /// The part's display color, sRGB `#RRGGBB` (opaque) or `#RRGGBBAA`: its appearance, set
    /// apart from its material (a painted skin, tinted glazing). Absent: the material's
    /// appearance ([`crate::appearance`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default)]
    pub properties: Vec<Property>,
    /// The product name the output travels under in STEP, when it differs from the output's
    /// identifier (`label = "Cold plate"`): what a simulation's `select name` reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

pub use crate::model::configuration::Configuration;

/// `configurations from table`: one configuration per row, named `<table>.<row key>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationTable {
    pub name: String,
    pub columns: Vec<InputName>,
    pub rows: Vec<TableRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableRow {
    pub key: String,
    pub values: Vec<ValueExpr>,
}

impl ConfigurationTable {
    pub fn expand(&self) -> Vec<Configuration> {
        self.rows
            .iter()
            .map(|row| Configuration {
                name: ConfigurationName::new(format!("{}.{}", self.name, row.key)),
                overrides: self
                    .columns
                    .iter()
                    .cloned()
                    .zip(row.values.iter().cloned())
                    .collect(),
                ..Configuration::default()
            })
            .collect()
    }
}
