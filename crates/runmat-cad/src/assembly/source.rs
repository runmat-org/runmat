//! The design-source seam: how the assembly layer reaches other files without loading the
//! product.
//!
//! Composition needs, per occurrence, only:
//! - the child `.assembly` document when the child is an assembly being expanded;
//! - a [`DefinitionSummary`] (R0: part number, configurations, typed inputs, declared
//!   representations, properties);
//! - a [`ConfiguredSummary`] for one configuration, *only* when an operation needs geometry
//!   facts: published frames for connectors and attachments, keyed patterns, fastener paths,
//!   boxes for interference. These are derived artifacts (`.artifacts`, server mirror) keyed by
//!   the definition's chained regeneration hash; a native part regenerates to produce one.
//!
//! [`MemorySource`] plays the server index's role for tests and local sessions, and counts every
//! load so tests can prove laziness.

use crate::assembly::model::{AssemblyDocument, DefinitionKind, DefinitionRef, Representation};
use crate::ids::{
    ConfigurationName, FilePath, InputName, MaterialRef, PartitionName, PublishName, RevisionId,
};
use crate::model::datum::Frame;
use crate::model::items::Property;
use crate::refs::InstanceKey;
use crate::regen::kernel::Transform;
use crate::units::QuantityKindId;
use crate::value::Quantity;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::Arc;
use thiserror::Error;

/// An axis-aligned box (SI).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: [f64::INFINITY; 3],
        max: [f64::NEG_INFINITY; 3],
    };

    pub fn new(min: [f64; 3], max: [f64; 3]) -> Self {
        Self { min, max }
    }

    pub fn is_empty(&self) -> bool {
        (0..3).any(|i| self.min[i] > self.max[i])
    }

    pub fn include(&mut self, p: [f64; 3]) {
        for (i, v) in p.iter().enumerate() {
            self.min[i] = self.min[i].min(*v);
            self.max[i] = self.max[i].max(*v);
        }
    }

    pub fn union(&self, o: &Aabb) -> Aabb {
        let mut b = *self;
        if !o.is_empty() {
            b.include(o.min);
            b.include(o.max);
        }
        b
    }

    /// The box of the transformed corners (conservative for any rigid transform).
    pub fn transformed(&self, t: &Transform) -> Aabb {
        if self.is_empty() {
            return *self;
        }
        let mut b = Aabb::EMPTY;
        for k in 0..8 {
            let p = [
                if k & 1 == 0 { self.min[0] } else { self.max[0] },
                if k & 2 == 0 { self.min[1] } else { self.max[1] },
                if k & 4 == 0 { self.min[2] } else { self.max[2] },
            ];
            b.include(t.apply(p));
        }
        b
    }

    /// Overlap with a margin (a clearance request inflates by its minimum distance).
    pub fn overlaps(&self, o: &Aabb, margin: f64) -> bool {
        (0..3).all(|i| self.min[i] - margin <= o.max[i] && o.min[i] - margin <= self.max[i])
    }

    pub fn intersection(&self, o: &Aabb) -> Aabb {
        let mut b = Aabb::EMPTY;
        for i in 0..3 {
            b.min[i] = self.min[i].max(o.min[i]);
            b.max[i] = self.max[i].min(o.max[i]);
        }
        b
    }

    pub fn volume(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            (0..3).map(|i| self.max[i] - self.min[i]).product()
        }
    }
}

/// One chain of a published edge set: points with the surface normal at each.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathChain {
    /// A stable key from the publisher (`upper_flange`, `chain0` from its own definition).
    pub key: String,
    pub points: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
}

/// Geometry a definition publishes, in definition coordinates (its interface at R0).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "geometry")]
pub enum PublishedGeometry {
    Frame {
        frame: Frame,
    },
    /// A plane: its frame (Z is the normal).
    Plane {
        frame: Frame,
    },
    Axis {
        point: [f64; 3],
        direction: [f64; 3],
    },
    Point {
        at: [f64; 3],
    },
    /// A keyed pattern (hole pattern): one frame per stable key.
    Pattern {
        frames: Vec<(InstanceKey, Frame)>,
    },
    /// An edge set as polylines (fastener rows run along these).
    Path {
        chains: Vec<PathChain>,
    },
    Envelope {
        aabb: Aabb,
    },
    Value {
        value: Quantity,
    },
    /// Published topology the assembly layer cannot use as a frame without the kernel.
    Entities {
        count: u32,
    },
}

impl PublishedGeometry {
    /// The frame this publishes, when it is one (an axis gives a frame with Z along it; a point
    /// a frame with world axes).
    pub fn frame(&self) -> Option<Frame> {
        match self {
            PublishedGeometry::Frame { frame } | PublishedGeometry::Plane { frame } => Some(*frame),
            PublishedGeometry::Axis { point, direction } => {
                let t = constraint_basis(*direction);
                Some(Frame {
                    origin: *point,
                    x: t.0,
                    y: t.1,
                    z: t.2,
                })
            }
            PublishedGeometry::Point { at } => Some(Frame {
                origin: *at,
                ..Frame::WORLD
            }),
            _ => None,
        }
    }
}

/// A right-handed basis with Z along `z`, X chosen deterministically.
pub(crate) fn constraint_basis(z: [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let n = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2]).sqrt();
    let z = if n > 0.0 {
        [z[0] / n, z[1] / n, z[2] / n]
    } else {
        [0.0, 0.0, 1.0]
    };
    let helper = if z[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    // y = z × helper, x = y × z.
    let y = cross(z, helper);
    let yn = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
    let y = [y[0] / yn, y[1] / yn, y[2] / yn];
    let x = cross(y, z);
    (x, y, z)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A resolved configuration of a definition: a named configuration, input values in SI, or
/// both. Its [`ConfigKey::digest`] is canonical (bit-exact SI values in name order).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfigKey {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<ConfigurationName>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub inputs: BTreeMap<InputName, f64>,
}

impl ConfigKey {
    pub fn named(name: &str) -> Self {
        Self {
            name: Some(ConfigurationName::new(name)),
            inputs: BTreeMap::new(),
        }
    }

    pub fn digest(&self) -> String {
        let mut s = String::new();
        if let Some(n) = &self.name {
            s.push_str(n.as_str());
        }
        for (k, v) in &self.inputs {
            let _ = write!(s, "|{}={:016x}", k, v.to_bits());
        }
        s
    }

    /// A short label for people: `light`, `light{thickness=0.005}`, `default`.
    pub fn label(&self) -> String {
        let mut s = self
            .name
            .as_ref()
            .map(|n| n.0.clone())
            .unwrap_or_else(|| "default".into());
        if !self.inputs.is_empty() {
            s.push('{');
            for (i, (k, v)) in self.inputs.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                let _ = write!(s, "{}={}", k, v);
            }
            s.push('}');
        }
        s
    }
}

/// R0 facts about a definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DefinitionSummary {
    pub definition: DefinitionRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part_number: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub configurations: Vec<ConfigurationName>,
    /// Typed inputs a configuration may set.
    #[serde(default)]
    pub inputs: Vec<(InputName, QuantityKindId)>,
    /// Representations the definition declares beyond the always-available ones.
    #[serde(default)]
    pub representations: BTreeSet<Representation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialRef>,
    #[serde(default)]
    pub properties: Vec<Property>,
    /// Content digest of the definition source (joins artifact keys).
    #[serde(default)]
    pub digest: String,
}

impl DefinitionSummary {
    pub fn new(definition: DefinitionRef) -> Self {
        Self {
            definition,
            part_number: None,
            description: None,
            configurations: Vec::new(),
            inputs: Vec::new(),
            representations: BTreeSet::new(),
            material: None,
            properties: Vec::new(),
            digest: String::new(),
        }
    }

    /// Whether the definition can supply `r`: `full` and `display` always; `envelope` is
    /// derived from the box when not declared; `interface` from the publish table; the
    /// idealisations must be declared.
    pub fn provides(&self, r: Representation) -> bool {
        match r {
            Representation::Full | Representation::Display | Representation::Envelope => true,
            Representation::Interface => true,
            Representation::Defeatured | Representation::Midsurface | Representation::Beam => {
                self.representations.contains(&r)
            }
        }
    }
}

/// Facts about one configuration of a definition (derived, cached by its chained key).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfiguredSummary {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Aabb>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mass: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center_of_mass: Option<[f64; 3]>,
    /// A configuration-specific part number (a dash number), overriding the definition's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part_number: Option<String>,
    #[serde(default)]
    pub publishes: BTreeMap<PublishName, PublishedGeometry>,
    /// The artifact key (the definition's final chained regeneration key).
    #[serde(default)]
    pub artifact: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error")]
pub enum SourceError {
    #[error("`{file}` is not in the project")]
    NotFound { file: FilePath },
    #[error("`{file}` has no revision `{revision}`")]
    RevisionMissing {
        file: FilePath,
        revision: RevisionId,
    },
    #[error("`{definition}` is a {found:?}, not a {expected:?}")]
    WrongKind {
        definition: String,
        expected: DefinitionKind,
        found: DefinitionKind,
    },
    #[error("`{definition}` has no configuration `{configuration}`")]
    NoConfiguration {
        definition: String,
        configuration: String,
    },
    #[error("{message}")]
    Failed { message: String },
}

pub trait DesignSource {
    fn assembly(
        &self,
        file: &FilePath,
        revision: Option<&RevisionId>,
    ) -> Result<Arc<AssemblyDocument>, SourceError>;
    fn definition(
        &self,
        definition: &DefinitionRef,
        revision: Option<&RevisionId>,
    ) -> Result<Arc<DefinitionSummary>, SourceError>;
    fn configured(
        &self,
        definition: &DefinitionRef,
        revision: Option<&RevisionId>,
        configuration: &ConfigKey,
    ) -> Result<Arc<ConfiguredSummary>, SourceError>;
    fn partition(&self, file: &FilePath) -> Option<PartitionName>;
    fn latest_release(&self, file: &FilePath) -> Option<RevisionId>;
}

/// Computes a configured summary for input values no stored configuration covers.
pub type ConfigureFn = fn(&DefinitionRef, &ConfigKey) -> Result<ConfiguredSummary, String>;

#[derive(Clone, Debug)]
pub struct MemoryPart {
    pub summary: Arc<DefinitionSummary>,
    /// By [`ConfigKey::digest`]; `""` is the default configuration.
    pub configured: BTreeMap<String, Arc<ConfiguredSummary>>,
    pub configure: Option<ConfigureFn>,
}

/// Load counters, for laziness tests and the scale report.
#[derive(Debug, Default)]
pub struct LoadCounters {
    pub assemblies: Cell<u64>,
    pub definitions: Cell<u64>,
    pub configured: Cell<u64>,
}

#[derive(Debug, Default)]
pub struct MemorySource {
    pub assemblies: BTreeMap<(FilePath, Option<RevisionId>), Arc<AssemblyDocument>>,
    pub parts: BTreeMap<(DefinitionRef, Option<RevisionId>), MemoryPart>,
    pub partitions: BTreeMap<FilePath, PartitionName>,
    pub releases: BTreeMap<FilePath, Vec<RevisionId>>,
    pub loads: LoadCounters,
}

impl MemorySource {
    pub fn add_assembly(&mut self, doc: AssemblyDocument) {
        self.assemblies
            .insert((doc.path.clone(), None), Arc::new(doc));
    }

    pub fn add_assembly_revision(&mut self, doc: AssemblyDocument, revision: &str) {
        let rev = RevisionId::new(revision);
        self.releases
            .entry(doc.path.clone())
            .or_default()
            .push(rev.clone());
        self.assemblies
            .insert((doc.path.clone(), Some(rev)), Arc::new(doc));
    }

    /// Add a part (or circuit) with its default configured summary.
    pub fn add_part(&mut self, summary: DefinitionSummary, default: ConfiguredSummary) {
        let mut configured = BTreeMap::new();
        configured.insert(String::new(), Arc::new(default));
        self.parts.insert(
            (summary.definition.clone(), None),
            MemoryPart {
                summary: Arc::new(summary),
                configured,
                configure: None,
            },
        );
    }

    pub fn add_part_revision(
        &mut self,
        summary: DefinitionSummary,
        default: ConfiguredSummary,
        revision: &str,
    ) {
        let rev = RevisionId::new(revision);
        self.releases
            .entry(summary.definition.file.clone())
            .or_default()
            .push(rev.clone());
        let mut configured = BTreeMap::new();
        configured.insert(String::new(), Arc::new(default));
        self.parts.insert(
            (summary.definition.clone(), Some(rev)),
            MemoryPart {
                summary: Arc::new(summary),
                configured,
                configure: None,
            },
        );
    }

    pub fn add_configuration(
        &mut self,
        definition: &DefinitionRef,
        key: &ConfigKey,
        summary: ConfiguredSummary,
    ) {
        if let Some(p) = self.parts.get_mut(&(definition.clone(), None)) {
            p.configured.insert(key.digest(), Arc::new(summary));
        }
    }

    pub fn set_configure(&mut self, definition: &DefinitionRef, f: ConfigureFn) {
        if let Some(p) = self.parts.get_mut(&(definition.clone(), None)) {
            p.configure = Some(f);
        }
    }

    pub fn set_partition(&mut self, file: &str, partition: &str) {
        self.partitions
            .insert(FilePath::new(file), PartitionName::new(partition));
    }

    fn part(
        &self,
        definition: &DefinitionRef,
        revision: Option<&RevisionId>,
    ) -> Result<&MemoryPart, SourceError> {
        if let Some(p) = self.parts.get(&(definition.clone(), revision.cloned())) {
            return Ok(p);
        }
        let known = self.parts.keys().any(|(d, _)| d.file == definition.file);
        match (known, revision) {
            (true, Some(r)) => Err(SourceError::RevisionMissing {
                file: definition.file.clone(),
                revision: r.clone(),
            }),
            _ => Err(SourceError::NotFound {
                file: definition.file.clone(),
            }),
        }
    }
}

impl DesignSource for MemorySource {
    fn assembly(
        &self,
        file: &FilePath,
        revision: Option<&RevisionId>,
    ) -> Result<Arc<AssemblyDocument>, SourceError> {
        self.loads.assemblies.set(self.loads.assemblies.get() + 1);
        if let Some(a) = self.assemblies.get(&(file.clone(), revision.cloned())) {
            return Ok(a.clone());
        }
        match revision {
            Some(r) if self.assemblies.contains_key(&(file.clone(), None)) => {
                Err(SourceError::RevisionMissing {
                    file: file.clone(),
                    revision: r.clone(),
                })
            }
            _ => Err(SourceError::NotFound { file: file.clone() }),
        }
    }

    fn definition(
        &self,
        definition: &DefinitionRef,
        revision: Option<&RevisionId>,
    ) -> Result<Arc<DefinitionSummary>, SourceError> {
        self.loads.definitions.set(self.loads.definitions.get() + 1);
        if definition.kind == DefinitionKind::Assembly {
            // The R0 summary of an assembly is derived from its document.
            let doc = self
                .assemblies
                .get(&(definition.file.clone(), revision.cloned()))
                .ok_or_else(|| SourceError::NotFound {
                    file: definition.file.clone(),
                })?;
            let mut s = DefinitionSummary::new(definition.clone());
            s.part_number = doc.part_number.clone();
            s.configurations = doc.configurations.iter().map(|c| c.name.clone()).collect();
            s.inputs = doc
                .inputs
                .iter()
                .map(|i| (i.name.clone(), i.kind.clone()))
                .collect();
            s.properties = doc.properties.clone();
            return Ok(Arc::new(s));
        }
        let p = self.part(definition, revision)?;
        if p.summary.definition.kind != definition.kind {
            return Err(SourceError::WrongKind {
                definition: definition.to_string(),
                expected: definition.kind,
                found: p.summary.definition.kind,
            });
        }
        Ok(p.summary.clone())
    }

    fn configured(
        &self,
        definition: &DefinitionRef,
        revision: Option<&RevisionId>,
        configuration: &ConfigKey,
    ) -> Result<Arc<ConfiguredSummary>, SourceError> {
        self.loads.configured.set(self.loads.configured.get() + 1);
        if definition.kind == DefinitionKind::Assembly {
            return Ok(Arc::new(ConfiguredSummary::default()));
        }
        let p = self.part(definition, revision)?;
        if let Some(c) = p.configured.get(&configuration.digest()) {
            return Ok(c.clone());
        }
        if let Some(f) = p.configure {
            return f(definition, configuration)
                .map(Arc::new)
                .map_err(|message| SourceError::Failed { message });
        }
        // Never substitute another configuration's facts: the regenerator would produce
        // them, and a stand-in that guessed would give quietly wrong geometry.
        Err(SourceError::NoConfiguration {
            definition: definition.to_string(),
            configuration: configuration.label(),
        })
    }

    fn partition(&self, file: &FilePath) -> Option<PartitionName> {
        self.partitions.get(file).cloned()
    }

    fn latest_release(&self, file: &FilePath) -> Option<RevisionId> {
        self.releases.get(file).and_then(|r| r.last().cloned())
    }
}
