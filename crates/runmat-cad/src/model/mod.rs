//! The in-memory part model.
//!
//! A [`PartDocument`] is one `.part` file: inputs and lets, one feature history (sketches,
//! datums and features in order), any number of part outputs, regions, publish entries,
//! materials, links, configurations and semantic PMI. The syntax layer (M1) builds it from the lossless tree;
//! nothing here parses text.

pub mod curves;
pub mod datum;
pub mod detail;
pub mod direct;
pub mod feature;
pub mod feature_params;
pub mod hole_tables;
pub mod items;
pub mod mold;
pub mod sheet_metal;
pub mod sketch;
pub mod surface;
pub mod visit;
pub mod weldment;

use crate::ids::{ConfigurationName, FeatureId, FilePath, PublishName};
use crate::pmi::PmiSet;
use crate::value::{InputDecl, LetDecl};
use serde::{Deserialize, Serialize};

pub use curves::*;
pub use datum::*;
pub use detail::*;
pub use direct::*;
pub use feature::*;
pub use items::*;
pub use mold::MoldFeature;
pub use sheet_metal::SheetMetalFeature;
pub use sketch::*;
pub use surface::SurfaceFeature;
pub use weldment::WeldmentFeature;

/// The language and std version header (`runmat 0.7` / `std 1`). Old versions compile forever.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionHeader {
    pub language: String,
    pub std: String,
}

impl Default for VersionHeader {
    fn default() -> Self {
        Self {
            language: "0.7".into(),
            std: "1".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartDocument {
    pub path: FilePath,
    #[serde(default)]
    pub header: VersionHeader,
    #[serde(default)]
    pub inputs: Vec<InputDecl>,
    #[serde(default)]
    pub lets: Vec<LetDecl>,
    #[serde(default)]
    pub features: Vec<Feature>,
    /// Regenerate up to and including this feature only (the rollback bar). Editor state, kept
    /// here so the regeneration plan can honour it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_after: Option<FeatureId>,
    #[serde(default)]
    pub parts: Vec<PartOutput>,
    #[serde(default)]
    pub regions: Vec<Region>,
    #[serde(default)]
    pub publishes: Vec<PublishEntry>,
    #[serde(default)]
    pub materials: Vec<MaterialAssignment>,
    #[serde(default)]
    pub links: Vec<Link>,
    #[serde(default)]
    pub configurations: Vec<Configuration>,
    #[serde(default)]
    pub configuration_tables: Vec<ConfigurationTable>,
    /// Semantic PMI (datums, GD&T, sizes, finishes), resolved by regeneration. Its lists sit
    /// beside the features in the text: `datum A = { on = … }`, `tolerance flat = { … }`.
    #[serde(default, flatten)]
    pub pmi: PmiSet,
    /// The owner marked the file closed: auto-publish is refused.
    #[serde(default)]
    pub closed: bool,
    /// `feature def` templates defined in this file (library features).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub feature_defs: Vec<FeatureDef>,
}

impl PartDocument {
    pub fn new(path: &str) -> Self {
        Self {
            path: FilePath::new(path),
            header: VersionHeader::default(),
            inputs: Vec::new(),
            lets: Vec::new(),
            features: Vec::new(),
            rollback_after: None,
            parts: Vec::new(),
            regions: Vec::new(),
            publishes: Vec::new(),
            materials: Vec::new(),
            links: Vec::new(),
            configurations: Vec::new(),
            configuration_tables: Vec::new(),
            pmi: PmiSet::default(),
            closed: false,
            feature_defs: Vec::new(),
        }
    }

    pub fn feature(&self, id: &FeatureId) -> Option<&Feature> {
        self.features.iter().find(|f| &f.id == id)
    }

    pub fn feature_mut(&mut self, id: &str) -> Option<&mut Feature> {
        self.features.iter_mut().find(|f| f.id.0 == id)
    }

    pub fn feature_index(&self, id: &FeatureId) -> Option<usize> {
        self.features.iter().position(|f| &f.id == id)
    }

    pub fn publish(&self, name: &PublishName) -> Option<&PublishEntry> {
        self.publishes.iter().find(|p| &p.name == name)
    }

    /// Every configuration, with table rows expanded.
    pub fn all_configurations(&self) -> Vec<Configuration> {
        let mut out = self.configurations.clone();
        for t in &self.configuration_tables {
            out.extend(t.expand());
        }
        out
    }

    pub fn configuration(&self, name: &ConfigurationName) -> Option<Configuration> {
        self.all_configurations()
            .into_iter()
            .find(|c| &c.name == name)
    }

    /// Replace every library instance by its definition's features (pure: the same document
    /// expands the same way). Instances whose definition is not in this file stay, and fail
    /// at regeneration with a diagnostic; an expansion error is returned with the instance.
    pub fn expand_library(
        &self,
    ) -> (
        PartDocument,
        Vec<(FeatureId, crate::model::detail::ExpandError)>,
    ) {
        let mut out = self.clone();
        let mut errors = Vec::new();
        let mut features = Vec::new();
        for f in &self.features {
            if let FeatureKind::Library(inst) = &f.kind {
                let name = inst.def.rsplit("feature_def.").next().unwrap_or(&inst.def);
                if let Some(def) = self.feature_defs.iter().find(|d| d.name == name) {
                    match def.expand(&f.id, inst) {
                        Ok(fs) => {
                            features.extend(fs);
                            continue;
                        }
                        Err(e) => errors.push((f.id.clone(), e)),
                    }
                }
            }
            features.push(f.clone());
        }
        out.features = features;
        (out, errors)
    }

    /// Move a feature to just after `after` (or to the front). Returns false if either is
    /// missing. Ids do not change, so references written against them keep their meaning.
    pub fn reorder(&mut self, feature: &str, after: Option<&str>) -> bool {
        let Some(from) = self.features.iter().position(|f| f.id.0 == feature) else {
            return false;
        };
        let f = self.features.remove(from);
        let to = match after {
            None => 0,
            Some(a) => match self.features.iter().position(|x| x.id.0 == a) {
                Some(i) => i + 1,
                None => {
                    self.features.insert(from, f);
                    return false;
                }
            },
        };
        self.features.insert(to, f);
        true
    }
}
