//! The in-memory part model.
//!
//! A [`PartDocument`] is one `.part` file: inputs and lets, one feature history (sketches,
//! datums and features in order), any number of part outputs, regions, publish entries,
//! materials, links and configurations. The syntax layer (M1) builds it from the lossless tree;
//! nothing here parses text.

pub mod datum;
pub mod feature;
pub mod items;
pub mod sketch;

use crate::ids::{ConfigurationName, FeatureId, FilePath, PublishName};
use crate::value::{InputDecl, LetDecl};
use serde::{Deserialize, Serialize};

pub use datum::*;
pub use feature::*;
pub use items::*;
pub use sketch::*;

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
    /// The owner marked the file closed: auto-publish is refused.
    #[serde(default)]
    pub closed: bool,
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
            closed: false,
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
