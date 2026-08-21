use crate::{ContentDigest, IdentityError, NormalizedRelativePath};
use runmat_config::project::ProjectSourceIndex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SOURCE_INVENTORY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceInventoryEntry {
    pub source_root: NormalizedRelativePath,
    pub relative_path: NormalizedRelativePath,
    pub qualified_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub class_qualified_name: Option<String>,
    pub is_private: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceInventory {
    pub schema_version: u32,
    pub tree_digest: ContentDigest,
    pub entries: Vec<SourceInventoryEntry>,
    pub package_dirs: Vec<NormalizedRelativePath>,
    pub class_dirs: Vec<NormalizedRelativePath>,
    pub private_dirs: Vec<NormalizedRelativePath>,
}

impl SourceInventory {
    pub fn from_project_index(
        tree_digest: ContentDigest,
        index: ProjectSourceIndex,
    ) -> Result<Self, IdentityError> {
        let mut entries = index
            .files
            .into_iter()
            .map(|source| {
                Ok(SourceInventoryEntry {
                    source_root: NormalizedRelativePath::new(source.source_root)?,
                    relative_path: NormalizedRelativePath::new(source.relative_path)?,
                    qualified_name: source.qualified_name,
                    package_path: source.package_path,
                    class_name: source.class_name,
                    class_qualified_name: source.class_qualified_name,
                    is_private: source.is_private,
                })
            })
            .collect::<Result<Vec<_>, IdentityError>>()?;
        entries.sort();
        let package_dirs = normalize_paths(index.package_dirs)?;
        let class_dirs = normalize_paths(index.class_dirs)?;
        let private_dirs = normalize_paths(index.private_dirs)?;
        let inventory = Self {
            schema_version: SOURCE_INVENTORY_SCHEMA_VERSION,
            tree_digest,
            entries,
            package_dirs,
            class_dirs,
            private_dirs,
        };
        inventory
            .validate()
            .map_err(|reason| IdentityError::InvalidRelativePath {
                value: "source inventory".to_string(),
                reason,
            })?;
        Ok(inventory)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version != SOURCE_INVENTORY_SCHEMA_VERSION {
            return Err("unsupported source inventory schema");
        }
        if self.entries.windows(2).any(|pair| {
            pair[0] >= pair[1]
                || (&pair[0].source_root, &pair[0].relative_path)
                    == (&pair[1].source_root, &pair[1].relative_path)
        }) {
            return Err("source inventory entries must be strictly sorted");
        }
        if !strict_paths(&self.package_dirs)
            || !strict_paths(&self.class_dirs)
            || !strict_paths(&self.private_dirs)
        {
            return Err("source inventory directories must be strictly sorted");
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.qualified_name.trim().is_empty())
        {
            return Err("source inventory qualified names must be non-empty");
        }
        Ok(())
    }
}

fn normalize_paths(
    paths: Vec<std::path::PathBuf>,
) -> Result<Vec<NormalizedRelativePath>, IdentityError> {
    paths
        .into_iter()
        .map(NormalizedRelativePath::new)
        .collect::<Result<BTreeSet<_>, _>>()
        .map(|paths| paths.into_iter().collect())
}

fn strict_paths(paths: &[NormalizedRelativePath]) -> bool {
    paths.windows(2).all(|pair| pair[0] < pair[1])
}
