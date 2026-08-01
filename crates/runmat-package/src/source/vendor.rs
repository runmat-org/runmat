use crate::{ContentDigest, NormalizedRelativePath, SourceId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const VENDOR_MANIFEST_FILENAME: &str = "runmat-vendor.json";
pub const VENDOR_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VendorManifest {
    pub schema_version: u32,
    pub lock_digest: ContentDigest,
    pub packages: Vec<VendoredPackage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VendoredPackage {
    pub identity: ContentDigest,
    pub source: SourceId,
    pub path: NormalizedRelativePath,
}

impl VendorManifest {
    pub fn new(
        lock_digest: ContentDigest,
        mut packages: Vec<VendoredPackage>,
    ) -> Result<Self, String> {
        packages.sort_by(|left, right| left.identity.cmp(&right.identity));
        let manifest = Self {
            schema_version: VENDOR_SCHEMA_VERSION,
            lock_digest,
            packages,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != VENDOR_SCHEMA_VERSION {
            return Err(format!(
                "unsupported vendor schema {}; supported schema is {}",
                self.schema_version, VENDOR_SCHEMA_VERSION
            ));
        }
        if self
            .packages
            .windows(2)
            .any(|pair| pair[0].identity >= pair[1].identity)
        {
            return Err(
                "vendor packages must be strictly sorted by unique package identity".to_string(),
            );
        }
        let mut paths = BTreeSet::new();
        for package in &self.packages {
            if !paths.insert(package.path.clone()) {
                return Err(format!(
                    "vendor path `{}` is selected by multiple packages",
                    package.path
                ));
            }
        }
        Ok(())
    }

    pub fn package(&self, identity: &ContentDigest) -> Option<&VendoredPackage> {
        self.packages
            .binary_search_by(|package| package.identity.cmp(identity))
            .ok()
            .map(|index| &self.packages[index])
    }
}
