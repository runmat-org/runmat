use crate::{CanonicalPackageId, PackageAlias};
use semver::VersionReq;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequirementPath {
    pub root: String,
    pub aliases: Vec<PackageAlias>,
}

impl Display for RequirementPath {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.root)?;
        for alias in &self.aliases {
            write!(formatter, " -> {alias}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Incompatibility {
    pub package: Box<CanonicalPackageId>,
    pub requirement: Box<VersionReq>,
    pub paths: Vec<RequirementPath>,
    pub reason: String,
}

impl Display for Incompatibility {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "cannot resolve {} {}: {}",
            self.package, self.requirement, self.reason
        )?;
        for path in &self.paths {
            write!(formatter, "\n  required by {path}")?;
        }
        Ok(())
    }
}
