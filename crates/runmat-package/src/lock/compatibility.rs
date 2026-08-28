use super::PackageLock;
use crate::LockError;
use semver::{Version, VersionReq};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockCompatibility {
    pub runmat_version: Version,
}

impl LockCompatibility {
    pub fn validate(&self, lock: &PackageLock) -> Result<(), LockError> {
        lock.validate()?;
        for package in &lock.packages {
            let Some(requirement) = package.runmat_version.as_deref() else {
                continue;
            };
            let requirement = VersionReq::parse(requirement).map_err(|error| {
                LockError::Invalid(format!(
                    "package {} has invalid RunMat requirement `{requirement}`: {error}",
                    package.instance.package
                ))
            })?;
            if !requirement.matches(&self.runmat_version) {
                return Err(LockError::Incompatible(format!(
                    "package {} requires RunMat {requirement}, current version is {}",
                    package.instance.package, self.runmat_version
                )));
            }
        }
        Ok(())
    }
}
