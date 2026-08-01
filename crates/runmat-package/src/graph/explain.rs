use crate::PackageAlias;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyPath {
    pub root: String,
    pub aliases: Vec<PackageAlias>,
}

impl Display for DependencyPath {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.root)?;
        for alias in &self.aliases {
            write!(formatter, " -> {alias}")?;
        }
        Ok(())
    }
}
