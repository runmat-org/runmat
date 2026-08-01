use globset::{Glob, GlobSet, GlobSetBuilder};

use super::{ArtifactEntryRole, PublicationEntry};
use crate::CacheError;

const ALWAYS_EXCLUDED: &[&str] = &[
    ".git",
    ".git/**",
    ".runmat",
    ".runmat/**",
    "target",
    "target/**",
];

pub struct PublicationPolicy {
    include: GlobSet,
    exclude: GlobSet,
    include_all: bool,
    allow_native: bool,
}

impl PublicationPolicy {
    pub fn new(
        include: &[String],
        exclude: &[String],
        allow_native: bool,
    ) -> Result<Self, CacheError> {
        Ok(Self {
            include: patterns(include)?,
            exclude: patterns(
                &exclude
                    .iter()
                    .map(String::as_str)
                    .chain(ALWAYS_EXCLUDED.iter().copied())
                    .map(str::to_string)
                    .collect::<Vec<_>>(),
            )?,
            include_all: include.is_empty(),
            allow_native,
        })
    }

    pub fn accepts(&self, entry: &PublicationEntry) -> Result<bool, CacheError> {
        let path = entry.path.as_str();
        let selected =
            (self.include_all || self.include.is_match(path)) && !self.exclude.is_match(path);
        if selected && entry.role == ArtifactEntryRole::Native && !self.allow_native {
            return Err(CacheError::InvalidObject(format!(
                "native publication entry `{path}` requires explicit native-artifact policy"
            )));
        }
        Ok(selected)
    }
}

impl Default for PublicationPolicy {
    fn default() -> Self {
        Self::new(&[], &[], false).expect("built-in publication patterns are valid")
    }
}

fn patterns(values: &[String]) -> Result<GlobSet, CacheError> {
    let mut builder = GlobSetBuilder::new();
    for value in values {
        if value.trim().is_empty() || value.starts_with('/') || value.contains('\\') {
            return Err(CacheError::InvalidObject(format!(
                "publication pattern `{value}` is not a normalized relative glob"
            )));
        }
        builder.add(
            Glob::new(value)
                .map_err(|error| CacheError::InvalidObject(format!("invalid glob: {error}")))?,
        );
    }
    builder
        .build()
        .map_err(|error| CacheError::InvalidObject(format!("invalid publication policy: {error}")))
}
