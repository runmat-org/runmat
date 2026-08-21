use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use super::CoverageSite;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoveragePathClass {
    Project,
    Generated,
    Vendor,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoverageFilter {
    pub roots: Vec<String>,
    pub exclude: Vec<String>,
    pub include_generated: bool,
    pub include_vendor: bool,
}

impl CoverageFilter {
    pub fn includes(&self, site: &CoverageSite) -> bool {
        let path = normalize(&site.relative_path);
        let mut candidates = vec![path.clone()];
        if let Some(owner) = site.owner_identity.strip_prefix("path:") {
            let owner = normalize(owner);
            candidates.push(if owner.is_empty() {
                path.clone()
            } else {
                format!("{owner}/{path}")
            });
        }
        let class = classify(&path);
        if class == CoveragePathClass::Generated && !self.include_generated {
            return false;
        }
        if class == CoveragePathClass::Vendor && !self.include_vendor {
            return false;
        }
        if !self.roots.is_empty()
            && !self.roots.iter().map(normalize).any(|root| {
                candidates
                    .iter()
                    .any(|path| path == &root || path.starts_with(&(root.clone() + "/")))
            })
        {
            return false;
        }
        !self
            .exclude
            .iter()
            .map(normalize)
            .any(|pattern| candidates.iter().any(|path| glob_matches(&pattern, path)))
    }
}

fn classify(path: &str) -> CoveragePathClass {
    let components = Path::new(path)
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        });
    for component in components {
        let lowercase = component.to_ascii_lowercase();
        if matches!(
            lowercase.as_str(),
            "vendor" | "vendors" | "third_party" | "third-party" | "node_modules"
        ) {
            return CoveragePathClass::Vendor;
        }
        if matches!(
            lowercase.as_str(),
            "generated" | "gen" | "target" | "dist" | "build"
        ) {
            return CoveragePathClass::Generated;
        }
    }
    CoveragePathClass::Project
}

fn normalize(path: impl AsRef<str>) -> String {
    path.as_ref()
        .replace('\\', "/")
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_owned()
}

fn glob_matches(pattern: &str, path: &str) -> bool {
    let pattern = pattern.as_bytes();
    let path = path.as_bytes();
    let mut memo = vec![vec![None; path.len() + 1]; pattern.len() + 1];
    fn matches(
        pattern: &[u8],
        path: &[u8],
        pattern_index: usize,
        path_index: usize,
        memo: &mut [Vec<Option<bool>>],
    ) -> bool {
        if let Some(result) = memo[pattern_index][path_index] {
            return result;
        }
        let result = if pattern_index == pattern.len() {
            path_index == path.len()
        } else if pattern[pattern_index] == b'*' && pattern.get(pattern_index + 1) == Some(&b'*') {
            matches(pattern, path, pattern_index + 2, path_index, memo)
                || (path_index < path.len()
                    && matches(pattern, path, pattern_index, path_index + 1, memo))
        } else if pattern[pattern_index] == b'*' {
            matches(pattern, path, pattern_index + 1, path_index, memo)
                || (path_index < path.len()
                    && path[path_index] != b'/'
                    && matches(pattern, path, pattern_index, path_index + 1, memo))
        } else {
            path_index < path.len()
                && (pattern[pattern_index] == path[path_index]
                    || (pattern[pattern_index] == b'?' && path[path_index] != b'/'))
                && matches(pattern, path, pattern_index + 1, path_index + 1, memo)
        };
        memo[pattern_index][path_index] = Some(result);
        result
    }
    matches(pattern, path, 0, 0, &mut memo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::CoverageMetric;

    fn site(path: &str) -> CoverageSite {
        CoverageSite {
            id: path.into(),
            counter_key: 0,
            metric: CoverageMetric::Statement,
            owner_identity: "root".into(),
            relative_path: path.into(),
            semantic_path: "test".into(),
            source_id: 0,
            start_byte: 0,
            end_byte: 1,
            start_line: 1,
            start_column: 1,
            end_line: 1,
            end_column: 2,
            instrumented: true,
            unsupported_reason: None,
        }
    }

    #[test]
    fn applies_roots_exclusions_and_generated_vendor_defaults() {
        let filter = CoverageFilter {
            roots: vec!["src".into()],
            exclude: vec!["src/private/**".into()],
            ..CoverageFilter::default()
        };
        assert!(filter.includes(&site("src/main.m")));
        assert!(!filter.includes(&site("tests/test_main.m")));
        assert!(!filter.includes(&site("src/private/secret.m")));
        assert!(!filter.includes(&site("src/vendor/dependency.m")));
        assert!(!filter.includes(&site("src/generated/table.m")));
    }

    #[test]
    fn path_owner_supports_absolute_runtime_roots() {
        let mut source = site("covered.m");
        source.owner_identity = "path:/workspace/src".into();
        let filter = CoverageFilter {
            roots: vec!["/workspace/src".into()],
            ..CoverageFilter::default()
        };
        assert!(filter.includes(&source));
    }
}
