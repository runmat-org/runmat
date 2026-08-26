use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClasspathLayer {
    Bootstrap,
    Project,
    Dynamic,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ClasspathError {
    #[error("classpath entry is empty")]
    Empty,
    #[error("classpath entry {0} is already present")]
    Duplicate(String),
    #[error("classpath entry {0} is not in the dynamic classpath")]
    NotDynamic(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClasspathSnapshot {
    pub bootstrap: Vec<PathBuf>,
    pub project: Vec<PathBuf>,
    pub dynamic: Vec<PathBuf>,
    pub revision: u64,
    pub identity: String,
}

#[derive(Debug, Clone, Default)]
pub struct SessionClasspath {
    bootstrap: Vec<PathBuf>,
    project: Vec<PathBuf>,
    dynamic: Vec<PathBuf>,
    revision: u64,
}

impl SessionClasspath {
    pub fn new(
        bootstrap: impl IntoIterator<Item = PathBuf>,
        project: impl IntoIterator<Item = PathBuf>,
    ) -> Result<Self, ClasspathError> {
        let bootstrap = unique(bootstrap)?;
        let project = unique(project)?;
        let overlap = bootstrap.iter().find(|entry| project.contains(entry));
        if let Some(entry) = overlap {
            return Err(ClasspathError::Duplicate(entry.display().to_string()));
        }
        Ok(Self {
            bootstrap,
            project,
            dynamic: Vec::new(),
            revision: 1,
        })
    }

    pub fn add_dynamic(&mut self, entry: PathBuf) -> Result<(), ClasspathError> {
        validate_entry(&entry)?;
        if self.contains(&entry) {
            return Err(ClasspathError::Duplicate(entry.display().to_string()));
        }
        self.dynamic.push(entry);
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    pub fn remove_dynamic(&mut self, entry: &Path) -> Result<(), ClasspathError> {
        let Some(index) = self.dynamic.iter().position(|candidate| candidate == entry) else {
            return Err(ClasspathError::NotDynamic(entry.display().to_string()));
        };
        self.dynamic.remove(index);
        self.revision = self.revision.saturating_add(1);
        Ok(())
    }

    pub fn entries(&self, layer: ClasspathLayer) -> &[PathBuf] {
        match layer {
            ClasspathLayer::Bootstrap => &self.bootstrap,
            ClasspathLayer::Project => &self.project,
            ClasspathLayer::Dynamic => &self.dynamic,
        }
    }

    pub fn effective_entries(&self) -> impl Iterator<Item = &PathBuf> {
        self.bootstrap
            .iter()
            .chain(self.project.iter())
            .chain(self.dynamic.iter())
    }

    pub fn snapshot(&self) -> ClasspathSnapshot {
        let mut hasher = Sha256::new();
        hasher.update(b"runmat-java-classpath-v1\0");
        for (layer, entries) in [
            (b"bootstrap".as_slice(), &self.bootstrap),
            (b"project".as_slice(), &self.project),
            (b"dynamic".as_slice(), &self.dynamic),
        ] {
            hasher.update(layer);
            hasher.update([0]);
            for entry in entries {
                hasher.update(entry.as_os_str().to_string_lossy().as_bytes());
                hasher.update([0]);
            }
        }
        ClasspathSnapshot {
            bootstrap: self.bootstrap.clone(),
            project: self.project.clone(),
            dynamic: self.dynamic.clone(),
            revision: self.revision,
            identity: format!("java-classpath:v1:{:x}", hasher.finalize()),
        }
    }

    fn contains(&self, entry: &Path) -> bool {
        self.effective_entries().any(|candidate| candidate == entry)
    }
}

fn unique(entries: impl IntoIterator<Item = PathBuf>) -> Result<Vec<PathBuf>, ClasspathError> {
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for entry in entries {
        validate_entry(&entry)?;
        if !seen.insert(entry.clone()) {
            return Err(ClasspathError::Duplicate(entry.display().to_string()));
        }
        result.push(entry);
    }
    Ok(result)
}

fn validate_entry(entry: &Path) -> Result<(), ClasspathError> {
    if entry.as_os_str().is_empty() {
        Err(ClasspathError::Empty)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_mutation_preserves_layer_order_and_revision() {
        let mut classpath = SessionClasspath::new([PathBuf::from("bootstrap.jar")], []).unwrap();
        let initial = classpath.snapshot();
        classpath.add_dynamic(PathBuf::from("fixture.jar")).unwrap();
        let changed = classpath.snapshot();
        assert_eq!(changed.revision, initial.revision + 1);
        assert_ne!(changed.identity, initial.identity);
        assert_eq!(
            classpath.effective_entries().cloned().collect::<Vec<_>>(),
            vec![PathBuf::from("bootstrap.jar"), PathBuf::from("fixture.jar")]
        );
        classpath.remove_dynamic(Path::new("fixture.jar")).unwrap();
        assert!(classpath.entries(ClasspathLayer::Dynamic).is_empty());
    }

    #[test]
    fn classpath_entries_are_unique_across_layers() {
        let error = SessionClasspath::new(
            [PathBuf::from("fixture.jar")],
            [PathBuf::from("fixture.jar")],
        )
        .unwrap_err();
        assert!(matches!(error, ClasspathError::Duplicate(_)));
    }
}
