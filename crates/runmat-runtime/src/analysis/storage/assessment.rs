//! Resource admission for authenticated assessment inputs on each artifact store.
use std::io::Write;
use std::path::Path;

/// Authenticated evidence shared by standard and catalog-backed numerical publications.
/// An artifact store supplies this projection after validating its own immutable format.
#[derive(serde::Serialize)]
pub struct AssessmentRunEvidence {
    pub identity: runmat_analysis_fea::AnalysisRunIdentity,
    pub numerically_acceptable: bool,
    pub observation_scalars:
        Vec<runmat_analysis_fea::results::assessment::ScalarObservationEvidence>,
}

impl From<&super::AnalysisRunResult> for AssessmentRunEvidence {
    fn from(run: &super::AnalysisRunResult) -> Self {
        Self {
            identity: run.run.identity.clone(),
            numerically_acceptable: run.numerically_acceptable,
            observation_scalars: run.observation_scalars.clone(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AssessmentStoreError {
    #[error("assessment artifact exceeds its byte budget")]
    ResourceLimit,
    #[error("invalid assessment artifact: {0}")]
    Invalid(String),
}

struct BoundedSize {
    remaining: u64,
    exceeded: bool,
}
impl Write for BoundedSize {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        if length > self.remaining {
            self.exceeded = true;
            return Err(std::io::Error::other("artifact byte budget exceeded"));
        }
        self.remaining -= length;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) fn check_encoded_size(
    value: &impl serde::Serialize,
    maximum_bytes: u64,
) -> Result<(), AssessmentStoreError> {
    encoded_size(value, maximum_bytes).map(|_| ())
}

pub(super) fn encoded_size(
    value: &impl serde::Serialize,
    maximum_bytes: u64,
) -> Result<u64, AssessmentStoreError> {
    let mut writer = BoundedSize {
        remaining: maximum_bytes,
        exceeded: false,
    };
    let result = serde_json::to_writer(&mut writer, value);
    if writer.exceeded {
        return Err(AssessmentStoreError::ResourceLimit);
    }
    result.map_err(|error| AssessmentStoreError::Invalid(error.to_string()))?;
    Ok(maximum_bytes - writer.remaining)
}

pub(super) fn read_bounded_artifact(
    store: &super::access::FilesystemAccess<'_>,
    path: &Path,
    maximum_bytes: u64,
) -> Result<Option<Vec<u8>>, AssessmentStoreError> {
    let metadata = match store.fs_symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(AssessmentStoreError::Invalid(error.to_string())),
    };
    if metadata.file_type() != runmat_filesystem::FsFileType::File {
        return Err(AssessmentStoreError::Invalid(
            "artifact must be a regular file".into(),
        ));
    }
    if metadata.len() > maximum_bytes {
        return Err(AssessmentStoreError::ResourceLimit);
    }
    if metadata.len() > store.remaining_bytes() {
        return Err(AssessmentStoreError::ResourceLimit);
    }
    let bytes = store.fs_read(path).map_err(|error| {
        AssessmentStoreError::from(super::AnalysisPublicationError::from_io(error))
    })?;
    if bytes.len() as u64 > maximum_bytes {
        return Err(AssessmentStoreError::ResourceLimit);
    }
    Ok(Some(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read_bounded_artifact(
        path: &Path,
        maximum: u64,
    ) -> Result<Option<Vec<u8>>, AssessmentStoreError> {
        let root = path
            .parent()
            .ok_or_else(|| AssessmentStoreError::Invalid("artifact parent absent".into()))?;
        let store = super::super::FilesystemAnalysisArtifactStore::new(root.to_path_buf());
        let access = store.access(false).map_err(AssessmentStoreError::Invalid)?;
        super::read_bounded_artifact(&access, path, maximum)
    }
    #[test]
    fn size_admission_counts_without_materializing_another_artifact() {
        assert!(check_encoded_size(&vec![1, 2, 3], 7).is_ok());
        assert!(matches!(
            check_encoded_size(&vec![1, 2, 3], 6),
            Err(AssessmentStoreError::ResourceLimit)
        ));
    }
    #[test]
    fn file_admission_checks_budget_before_decoding() {
        let _guard = crate::analysis::tests::analysis_test_guard();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("artifact.json");
        std::fs::write(&path, b"[1,2,3]").unwrap();
        assert!(matches!(
            read_bounded_artifact(&path, 6),
            Err(AssessmentStoreError::ResourceLimit)
        ));
        assert_eq!(
            read_bounded_artifact(&path, 7).unwrap(),
            Some(b"[1,2,3]".to_vec())
        );
        assert!(read_bounded_artifact(&directory.path().join("missing"), 7)
            .unwrap()
            .is_none());
    }
}
