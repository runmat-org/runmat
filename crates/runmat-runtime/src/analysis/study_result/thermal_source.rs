//! Placement-neutral replay authentication and complete durable thermal evidence.
use super::ThermalStudyCompletion;
use runmat_execution::{value::PublishedObjectSource, Digest};
use std::sync::Arc;
pub trait ThermalStudySource: PublishedObjectSource {
    fn authenticate(&self, completed: &ThermalStudyCompletion) -> Result<(), String>;
}
#[derive(Clone)]
pub struct ThermalStudyObjectSource(Arc<dyn ThermalStudySource>);
impl std::fmt::Debug for ThermalStudyObjectSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThermalStudyObjectSource")
            .finish_non_exhaustive()
    }
}
impl ThermalStudyObjectSource {
    pub fn new(source: impl ThermalStudySource + 'static) -> Self {
        Self(Arc::new(source))
    }
    pub(in crate::analysis) fn authenticate(
        &self,
        completed: &ThermalStudyCompletion,
    ) -> Result<(), String> {
        self.0.authenticate(completed)
    }
    pub(in crate::analysis) fn persist(
        &self,
        completed: &ThermalStudyCompletion,
        study_digest: Digest,
    ) -> Result<Vec<String>, String> {
        self.authenticate(completed)?;
        completed
            .catalog
            .objects
            .iter()
            .map(|reference| {
                let bytes = self.0.read_verified(reference)?;
                if Digest::sha256(&bytes) != reference.logical_digest
                    || bytes.len() as u64 != reference.encoded_length
                {
                    return Err("thermal catalog object digest or length mismatch".into());
                }
                super::super::structural_study::persist_content_addressed_study_evidence(
                    &study_digest.to_string(),
                    &format!(
                        "thermal-object-{}",
                        reference
                            .logical_digest
                            .to_string()
                            .trim_start_matches("sha256:")
                    ),
                    serde_json::json!({"reference": reference, "bytes": bytes}),
                )
            })
            .collect()
    }
}
