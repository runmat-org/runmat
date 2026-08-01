use futures::future::LocalBoxFuture;
use runmat_package::{
    RegistryAcquisitionPlan, RegistryCandidatePlan, RegistryCandidateRecord,
    RegistryReleaseMetadata, RegistrySourceId,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryArtifactTransfer {
    pub package_id: String,
    pub source: RegistrySourceId,
    pub metadata: RegistryReleaseMetadata,
    pub artifact_bytes: Vec<u8>,
}

pub trait RegistryTransport: Send + Sync {
    fn candidates<'a>(
        &'a self,
        plan: &'a RegistryCandidatePlan,
    ) -> LocalBoxFuture<'a, Result<Vec<RegistryCandidateRecord>, String>>;

    fn fetch<'a>(
        &'a self,
        plan: &'a RegistryAcquisitionPlan,
    ) -> LocalBoxFuture<'a, Result<RegistryArtifactTransfer, String>>;
}
