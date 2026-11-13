use runmat_execution::resource::{
    ExactShared, ProviderHostMetadataAdmission, RetainedHostAllocationClaim,
};
use std::sync::Arc;

struct Payload<T> {
    pipeline: T,
    // ExactShared deallocates the physical header before this payload drops;
    // its claim also survives the pipeline wrapper's own destruction.
    _claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

pub(crate) struct PipelineOwner<T = wgpu::ComputePipeline>(ExactShared<Payload<T>>);

impl<T> Clone for PipelineOwner<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> std::ops::Deref for PipelineOwner<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.get().pipeline
    }
}
impl<T> AsRef<T> for PipelineOwner<T> {
    fn as_ref(&self) -> &T {
        self
    }
}
impl<T> PipelineOwner<T> {
    #[cfg(test)]
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
    pub(crate) fn unbounded(pipeline: T) -> Self {
        Self(ExactShared::new(Payload {
            pipeline,
            _claim: None,
        }))
    }
}
impl<T: std::fmt::Debug> std::fmt::Debug for PipelineOwner<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PipelineOwner").field(self.as_ref()).finish()
    }
}

/// Stack-only pre-effect admission, consumed by the exact wrapper allocation.
pub(crate) struct PipelineAllocation<T = wgpu::ComputePipeline> {
    claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    marker: std::marker::PhantomData<T>,
}
impl<T> PipelineAllocation<T> {
    pub(crate) fn requested_bytes() -> anyhow::Result<u64> {
        Ok(u64::try_from(ExactShared::<Payload<T>>::requested_bytes())?)
    }
    pub(crate) fn prepare(
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> anyhow::Result<Self> {
        let bytes = Self::requested_bytes()?;
        let claim = metadata
            .map(|metadata| metadata.claim_record(bytes))
            .transpose()?;
        Ok(Self {
            claim,
            marker: std::marker::PhantomData,
        })
    }
    pub(crate) fn allocate(self, pipeline: T) -> anyhow::Result<PipelineOwner<T>> {
        ExactShared::try_new(Payload {
            pipeline,
            _claim: self.claim,
        })
        .map(PipelineOwner)
        .map_err(|payload| {
            drop(payload);
            anyhow::anyhow!("fused pipeline wrapper allocation failed")
        })
    }
}
