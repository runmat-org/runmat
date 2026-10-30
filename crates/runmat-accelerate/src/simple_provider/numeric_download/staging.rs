//! Synchronous registry readback inside an unpolled, finitely quoted future.
use super::*;
use runmat_accelerate_api::{NumericDownloadStageLease, PreparedNumericDownloadStage};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::{any::Any, sync::Arc};

#[derive(Clone)]
struct RegistryStage {
    device_id: u32,
}

impl NumericDownloadStageLease for RegistryStage {
    fn clone_for_callback(&self) -> Box<dyn NumericDownloadStageLease> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl InProcessProvider {
    pub(in crate::simple_provider) fn prepare_registry_numeric_download(
        &self,
        source: &GpuTensorHandle,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
        claim: Arc<dyn RetainedHostAllocationClaim>,
    ) -> Result<PreparedNumericDownloadStage> {
        ensure!(
            self.native_numeric_download_feasibility(source, query)
                .is_supported(),
            "in-process numeric download source/query is not admitted"
        );
        // There is no provider queue, mapped buffer or callback here. The
        // certificate records the selected owner while the borrowing future
        // retains its source until its one synchronous registry clone.
        Ok(PreparedNumericDownloadStage::issued(
            Box::new(RegistryStage {
                device_id: self.device_id,
            }),
            claim,
            source,
            0,
            0,
        ))
    }

    pub(in crate::simple_provider) fn registry_numeric_download_completion_bound(
        &self,
        source: &GpuTensorHandle,
    ) -> runmat_execution::ResourceByteBound {
        // Constructing this unpolled future allocates nothing and performs no
        // registry access. Its concrete layout includes the optional admitted
        // stage and funding handles; the only other allocation is its lease Box.
        let future = self.download_registry_numeric(source, None, None);
        std::mem::size_of_val(&future)
            .checked_add(std::mem::size_of::<RegistryStage>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .map(runmat_execution::ResourceByteBound::Known)
            .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
                runmat_execution::ResourceBoundUnavailable::NoScratchBound,
            ))
    }

    pub(in crate::simple_provider) async fn download_registry_numeric(
        &self,
        source: &GpuTensorHandle,
        funding: Option<Arc<dyn RetainedHostAllocationClaim>>,
        stage: Option<PreparedNumericDownloadStage>,
    ) -> Result<HostNumericTensorOwned> {
        ensure!(
            source.device_id() == self.device_id,
            "in-process numeric download requires its original provider"
        );
        source.allocation_lease().validate_live()?;
        source.view().validate(source.physical())?;
        if let Some(stage) = &stage {
            ensure!(
                stage.matches_source(source)
                    && stage.provider_scratch_bytes() == 0
                    && stage.queue_slots() == 0
                    && stage
                        .lease_as_any()
                        .downcast_ref::<RegistryStage>()
                        .is_some_and(|owner| owner.device_id == self.device_id),
                "in-process numeric download stage differs from its exact source/owner"
            );
        }
        let data = {
            let registry = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("in-process allocation owner is poisoned"))?;
            registry
                .get(source)
                .ok_or_else(|| anyhow!("numeric buffer not found: {}", source.buffer_id()))?
                .clone()
        };
        self.telemetry.record_download_bytes(data.byte_len());
        let output = HostNumericTensorOwned {
            data: data.into_host_numeric(),
            shape: source.shape().to_vec(),
            storage: runmat_accelerate_api::handle_storage(source),
        };
        output.validate()?;
        // All readback effects and stage owners retire before Ready. There is
        // no asynchronous callback that can outlive a cancelled waiter. The
        // runtime retains its separate frame-funding clone through Box Drop.
        drop(stage);
        drop(funding);
        Ok(output)
    }
}
