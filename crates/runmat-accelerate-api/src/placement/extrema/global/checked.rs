//! Actual scalar probe evidence; proposals and wire metadata cannot mint it.
use super::*;
use crate::{allocation::TensorViewIdentity, HostNumericDataOwned, NumericElementType};

pub struct ProviderCheckedGlobalExtrema {
    owner: &'static dyn AccelProvider,
    source: GpuTensorHandle,
    kind: ProviderExtremaKind,
    recipe: runmat_execution::Digest,
    executed: ProviderExecutedGlobalExtrema,
}

pub struct ProviderGlobalExtremaScalar {
    source: TensorViewIdentity,
    kind: ProviderExtremaKind,
    recipe: runmat_execution::Digest,
    value: crate::scalar_program::ScalarLiteral,
}

impl ProviderGlobalExtremaScalar {
    pub fn source(&self) -> &TensorViewIdentity {
        &self.source
    }
    pub fn kind(&self) -> ProviderExtremaKind {
        self.kind
    }
    pub fn recipe_identity(&self) -> runmat_execution::Digest {
        self.recipe
    }
    pub fn value(&self) -> crate::scalar_program::ScalarLiteral {
        self.value
    }
}

impl ProviderCheckedGlobalExtrema {
    pub(super) fn new(
        recipe: &ProviderGlobalExtremaRecipe,
        executed: ProviderExecutedGlobalExtrema,
    ) -> Self {
        Self {
            owner: recipe.owner,
            source: recipe.source.clone(),
            kind: recipe.request.kind,
            recipe: recipe.identity().expect("global recipe serializes"),
            executed,
        }
    }

    pub async fn read_scalar(self) -> ProviderExecutionResult<ProviderGlobalExtremaScalar> {
        let result = self.read().await;
        // Ordinary completion reports retirement failure before a scalar may
        // become domain evidence. Abandonment still drops the private lease;
        // the allocation owner persists its once-only retirement outcome.
        if self.executed.output.shares_allocation_with(&self.source) {
            return result;
        }
        let cleanup = self
            .executed
            .output
            .release()
            .err()
            .into_iter()
            .collect::<Vec<_>>();
        if cleanup.is_empty() {
            result
        } else {
            Err(
                ProviderExecutionError::from_source(ProviderExecutionCompletionError::new(
                    result.err(),
                    cleanup,
                ))
                .after_effect(),
            )
        }
    }

    async fn read(&self) -> ProviderExecutionResult<ProviderGlobalExtremaScalar> {
        let exact_owner = |handle: &GpuTensorHandle| {
            crate::provider_for_handle(handle)
                .is_some_and(|actual| crate::same_provider_instance(actual, self.owner))
        };
        if !exact_owner(&self.source) || !exact_owner(&self.executed.output) {
            return Err(ProviderExecutionError::from(anyhow::anyhow!(
                "global scalar readback owner changed"
            )));
        }
        self.source
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        let output = &self.executed.output;
        output
            .allocation_lease()
            .validate_live()
            .map_err(ProviderExecutionError::from_source)?;
        let metadata = crate::ResidencyRestoreGuard::new(output);
        let host = self
            .owner
            .download_numeric(output)
            .await
            .map_err(|cause| ProviderExecutionError::from(cause).after_effect())?;
        metadata.restore();
        drop(metadata);
        let malformed = || {
            ProviderExecutionError::from(anyhow::anyhow!(
                "global extrema scalar readback changed native representation"
            ))
            .after_effect()
        };
        if host.shape != [1, 1]
            || host.storage != GpuTensorStorage::Real
            || self.source.allocation_lease().validate_live().is_err()
            || output.allocation_lease().validate_live().is_err()
            || !exact_owner(&self.source)
            || !exact_owner(output)
        {
            return Err(malformed());
        }
        let value = match (output.physical().element_type(), host.data) {
            (NumericElementType::F32, HostNumericDataOwned::F32(values)) if values.len() == 1 => {
                crate::scalar_program::ScalarLiteral::F32Bits(values[0].to_bits())
            }
            (NumericElementType::F64, HostNumericDataOwned::F64(values)) if values.len() == 1 => {
                crate::scalar_program::ScalarLiteral::F64Bits(values[0].to_bits())
            }
            _ => return Err(malformed()),
        };
        Ok(ProviderGlobalExtremaScalar {
            source: self.source.view_identity(),
            kind: self.kind,
            recipe: self.recipe,
            value,
        })
    }
}
