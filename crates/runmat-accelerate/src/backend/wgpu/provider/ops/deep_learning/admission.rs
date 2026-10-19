//! Pure physical capability queries for planned feed-forward stages.

use super::*;
use runmat_accelerate_api::{
    DeepLearningFeedForwardCapability as C, DeepLearningFeedForwardStage as S,
};

fn stage_extents(stage: S) -> ([usize; 3], usize) {
    match stage {
        S::Transpose2D { rows, columns } | S::Permute2D { rows, columns } => {
            ([rows, columns, 0], 2)
        }
        S::Dense {
            batch,
            input_channels,
            output_channels,
        } => ([batch, input_channels, output_channels], 3),
        S::Relu { elements } | S::Elu { elements } => ([elements, 0, 0], 1),
        S::SoftmaxRows { rows, columns } => ([rows, columns, 0], 2),
    }
}

impl WgpuProvider {
    pub(crate) fn deep_learning_feed_forward_stage_capability_exec(
        &self,
        anchor: &GpuTensorHandle,
        stage: S,
    ) -> Result<C> {
        let entry = self.get_entry_raw(anchor)?;
        ensure!(
            entry.element_type == anchor.physical().element_type()
                && entry.storage == anchor.physical().storage()
                && entry.len
                    == anchor
                        .shape()
                        .iter()
                        .try_fold(1usize, |count, extent| { count.checked_mul(*extent) })
                        .ok_or_else(|| anyhow!("feed-forward anchor shape overflows"))?
                && entry.logical == runmat_accelerate_api::handle_is_logical(anchor)
                && entry.integer_type() == runmat_accelerate_api::handle_integer_type(anchor),
            "feed-forward anchor differs from provider-owned descriptor"
        );
        if entry.storage != GpuTensorStorage::Real
            || !input_admission::classify_native_real_input(self, anchor, &entry)?.is_ready()
        {
            return Ok(C::UnsupportedOperation);
        }
        Ok(self.deep_learning_feed_forward_stage_capability_for_type(entry.element_type, stage))
    }

    pub(crate) fn deep_learning_feed_forward_stage_capability_for_type(
        &self,
        element_type: runmat_accelerate_api::NumericElementType,
        stage: S,
    ) -> C {
        if !self.supports_native_real_element_type(element_type) {
            return C::UnsupportedOperation;
        }
        let (extents, count) = stage_extents(stage);
        if extents[..count]
            .iter()
            .any(|extent| *extent > u32::MAX as usize)
        {
            return C::ResourceLimit;
        }
        let (input_count, output_count) = match stage {
            S::Transpose2D { rows, columns }
            | S::Permute2D { rows, columns }
            | S::SoftmaxRows { rows, columns } => {
                let Some(elements) = rows.checked_mul(columns) else {
                    return C::ResourceLimit;
                };
                (elements, elements)
            }
            S::Dense {
                batch,
                input_channels,
                output_channels,
            } => {
                let Some(input) = batch.checked_mul(input_channels) else {
                    return C::ResourceLimit;
                };
                let Some(output) = batch.checked_mul(output_channels) else {
                    return C::ResourceLimit;
                };
                let Some(weights) = input_channels.checked_mul(output_channels) else {
                    return C::ResourceLimit;
                };
                if !self.within_binding_limit(weights) {
                    return C::ResourceLimit;
                }
                (input, output)
            }
            S::Relu { elements } | S::Elu { elements } => (elements, elements),
        };
        if input_count > u32::MAX as usize || output_count > u32::MAX as usize {
            return C::ResourceLimit;
        }
        if !self.within_binding_limit(input_count) || !self.within_binding_limit(output_count) {
            return C::ResourceLimit;
        }
        if let S::SoftmaxRows { rows, columns } = stage {
            if rows != 0 && columns != 0 {
                let Some(bytes) = u64::try_from(input_count)
                    .ok()
                    .and_then(|count| count.checked_mul(self.element_size as u64))
                else {
                    return C::ResourceLimit;
                };
                if softmax_validated::check_softmax_limits(
                    &self.adapter_limits,
                    bytes,
                    bytes,
                    self.workgroup_config.scalar,
                    self.element_size,
                )
                .is_err()
                {
                    return C::ResourceLimit;
                }
            }
        }
        C::Ready
    }

    pub(crate) fn within_binding_limit(&self, elements: usize) -> bool {
        let limit = u64::from(self.adapter_limits.max_storage_buffer_binding_size)
            .min(self.adapter_limits.max_buffer_size);
        u64::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(self.element_size as u64))
            .is_some_and(|bytes| bytes <= limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{AccelProvider, HostTensorView};

    #[test]
    #[ignore = "requires actual WGPU device; run explicitly for pure planned-stage admission"]
    fn planned_dense_and_activations_do_not_dispatch_or_allocate() {
        let provider = WgpuProvider::new(Default::default()).expect("actual WGPU device required");
        let anchor = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2, 1],
            })
            .unwrap();
        provider.reset_telemetry();
        let before = provider.telemetry_snapshot();
        for stage in [
            S::Transpose2D {
                rows: 2,
                columns: 1,
            },
            S::Permute2D {
                rows: 2,
                columns: 1,
            },
            S::Dense {
                batch: 4,
                input_channels: 3,
                output_channels: 5,
            },
            S::Relu { elements: 20 },
            S::Elu { elements: 20 },
            S::SoftmaxRows {
                rows: 4,
                columns: 5,
            },
        ] {
            assert_eq!(
                provider
                    .deep_learning_feed_forward_stage_capability_exec(&anchor, stage)
                    .unwrap(),
                C::Ready
            );
        }
        if let Ok(too_large) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert_eq!(
                provider
                    .deep_learning_feed_forward_stage_capability_exec(
                        &anchor,
                        S::Relu {
                            elements: too_large,
                        },
                    )
                    .unwrap(),
                C::ResourceLimit
            );
        }
        let telemetry = provider.telemetry_snapshot();
        assert_eq!(
            telemetry, before,
            "capability queries must have no provider effects"
        );
        anchor.release().unwrap();
    }
}
