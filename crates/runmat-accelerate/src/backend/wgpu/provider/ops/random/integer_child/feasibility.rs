//! Physical feasibility for fixed and bounded integer child work.

use super::*;

impl WgpuProvider {
    pub(crate) fn integer_child_feasibility(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> Option<ProviderFeasibility> {
        if query.operation.0 == PROVIDER_GPU_INTEGER_BATCH_OPERATION {
            return Some(self.integer_child_batch_feasibility(query));
        }
        let identity = query.operation.0 == PROVIDER_GPU_INTEGER_OPERATION;
        let Some(workload) = query.workload.gpu_integer else {
            return identity.then(|| {
                ProviderFeasibility::rejected(
                    ProviderRejectionCode::UnknownResourceEstimate,
                    "random.gpu_child.integer.missing_workload",
                )
            });
        };
        let reject = |code, detail| ProviderFeasibility::rejected(code, detail);
        if !identity
            || query.family != ProviderOperationFamily::Library
            || !query.inputs.is_empty()
            || query.outputs.len() != 1
        {
            return Some(reject(
                ProviderRejectionCode::InvalidShape,
                "random.gpu_child.integer.query",
            ));
        }
        let output = &query.outputs[0];
        if output.storage != ProviderStorage::DenseReal
            || output.layout != ProviderLayout::ColumnMajorContiguous
            || output.residency != ProviderResidency::Device
        {
            return Some(reject(
                ProviderRejectionCode::UnsupportedStorage,
                "random.gpu_child.integer.output",
            ));
        }
        let class = match output.element_type {
            ProviderElementType::F32 => NumericElementType::F32,
            ProviderElementType::F64 => NumericElementType::F64,
            ProviderElementType::I32 => NumericElementType::I32,
            ProviderElementType::U32 => NumericElementType::U32,
            _ => {
                return Some(reject(
                    ProviderRejectionCode::UnsupportedElementType,
                    "random.gpu_child.integer.class",
                ))
            }
        };
        let Some(count) = output
            .checked_element_count()
            .and_then(|count| usize::try_from(count).ok())
        else {
            return Some(reject(
                ProviderRejectionCode::InvalidShape,
                "random.gpu_child.integer.shape",
            ));
        };
        let shape = [count];
        let request = GpuIntegerRequest {
            shape: &shape,
            class,
            domain: workload.domain,
            reservation: workload.reservation,
        };
        if let Some(physical) = self.propose_integer_child_exec(&request) {
            return Some(ProviderFeasibility::supported(ProviderResourceEstimate {
                transient_bytes: physical.requested_incremental_peak_bytes(),
                output_bytes: Some(physical.output_buffer_bytes),
                dispatches: Some(physical.queue_submissions),
            }));
        }
        let Some(physical) = self.propose_integer_child_batch_exec(GpuIntegerBatchRequest {
            integer: request,
            first_attempt: 0,
            attempts_per_item: 8,
            pending: None,
        }) else {
            return Some(reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "random.gpu_child.integer.physical_route",
            ));
        };
        Some(ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: physical.peak_bytes(),
            output_bytes: Some(physical.output_buffer_bytes),
            dispatches: Some(physical.queue_submissions),
        }))
    }

    fn integer_child_batch_feasibility(
        &self,
        query: &ProviderFeasibilityQuery,
    ) -> ProviderFeasibility {
        let reject = |code, detail| ProviderFeasibility::rejected(code, detail);
        let (Some(workload), Some(batch)) =
            (query.workload.gpu_integer, query.workload.gpu_integer_batch)
        else {
            return reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "random.gpu_child.integer.batch.workload",
            );
        };
        if query.family != ProviderOperationFamily::Library
            || query.outputs.len() != 2
            || query.inputs.len() != if batch.pending { 2 } else { 0 }
        {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "random.gpu_child.integer.batch.query",
            );
        }
        let output = &query.outputs[0];
        let accepted = &query.outputs[1];
        if output.storage != ProviderStorage::DenseReal
            || output.layout != ProviderLayout::ColumnMajorContiguous
            || output.residency != ProviderResidency::Device
            || accepted.storage != ProviderStorage::DenseReal
            || accepted.layout != ProviderLayout::ColumnMajorContiguous
            || accepted.residency != ProviderResidency::Device
            || accepted.element_type != ProviderElementType::U32
            || accepted.shape != output.shape
            || (batch.pending && query.inputs != query.outputs)
        {
            return reject(
                ProviderRejectionCode::UnsupportedStorage,
                "random.gpu_child.integer.batch.representation",
            );
        }
        let class = match output.element_type {
            ProviderElementType::F32 => NumericElementType::F32,
            ProviderElementType::F64 => NumericElementType::F64,
            ProviderElementType::I32 => NumericElementType::I32,
            ProviderElementType::U32 => NumericElementType::U32,
            _ => {
                return reject(
                    ProviderRejectionCode::UnsupportedElementType,
                    "random.gpu_child.integer.batch.class",
                )
            }
        };
        let Some(count) = output
            .checked_element_count()
            .and_then(|count| usize::try_from(count).ok())
        else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "random.gpu_child.integer.batch.shape",
            );
        };
        let shape = [count];
        let request = GpuIntegerBatchRequest {
            integer: GpuIntegerRequest {
                shape: &shape,
                class,
                domain: workload.domain,
                reservation: workload.reservation,
            },
            first_attempt: batch.first_attempt,
            attempts_per_item: batch.attempts_per_item,
            pending: None,
        };
        let Some(physical) = self.propose_integer_child_batch_exec(request) else {
            return reject(
                ProviderRejectionCode::UnknownResourceEstimate,
                "random.gpu_child.integer.batch.physical",
            );
        };
        ProviderFeasibility::supported(ProviderResourceEstimate {
            transient_bytes: physical.peak_bytes(),
            output_bytes: physical.newly_retained_output_bytes(batch),
            dispatches: Some(physical.queue_submissions),
        })
    }
}
