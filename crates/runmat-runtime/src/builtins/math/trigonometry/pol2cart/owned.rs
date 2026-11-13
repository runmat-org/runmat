use super::*;
use runmat_accelerate_api::ProviderPrecision;

#[cfg(test)]
#[path = "owned_tests.rs"]
mod tests;

/// Fresh transform temporaries only. Borrowed inputs are never adopted, even
/// when a malformed provider returns one as an output. Drop handles cancellation.
pub(super) struct Outputs {
    borrowed: Vec<GpuTensorHandle>,
    owned: Vec<GpuTensorHandle>,
    provenance: Option<runmat_accelerate_api::GpuHandleProvenance>,
}

impl Outputs {
    pub(super) fn new(inputs: &[Value]) -> Self {
        Self {
            borrowed: inputs
                .iter()
                .filter_map(|v| match v {
                    Value::GpuTensor(h) => Some(h.clone()),
                    _ => None,
                })
                .collect(),
            owned: Vec::new(),
            provenance: inputs
                .iter()
                .filter_map(|value| match value {
                    Value::GpuTensor(handle) => runmat_accelerate_api::handle_provenance(handle),
                    _ => None,
                })
                .reduce(|left, right| {
                    use runmat_accelerate_api::GpuHandleProvenance::Explicit;
                    if left == Explicit || right == Explicit {
                        Explicit
                    } else {
                        left
                    }
                }),
        }
    }

    pub(super) async fn run_protected<T, E: Into<runmat_accelerate_api::ProviderExecutionError>>(
        &self,
        future: impl std::future::Future<Output = Result<T, E>>,
    ) -> runmat_accelerate_api::ProviderExecutionResult<T> {
        let guards: Vec<_> = self
            .protected()
            .into_iter()
            .map(runmat_accelerate_api::ResidencyRestoreGuard::new)
            .collect();
        let result = future.await.map_err(|error| {
            let error = error.into();
            if self.owned.is_empty() {
                error
            } else {
                error.after_effect()
            }
        });
        drop(guards);
        result
    }

    pub(super) fn protected(&self) -> Vec<&GpuTensorHandle> {
        self.borrowed.iter().chain(self.owned.iter()).collect()
    }

    pub(super) fn adopt(
        &mut self,
        provider: &'static dyn AccelProvider,
        mut output: GpuTensorHandle,
        anchor: &GpuTensorHandle,
        shape: &[usize],
        precision: ProviderPrecision,
    ) -> BuiltinResult<GpuTensorHandle> {
        let protected = self.protected();
        let contract = gpu_helpers::BinaryGpuOutputContract {
            shape: shape.to_vec(),
            storage: GpuTensorStorage::Real,
            precision: Some(precision),
            integer: None,
            logical: false,
            alias: gpu_helpers::GpuOutputAliasPolicy::RequireDistinctAllocation,
        };
        let valid = !protected.iter().any(|h| h.shares_allocation_with(&output))
            && gpu_helpers::binary_gpu_output_matches(&output, anchor, anchor, provider, &contract)
            && gpu_helpers::expected_handle_numeric_element_type(&output).is_ok_and(|element| {
                gpu_helpers::numeric_descriptor_matches(&output, element, GpuTensorStorage::Real)
            });
        if !valid {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                pol2cart_error(
                    &ERROR_INTERNAL,
                    "provider returned an invalid or aliased coordinate result",
                ),
            ));
        }
        if let Some(provenance) = self.provenance {
            output = output.with_provenance(provenance).map_err(|error| {
                build_runtime_error(format!("pol2cart: cannot preserve input intent: {error}"))
                    .with_builtin("pol2cart")
                    .with_source(error)
                    .build()
            })?;
        }
        self.owned.push(output.clone());
        Ok(output)
    }

    pub(super) fn publish(&mut self, handle: GpuTensorHandle) -> BuiltinResult<Value> {
        self.owned
            .retain(|owned| !gpu_helpers::same_gpu_handle(owned, &handle));
        gpu_helpers::resident_gpu_value(handle)
    }
}

// Owning handles retain their exact allocation authority across suspension and
// thread-provider changes. Vector drop retires only its own references.
