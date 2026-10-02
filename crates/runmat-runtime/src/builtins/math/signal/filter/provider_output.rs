//! Admit a provider's output and state as one filtering result.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderIirFilterResult,
};

use crate::builtins::common::gpu_helpers;
use crate::builtins::math::signal::resident_output::{self, OutputContext};
use crate::{BuiltinResult, GpuGatherRetry, RuntimeError};

use super::{filter_error, FILTER_ERROR_PROVIDER_OUTPUT};

fn invalid_output() -> RuntimeError {
    let mut error = filter_error(&FILTER_ERROR_PROVIDER_OUTPUT);
    error.gpu_gather_retry = GpuGatherRetry::Never;
    error
}

pub(super) struct ProviderOutputContext<'a> {
    pub provider: &'static dyn AccelProvider,
    pub signal: &'a GpuTensorHandle,
    pub numerator: &'a GpuTensorHandle,
    pub denominator: &'a GpuTensorHandle,
    pub original_numerator: Option<&'a GpuTensorHandle>,
    pub original_denominator: Option<&'a GpuTensorHandle>,
    pub initial_state: Option<&'a GpuTensorHandle>,
    pub state_shape: &'a [usize],
    pub state_len: usize,
}

impl ProviderOutputContext<'_> {
    pub(super) fn admit(
        self,
        result: ProviderIirFilterResult,
    ) -> BuiltinResult<(GpuTensorHandle, Option<GpuTensorHandle>)> {
        let ProviderIirFilterResult {
            output,
            final_state,
        } = result;
        let mut borrowed = vec![self.numerator, self.denominator];
        if let Some(original) = self.original_numerator {
            borrowed.push(original);
        }
        if let Some(original) = self.original_denominator {
            borrowed.push(original);
        }
        if let Some(initial_state) = self.initial_state {
            borrowed.push(initial_state);
        }
        let context = OutputContext {
            input: self.signal,
            provider: self.provider,
            caller_inputs: &borrowed,
        };
        // Keep every caller reference live across even this early rejection.
        // reject_gpu_output releases only the result's owning reference; an
        // aliased borrowed input cannot be physically retired here.
        if self.state_len > 0 && final_state.is_none() {
            return Err(gpu_helpers::reject_gpu_output(output, invalid_output()));
        }
        let output_contract = resident_output::floating_contract(
            self.signal,
            self.signal.shape().to_vec(),
            GpuTensorStorage::Real,
        );
        let output = match final_state {
            Some(state) => {
                let output =
                    match context.admit(output, &output_contract, &[&state], invalid_output) {
                        Ok(output) => output,
                        Err(error) => return Err(gpu_helpers::reject_gpu_output(state, error)),
                    };
                let state_contract = resident_output::floating_contract(
                    self.signal,
                    self.state_shape.to_vec(),
                    GpuTensorStorage::Real,
                );
                let state = match context.admit(state, &state_contract, &[&output], invalid_output)
                {
                    Ok(state) => state,
                    Err(error) => return Err(gpu_helpers::reject_gpu_output(output, error)),
                };
                return Ok((output, Some(state)));
            }
            None => output,
        };
        context
            .admit(output, &output_contract, &[], invalid_output)
            .map(|output| (output, None))
    }
}

#[cfg(test)]
#[path = "provider_output/tests.rs"]
mod tests;
