//! Exact-owner exports used by one rendering preparation stage.
use super::{plotting_error, preparation_after_effect, BuiltinResult, GpuTensorHandle};

/// One rendering-preparation export stage. A materialized view is an effect;
/// a later capability miss cannot silently restart preparation on the host.
#[derive(Default)]
pub(crate) struct GpuBufferExports {
    materialized: bool,
    unavailable: bool,
    retained: Vec<(GpuTensorHandle, runmat_accelerate_api::WgpuBufferRef)>,
}

impl GpuBufferExports {
    pub(crate) fn element_count(
        input: &GpuTensorHandle,
        name: &'static str,
    ) -> BuiltinResult<usize> {
        if input.shape().contains(&0) {
            return Ok(0);
        }
        input
            .shape()
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or_else(|| plotting_error(name, format!("{name}: GPU input shape overflow")))
    }

    pub(crate) fn complete<T>(
        &self,
        result: BuiltinResult<T>,
        name: &'static str,
    ) -> BuiltinResult<T> {
        result.map_err(|error| {
            if self.materialized {
                preparation_after_effect(error, name)
            } else {
                error
            }
        })
    }

    pub(crate) fn export(
        &mut self,
        input: &GpuTensorHandle,
        name: &'static str,
    ) -> BuiltinResult<Option<runmat_accelerate_api::WgpuBufferRef>> {
        let cached = self
            .retained
            .iter()
            .find(|(original, _)| {
                original.shares_allocation_with(input)
                    && original.view_identity() == input.view_identity()
            })
            .map(|(_, exported)| exported);
        // Tuple/optional-operand callers still evaluate later expressions;
        // don't begin another export after this stage has already declined.
        if self.unavailable && cached.is_none() {
            return Ok(None);
        }
        let result = match cached {
            Some(exported) => input
                .allocation_lease()
                .validate_live()
                .and_then(|()| exported.owning_handle().allocation_lease().validate_live())
                .map(|()| Some(exported.clone()))
                .map_err(runmat_accelerate_api::ProviderExecutionError::from_source),
            None => runmat_accelerate_api::export_wgpu_buffer(input),
        }
        .map_err(|error| {
            let error = if self.materialized {
                error.after_effect()
            } else {
                error
            };
            crate::runtime_error::with_provider_execution_source(
                plotting_error(name, format!("{name}: GPU buffer export failed: {error}")),
                error,
            )
        })?;
        let Some(output) = result else {
            if self.materialized {
                return Err(crate::runtime_error::with_provider_execution_source(
                    plotting_error(
                        name,
                        format!("{name}: GPU buffer export unavailable after materialization"),
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from(
                        runmat_accelerate_api::unsupported_provider_operation(
                            "rendering export unavailable after an earlier materialized view",
                        ),
                    )
                    .after_effect(),
                ));
            }
            self.unavailable = true;
            return Ok(None);
        };
        self.materialized |= !output.owning_handle().shares_allocation_with(input);
        let valid = output.shape == input.shape()
            && runmat_accelerate_api::handle_precision(input) == Some(output.precision)
            && output.owning_handle().device_id() == input.device_id()
            && runmat_accelerate_api::handle_is_logical(output.owning_handle())
                == runmat_accelerate_api::handle_is_logical(input);
        if !valid {
            return Err(crate::runtime_error::with_provider_execution_source(
                plotting_error(name, format!("{name}: invalid GPU buffer export")),
                runmat_accelerate_api::ProviderExecutionError::from(anyhow::anyhow!(
                    "GPU export does not match the original rendering input"
                ))
                .after_effect(),
            ));
        }
        if cached.is_none() {
            self.retained.push((input.clone(), output.clone()));
        }
        Ok(Some(output))
    }
}
