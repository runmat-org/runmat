//! Exact-owner exports used by one rendering preparation stage.
use super::{plotting_error, preparation_failure, BuiltinResult, GpuTensorHandle};

/// One rendering-preparation export stage. A materialized view or completed numerical stage is an effect;
/// a later capability miss cannot silently restart preparation on the host.
#[derive(Default)]
pub(crate) struct GpuBufferExports {
    completed_effect: bool,
    unavailable: bool,
    retained: Vec<(GpuTensorHandle, runmat_accelerate_api::WgpuBufferRef)>,
    access: Option<runmat_accelerate_api::WgpuDeviceAccess>,
}

impl GpuBufferExports {
    pub(crate) fn has_completed_effect(&self) -> bool {
        self.completed_effect
    }
    /// All actual live inputs, including independently materialized views,
    /// remain borrowed operands of subsequent reduction/packing stages.
    pub(crate) fn protected_handles(&self) -> Vec<&GpuTensorHandle> {
        self.retained
            .iter()
            .flat_map(|(input, output)| [input, output.owning_handle()])
            .collect()
    }
    /// Called only by the owning stage after actual reduction/packing success.
    pub(crate) fn record_completed_effect(&mut self) {
        self.completed_effect = true;
    }
    /// One shared physical admission and retained export stage. Numerical
    /// forms, geometry and optional rendering policy remain caller-owned.
    pub(crate) fn prepare(
        inputs: &[&GpuTensorHandle],
        name: &'static str,
    ) -> BuiltinResult<Option<(runmat_plot::SharedWgpuContext, Self)>> {
        let Some(context) = super::prepare_shared_wgpu_context(inputs, name)? else {
            return Ok(None);
        };
        // These are the existing WgpuBufferRef floating-real carrier limits,
        // not a new builtin support policy. Integer/logical rendering stays host.
        if inputs.iter().any(|input| {
            runmat_accelerate_api::handle_precision(input).is_none()
                || input.physical().storage() != runmat_accelerate_api::GpuTensorStorage::Real
                || runmat_accelerate_api::handle_is_logical(input)
        }) {
            return Ok(None);
        }
        let mut exports = Self::for_context(&context);
        for input in inputs {
            if exports.export(input, name)?.is_none() {
                return Ok(None);
            }
        }
        Ok(Some((context, exports)))
    }
    pub(crate) fn for_context(context: &runmat_plot::SharedWgpuContext) -> Self {
        Self {
            access: Some(context.access.clone()),
            ..Self::default()
        }
    }
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
        result.map_err(|error| preparation_failure(error, name, self.completed_effect))
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
            let error = if self.completed_effect {
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
            if self.completed_effect {
                return Err(crate::runtime_error::with_provider_execution_source(
                    plotting_error(
                        name,
                        format!("{name}: GPU buffer export unavailable after completed preparation"),
                    ),
                    runmat_accelerate_api::ProviderExecutionError::from(
                        runmat_accelerate_api::unsupported_provider_operation(
                            "rendering export unavailable after an earlier completed preparation stage",
                        ),
                    )
                    .after_effect(),
                ));
            }
            self.unavailable = true;
            return Ok(None);
        };
        self.completed_effect |= !output.owning_handle().shares_allocation_with(input);
        let valid = output.shape == input.shape()
            && self
                .access
                .as_ref()
                .is_none_or(|access| access.same_device(output.access()))
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
