//! The existing physical constructor owns both pure absence and allocation phase.
use super::*;
use runmat_accelerate_api::{ProviderExecutionError, ProviderExecutionResult};
use std::cell::Cell;

fn execute_constructor(
    action: impl FnOnce(&Cell<bool>) -> Result<GpuTensorHandle>,
) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
    let effect = Cell::new(false);
    action(&effect).map(Some).map_err(|source| {
        let source = ProviderExecutionError::from(source);
        if effect.get() {
            source.after_effect()
        } else {
            source.with_unproven_effects()
        }
    })
}

impl WgpuProvider {
    pub(crate) fn fill_constructor(
        &self,
        shape: &[usize],
        value: f64,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        execute_constructor(|effect| {
            self.fill_produced_exec(shape, value, false, None, Some(effect), None)
        })
    }

    pub(crate) fn logical_fill_constructor(
        &self,
        shape: &[usize],
        value: bool,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        execute_constructor(|effect| {
            self.fill_produced_exec(
                shape,
                if value { 1.0 } else { 0.0 },
                true,
                None,
                Some(effect),
                None,
            )
        })
    }

    pub(crate) fn zeros_like_constructor(
        &self,
        prototype: &GpuTensorHandle,
        shape: &[usize],
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        execute_constructor(|effect| self.zeros_like_exec(prototype, shape, effect))
    }

    pub(crate) fn fill_like_constructor(
        &self,
        prototype: &GpuTensorHandle,
        value: f64,
    ) -> ProviderExecutionResult<Option<GpuTensorHandle>> {
        if value.to_bits() == 0 {
            return self.zeros_like_constructor(prototype, prototype.shape());
        }
        let entry = self
            .get_entry_raw(prototype)
            .map_err(|source| ProviderExecutionError::from(source).with_unproven_effects())?;
        let expected = match self.precision {
            NumericPrecision::F32 => runmat_accelerate_api::NumericElementType::F32,
            NumericPrecision::F64 => runmat_accelerate_api::NumericElementType::F64,
        };
        if entry.element_type != expected
            || entry.storage != GpuTensorStorage::Real
            || prototype.view().is_logical()
        {
            return Ok(None);
        }
        self.fill_constructor(prototype.shape(), value)
    }
}
