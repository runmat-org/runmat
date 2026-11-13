//! Inline session device selection; the existing context poll guard activates
//! it. No selection lives in provider-global or thread-local mutable state.
use super::RuntimeContext;
use runmat_accelerate_api::{AccelProvider, DeviceAllocationScope};
mod reset;
pub(super) use reset::GpuDeviceResetState;

#[derive(Clone, Copy)]
pub(crate) enum GpuDeviceSelection {
    Inherit,
    Selected(&'static dyn AccelProvider),
    Deselected,
    Resetting(&'static dyn AccelProvider),
    ResetFailed(&'static dyn AccelProvider),
}
impl std::fmt::Debug for GpuDeviceSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inherit => f.write_str("Inherit"),
            Self::Deselected => f.write_str("Deselected"),
            Self::Selected(provider) => f
                .debug_tuple("Selected")
                .field(&provider.device_id())
                .finish(),
            Self::Resetting(provider) => f
                .debug_tuple("Resetting")
                .field(&provider.device_id())
                .finish(),
            Self::ResetFailed(provider) => f
                .debug_tuple("ResetFailed")
                .field(&provider.device_id())
                .finish(),
        }
    }
}
impl RuntimeContext {
    pub(crate) fn gpu_device_selection(&self) -> GpuDeviceSelection {
        self.state().gpu_device_selection.get()
    }
    pub(crate) fn gpu_device_allocation_scope(&self) -> Option<DeviceAllocationScope> {
        if matches!(
            self.gpu_device_selection(),
            GpuDeviceSelection::Resetting(_) | GpuDeviceSelection::ResetFailed(_)
        ) {
            return None;
        }
        DeviceAllocationScope::new(
            self.gpu_device_session_key(),
            self.state().gpu_device_epoch.get(),
        )
    }
    pub(crate) fn select_gpu_device_without_reset(
        &self,
        provider: &'static dyn AccelProvider,
    ) -> bool {
        if !runmat_accelerate_api::device_control_provider_is_authorized(provider)
            || matches!(
                self.gpu_device_selection(),
                GpuDeviceSelection::Resetting(_) | GpuDeviceSelection::ResetFailed(_)
            )
        {
            return false;
        }
        if let GpuDeviceSelection::Selected(selected) = self.gpu_device_selection() {
            if !runmat_accelerate_api::same_provider_instance(selected, provider) {
                return false;
            }
        }
        self.state()
            .gpu_device_selection
            .set(GpuDeviceSelection::Selected(provider));
        true
    }
    pub(crate) fn begin_gpu_device_reset(
        &self,
        provider: &'static dyn AccelProvider,
    ) -> Option<DeviceAllocationScope> {
        if !runmat_accelerate_api::device_control_provider_is_authorized(provider) {
            return None;
        }
        if matches!(
            self.gpu_device_selection(),
            GpuDeviceSelection::Resetting(_)
        ) {
            return None;
        }
        if let GpuDeviceSelection::ResetFailed(failed) = self.gpu_device_selection() {
            if !runmat_accelerate_api::same_provider_instance(failed, provider) {
                return None;
            }
        }
        let scope = DeviceAllocationScope::new(
            self.gpu_device_session_key(),
            self.state().gpu_device_epoch.get(),
        )?;
        scope.successor()?;
        self.state()
            .gpu_device_selection
            .set(GpuDeviceSelection::Resetting(provider));
        Some(scope)
    }
    pub(crate) fn finish_gpu_device_reset(
        &self,
        provider: &'static dyn AccelProvider,
        scope: DeviceAllocationScope,
        deselect: bool,
    ) -> bool {
        let GpuDeviceSelection::Resetting(active) = self.gpu_device_selection() else {
            return false;
        };
        if !runmat_accelerate_api::same_provider_instance(active, provider)
            || DeviceAllocationScope::new(
                self.gpu_device_session_key(),
                self.state().gpu_device_epoch.get(),
            ) != Some(scope)
        {
            return false;
        }
        let Some(next) = scope.successor() else {
            return false;
        };
        self.state().gpu_device_epoch.set(next.epoch());
        self.state().gpu_device_selection.set(if deselect {
            GpuDeviceSelection::Deselected
        } else {
            GpuDeviceSelection::Selected(provider)
        });
        true
    }
}
pub(super) fn current_device_allocation_scope() -> runmat_accelerate_api::DeviceAllocationAdmission
{
    use runmat_accelerate_api::DeviceAllocationAdmission;
    super::scope::with_active_runtime_context(|context| match context {
        None => DeviceAllocationAdmission::Unscoped,
        Some(context) => match context.gpu_device_selection() {
            GpuDeviceSelection::Deselected
            | GpuDeviceSelection::Resetting(_)
            | GpuDeviceSelection::ResetFailed(_) => DeviceAllocationAdmission::Fenced,
            _ => context.gpu_device_allocation_scope().map_or(
                DeviceAllocationAdmission::Fenced,
                DeviceAllocationAdmission::Scoped,
            ),
        },
    })
}
