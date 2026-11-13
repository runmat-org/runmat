use std::sync::atomic::{AtomicU8, Ordering};

use runmat_accelerate_api::SparseProviderLossReason;

const HEALTHY: u8 = 0;
const UNKNOWN: u8 = 1;
const DESTROYED: u8 = 2;

#[derive(Default)]
pub(in crate::backend::wgpu::provider) struct WgpuDeviceHealth {
    loss_reason: AtomicU8,
}

impl WgpuDeviceHealth {
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn mark_lost(&self, reason: SparseProviderLossReason) {
        let _ = self.loss_reason.compare_exchange(
            HEALTHY,
            encode(reason),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub(super) fn loss_reason(&self) -> Option<SparseProviderLossReason> {
        match self.loss_reason.load(Ordering::Acquire) {
            HEALTHY => None,
            UNKNOWN => Some(SparseProviderLossReason::Unknown),
            DESTROYED => Some(SparseProviderLossReason::Destroyed),
            _ => Some(SparseProviderLossReason::Unknown),
        }
    }

    #[cfg(test)]
    pub(super) fn inject_loss(&self, reason: SparseProviderLossReason) -> TestDeviceLossGuard<'_> {
        let previous = self.loss_reason.swap(encode(reason), Ordering::AcqRel);
        TestDeviceLossGuard {
            health: self,
            previous,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn encode(reason: SparseProviderLossReason) -> u8 {
    match reason {
        SparseProviderLossReason::Unknown => UNKNOWN,
        SparseProviderLossReason::Destroyed => DESTROYED,
    }
}

#[cfg(test)]
pub(super) struct TestDeviceLossGuard<'a> {
    health: &'a WgpuDeviceHealth,
    previous: u8,
}

#[cfg(test)]
impl Drop for TestDeviceLossGuard<'_> {
    fn drop(&mut self) {
        self.health
            .loss_reason
            .store(self.previous, Ordering::Release);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn loss_reason(reason: wgpu::DeviceLostReason) -> SparseProviderLossReason {
    match reason {
        wgpu::DeviceLostReason::Unknown => SparseProviderLossReason::Unknown,
        wgpu::DeviceLostReason::Destroyed => SparseProviderLossReason::Destroyed,
    }
}

#[cfg(test)]
mod tests {
    use super::WgpuDeviceHealth;
    use runmat_accelerate_api::SparseProviderLossReason;

    #[test]
    fn device_health_preserves_the_first_typed_loss_reason() {
        let health = WgpuDeviceHealth::default();
        assert_eq!(health.loss_reason(), None);
        health.mark_lost(SparseProviderLossReason::Destroyed);
        health.mark_lost(SparseProviderLossReason::Unknown);
        assert_eq!(
            health.loss_reason(),
            Some(SparseProviderLossReason::Destroyed)
        );
    }
}
