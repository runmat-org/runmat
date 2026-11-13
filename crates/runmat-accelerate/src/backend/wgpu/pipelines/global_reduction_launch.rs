//! Constructor-frozen launch geometry for the global two-elements-per-lane
//! reduction tree. The shader tile, chunk stride and dispatch share this one
//! typed selection, not a process-global scalar workgroup override.

use crate::backend::wgpu::types::NumericPrecision;
use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrozenGlobalReductionLaunch {
    workgroup: NonZeroU32,
    max_groups: NonZeroU32,
}

impl FrozenGlobalReductionLaunch {
    pub(crate) fn new(
        workgroup: u32,
        precision: NumericPrecision,
        enabled: &wgpu::Limits,
    ) -> Option<Self> {
        let workgroup = NonZeroU32::new(workgroup)?;
        if !workgroup.get().is_power_of_two()
            || workgroup.get() > enabled.max_compute_workgroup_size_x
            || workgroup.get() > enabled.max_compute_invocations_per_workgroup
        {
            return None;
        }
        let element_bytes = match precision {
            NumericPrecision::F32 => 4,
            NumericPrecision::F64 => 8,
        };
        if workgroup.get().checked_mul(element_bytes)? > enabled.max_compute_workgroup_storage_size
        {
            return None;
        }
        let max_groups = NonZeroU32::new(
            enabled
                .max_compute_workgroups_per_dimension
                .min(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS),
        )?;
        Some(Self {
            workgroup,
            max_groups,
        })
    }

    pub(crate) fn workgroup(self) -> u32 {
        self.workgroup.get()
    }

    pub(crate) fn elements_per_group(self) -> usize {
        self.workgroup.get() as usize * 2
    }

    pub(crate) fn max_input_per_pass(self) -> Option<usize> {
        self.elements_per_group()
            .checked_mul(self.max_groups.get() as usize)
    }

    pub(crate) fn output_len(self, input_len: usize) -> usize {
        input_len.div_ceil(self.elements_per_group())
    }

    pub(crate) fn dispatch_groups(self, input_len: usize) -> Option<u32> {
        let groups = self.output_len(input_len);
        (groups <= self.max_groups.get() as usize)
            .then(|| u32::try_from(groups).ok())
            .flatten()
    }
}

#[cfg(test)]
#[path = "global_reduction_launch/tests.rs"]
mod tests;
