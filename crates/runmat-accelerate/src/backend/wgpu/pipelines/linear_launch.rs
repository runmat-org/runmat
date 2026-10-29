//! Constructor-frozen launch geometry for one-dimensional WGPU shaders.
//! A pipeline opts into this contract explicitly; neither shader text nor a
//! process-global configuration is evidence of its launch domain.

use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrozenLinearLaunch {
    workgroup: NonZeroU32,
    max_dispatch_groups: NonZeroU32,
}

impl FrozenLinearLaunch {
    pub(crate) fn new(workgroup: u32, enabled_max_dispatch_groups: u32) -> Option<Self> {
        Some(Self {
            workgroup: NonZeroU32::new(workgroup)?,
            max_dispatch_groups: NonZeroU32::new(
                enabled_max_dispatch_groups
                    .min(crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS),
            )?,
        })
    }

    pub(crate) fn workgroup(self) -> u32 {
        self.workgroup.get()
    }

    pub(crate) fn chunk_capacity(self) -> Option<usize> {
        usize::try_from(
            u64::from(self.workgroup.get())
                .checked_mul(u64::from(self.max_dispatch_groups.get()))?,
        )
        .ok()
    }

    pub(crate) fn dispatch_groups(self, elements: usize) -> Option<u32> {
        let elements = u64::try_from(elements).ok()?;
        let groups = elements.div_ceil(u64::from(self.workgroup.get()));
        if groups > u64::from(self.max_dispatch_groups.get()) {
            return None;
        }
        u32::try_from(groups).ok()
    }
}

/// Token substitution is a compile-time selection, not an arithmetic or
/// matmul placement certificate. Only explicitly tagged 1-D bundles expose
/// its `linear` geometry to dispatch owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrozenShaderTokens {
    linear: FrozenLinearLaunch,
    matmul_tile: u32,
}

impl FrozenShaderTokens {
    pub(crate) fn new(linear: FrozenLinearLaunch, matmul_tile: u32) -> Self {
        Self {
            linear,
            matmul_tile,
        }
    }

    pub(crate) fn linear(self) -> FrozenLinearLaunch {
        self.linear
    }

    pub(crate) fn substitute(self, source: &str) -> String {
        // This constructor is an explicitly unbounded, infallible pipeline
        // API. Managed reduction source never enters through this boundary.
        super::substitute_tokens(source, self.linear.workgroup(), self.matmul_tile, None)
            .expect("unbounded frozen launch source substitution failed")
            .into_parts()
            .0
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn frozen_linear_launch_covers_tails_without_exceeding_enabled_limits() {
        let first = FrozenLinearLaunch::new(64, 3).unwrap();
        let second = FrozenLinearLaunch::new(128, 3).unwrap();
        assert_ne!(first, second);
        assert_eq!(first.chunk_capacity(), Some(192));
        assert_eq!(second.chunk_capacity(), Some(384));
        assert_eq!(first.dispatch_groups(0), Some(0));
        assert_eq!(first.dispatch_groups(64), Some(1));
        assert_eq!(first.dispatch_groups(65), Some(2));
        assert_eq!(first.dispatch_groups(192), Some(3));
        assert_eq!(first.dispatch_groups(193), None);
        assert_eq!(second.dispatch_groups(193), Some(2));
        assert!(FrozenLinearLaunch::new(0, 3).is_none());
        assert!(FrozenLinearLaunch::new(64, 0).is_none());
        let first_tokens = FrozenShaderTokens::new(first, 8);
        let second_tokens = FrozenShaderTokens::new(second, 8);
        assert_eq!(first_tokens.linear(), first);
        assert_eq!(
            first_tokens.substitute("@compute @workgroup_size(@WG@) // @MT@"),
            "@compute @workgroup_size(64) // 8"
        );
        assert_eq!(
            second_tokens.substitute("@compute @workgroup_size(@WG@) // @MT@"),
            "@compute @workgroup_size(128) // 8"
        );
    }
}
