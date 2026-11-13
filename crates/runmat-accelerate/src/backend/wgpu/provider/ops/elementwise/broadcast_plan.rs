//! Shared WGPU broadcast geometry and one-hook requested-buffer bound.
//! This does not establish the complete staged image route or cache state.
use anyhow::{anyhow, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BroadcastRoute {
    pub shape_a: Vec<usize>,
    pub shape_b: Vec<usize>,
    pub out_shape: Vec<usize>,
    pub len: usize,
    pub chunk_capacity: usize,
    pub submissions: u32,
}

impl BroadcastRoute {
    #[cfg(test)]
    pub fn new(a: &[usize], b: &[usize], max_rank: usize, element_bytes: u64) -> Result<Self> {
        Self::with_enabled_limit(
            a,
            b,
            max_rank,
            element_bytes,
            crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS,
        )
    }

    #[cfg(test)]
    pub fn with_enabled_limit(
        a: &[usize],
        b: &[usize],
        max_rank: usize,
        element_bytes: u64,
        enabled_max_dispatch_groups: u32,
    ) -> Result<Self> {
        let launch = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(
            crate::backend::wgpu::config::WORKGROUP_SIZE,
            enabled_max_dispatch_groups,
        )
        .ok_or_else(|| anyhow!("broadcast dispatch capacity overflow"))?;
        Self::with_launch(a, b, max_rank, element_bytes, launch)
    }

    pub fn with_launch(
        a: &[usize],
        b: &[usize],
        max_rank: usize,
        element_bytes: u64,
        launch: crate::backend::wgpu::pipelines::FrozenLinearLaunch,
    ) -> Result<Self> {
        let (shape_a, shape_b, out_shape) = super::super::backend_shared::matlab_broadcast_shapes(
            "binary operation",
            a,
            b,
            max_rank,
        )?;
        let len = out_shape
            .iter()
            .copied()
            .fold(1usize, |product, n| product.saturating_mul(n));
        if len > u32::MAX as usize {
            return Err(
                super::super::backend_shared::gpu_dispatch_length_limit_error(
                    "binary_op_broadcast",
                    len,
                ),
            );
        }
        let chunk_capacity = launch
            .chunk_capacity()
            .ok_or_else(|| anyhow!("broadcast dispatch capacity overflow"))?;
        let submissions = if len == 0 {
            0
        } else {
            u32::try_from(len.div_ceil(chunk_capacity))
                .map_err(|_| anyhow!("broadcast submission count overflow"))?
        };
        Self::requested_buffer_bound(len as u64, element_bytes)
            .ok_or_else(|| anyhow!("broadcast buffer demand overflow"))?;
        Ok(Self {
            shape_a,
            shape_b,
            out_shape,
            len,
            chunk_capacity,
            submissions,
        })
    }

    pub fn requested_buffer_bound(len: u64, element_bytes: u64) -> Option<u64> {
        let output = len.max(1).checked_mul(element_bytes)?;
        if len == 0 {
            return Some(output);
        }
        let uniform = (std::mem::size_of::<crate::backend::wgpu::params::BinaryBroadcastParams>()
            as u64)
            .max(1)
            .div_ceil(256)
            .checked_mul(256)?;
        output.checked_add(uniform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::params::BCAST_MAX_RANK;

    #[test]
    fn image_moment_and_scalar_routes_share_execution_chunk_geometry() {
        let route = BroadcastRoute::new(&[2, 3, 4], &[2, 1, 1], BCAST_MAX_RANK, 4).unwrap();
        assert_eq!(route.out_shape, [2, 3, 4]);
        assert_eq!(route.len, 24);
        assert_eq!(route.submissions, 1);
        assert_eq!(
            BroadcastRoute::requested_buffer_bound(route.len as u64, 4).unwrap(),
            route.len as u64 * 4
                + (std::mem::size_of::<crate::backend::wgpu::params::BinaryBroadcastParams>()
                    as u64)
                    .max(1)
                    .div_ceil(256)
                    * 256
        );
        let power = BroadcastRoute::new(&[2, 3, 4], &[1, 1], BCAST_MAX_RANK, 4).unwrap();
        assert_eq!(power.out_shape, route.out_shape);
        assert_eq!(power.submissions, route.submissions);
    }

    #[test]
    fn browser_device_launch_preserves_broadcast_chunk_tails() {
        let launch = crate::backend::wgpu::pipelines::FrozenLinearLaunch::new(256, 1).unwrap();
        let route =
            BroadcastRoute::with_launch(&[2, 513], &[2, 1], BCAST_MAX_RANK, 4, launch).unwrap();
        assert_eq!(route.chunk_capacity, 256);
        assert_eq!(route.submissions, 5);
        assert_eq!(route.len, 1026);
        assert_eq!(route.out_shape, [2, 513]);
    }

    #[test]
    fn chunk_boundary_and_empty_route_are_exact() {
        let capacity = (crate::backend::wgpu::config::MAX_DISPATCH_WORKGROUPS as usize)
            * crate::backend::wgpu::config::WORKGROUP_SIZE as usize;
        let one = BroadcastRoute::new(&[capacity, 1], &[1, 1], BCAST_MAX_RANK, 4).unwrap();
        let two = BroadcastRoute::new(&[capacity + 1, 1], &[1, 1], BCAST_MAX_RANK, 4).unwrap();
        assert_eq!(one.submissions, 1);
        assert_eq!(two.submissions, 2);
        let empty = BroadcastRoute::new(&[0, 1], &[1, 1], BCAST_MAX_RANK, 4).unwrap();
        assert_eq!(empty.submissions, 0);
        assert_eq!(
            BroadcastRoute::requested_buffer_bound(empty.len as u64, 4),
            Some(4)
        );
    }
}
