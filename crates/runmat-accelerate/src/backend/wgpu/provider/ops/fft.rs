use anyhow::{anyhow, ensure, Result};
use bytemuck::{bytes_of, cast_slice, Pod};
use num_complex::Complex;
use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage, HostTensorOwned};
use runmat_runtime::builtins::common::shape::normalize_scalar_shape;
use rustfft::FftPlanner;
use std::sync::Arc;

use crate::backend::wgpu::resources::UniformBufferKey;
use crate::backend::wgpu::types::NumericPrecision;

use super::WgpuProvider;

fn fft_trim_trailing_ones(shape: &mut Vec<usize>, minimum_rank: usize) {
    while shape.len() > minimum_rank && shape.last() == Some(&1) {
        shape.pop();
    }
    *shape = normalize_scalar_shape(shape);
}

fn fft_is_power_of_two(len: usize) -> bool {
    len != 0 && (len & (len - 1)) == 0
}

fn fft_log2_pow2(len: usize) -> Option<u32> {
    if !fft_is_power_of_two(len) {
        return None;
    }
    Some(len.trailing_zeros())
}

fn fft_log3_pow3(mut len: usize) -> Option<u32> {
    if len == 0 {
        return None;
    }
    let mut d = 0u32;
    while len > 1 {
        if !len.is_multiple_of(3) {
            return None;
        }
        len /= 3;
        d += 1;
    }
    Some(d)
}

fn fft_log5_pow5(mut len: usize) -> Option<u32> {
    if len == 0 {
        return None;
    }
    let mut d = 0u32;
    while len > 1 {
        if !len.is_multiple_of(5) {
            return None;
        }
        len /= 5;
        d += 1;
    }
    Some(d)
}

fn fft_factor_smooth_235(mut len: usize) -> Option<Vec<u32>> {
    if len <= 1 {
        return None;
    }
    let mut factors = Vec::new();
    while len.is_multiple_of(5) {
        factors.push(5);
        len /= 5;
    }
    while len.is_multiple_of(3) {
        factors.push(3);
        len /= 3;
    }
    while len.is_multiple_of(2) {
        factors.push(2);
        len /= 2;
    }
    if len == 1 && !factors.is_empty() {
        Some(factors)
    } else {
        None
    }
}

#[path = "fft/fallback.rs"]
mod fallback;
#[path = "fft/helpers.rs"]
mod helpers;
#[path = "fft/kernels.rs"]
mod kernels;
