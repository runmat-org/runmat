//! Resources and an explicit static work bound for the digit kernel.
#[cfg(feature = "wgpu")]
use super::*;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{NumericalDomainEvidence, RoundDigitsMode};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct RoundDigitsWork {
    pub dispatches: u32,
    pub queue_submissions: u32,
    pub allocations: u32,
    pub scratch_bytes: u64,
    pub extra_upload_bytes: u64,
    pub execution_units: u64,
}

#[cfg(feature = "wgpu")]
pub(crate) fn wgpu_work(
    precision: ProviderPrecision,
    query: &ProviderFeasibilityQuery,
    linear_launch: crate::backend::wgpu::pipelines::FrozenLinearLaunch,
) -> Option<RoundDigitsWork> {
    if query.operation.0 != ROUND_CATALOG_ENTRY.identity.name
        || query.family != ProviderOperationFamily::Elementwise
    {
        return None;
    }
    let NumericalDomainEvidence::RoundDigits { digits, mode } = query.workload.numerical_domain
    else {
        return None;
    };
    if mode == RoundDigitsMode::Significant && digits <= 0 {
        return None;
    }
    let [input] = query.inputs.as_slice() else {
        return None;
    };
    if input.element_type != ProviderElementType::from(precision)
        || !matches!(
            input.storage,
            ProviderStorage::DenseReal | ProviderStorage::DenseComplexInterleaved
        )
        || input.layout != ProviderLayout::ColumnMajorContiguous
    {
        return None;
    }
    let elements = input.checked_element_count()?;
    let scalar_lanes = elements.checked_mul(
        if input.storage == ProviderStorage::DenseComplexInterleaved {
            2
        } else {
            1
        },
    )?;
    if scalar_lanes > u64::from(u32::MAX) {
        return None;
    }
    let capacity = u64::try_from(linear_launch.chunk_capacity()?).ok()?;
    let dispatches = u32::try_from(scalar_lanes.div_ceil(capacity)).ok()?;
    // Significant controls resolve their scale from each represented value;
    // bound both integer-power loops by the format's possible scale range.
    let (maximum, tail) = match precision {
        ProviderPrecision::F32 => (38_u32, 7_u32),
        ProviderPrecision::F64 => (308_u32, 16_u32),
    };
    let (first, second) = if mode == RoundDigitsMode::Decimals {
        let magnitude = digits.unsigned_abs();
        (
            magnitude.min(maximum),
            magnitude.saturating_sub(maximum).min(tail),
        )
    } else {
        (maximum, tail)
    };
    let power_steps = |exponent: u32| {
        if exponent == 0 {
            0
        } else {
            exponent.count_ones() + (u32::BITS - exponent.leading_zeros() - 1)
        }
    };
    // This is a static prior, not GPU completion telemetry. The fixed work
    // includes bit classification, up to two logarithms, fraction rounding,
    // normalization/packing, and four residual-corrected divisions.
    let steps = if mode == RoundDigitsMode::Significant {
        (0..=maximum).map(power_steps).max().unwrap_or(0)
            + (0..=tail).map(power_steps).max().unwrap_or(0)
    } else {
        power_steps(first) + power_steps(second)
    };
    let units_per_element = 96_u64.checked_add(u64::from(steps))?;
    Some(RoundDigitsWork {
        dispatches,
        queue_submissions: dispatches,
        allocations: if scalar_lanes == 0 { 1 } else { 2 },
        scratch_bytes: if scalar_lanes == 0 {
            0
        } else {
            std::mem::size_of::<crate::backend::wgpu::params::RoundDigitsParams>() as u64
        },
        execution_units: scalar_lanes.checked_mul(units_per_element)?,
        extra_upload_bytes: u64::from(dispatches).checked_mul(std::mem::size_of::<
            crate::backend::wgpu::params::RoundDigitsParams,
        >() as u64)?,
    })
}
