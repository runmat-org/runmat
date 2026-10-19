//! Physical resources of the admitted ordered matrix route. Borrowed resident
//! buffers are not new allocations; diagonal extraction retains the full product.
use super::*;
use runmat_accelerate_api::{MatmulOffset, MatmulWorkload, ProviderMatmulGeometry};
#[cfg(test)]
mod tests;

pub(crate) struct Work {
    pub estimate: ProviderResourceEstimate,
    largest_allocation: u64,
    #[cfg(any(test, feature = "wgpu"))]
    pub scratch_bytes: u64,
    #[cfg(any(test, feature = "wgpu"))]
    pub queue_submissions: u32,
    #[cfg(any(test, feature = "wgpu"))]
    pub allocations: u32,
    #[cfg(any(test, feature = "wgpu"))]
    pub uniform_bytes: u64,
    #[cfg(any(test, feature = "wgpu"))]
    pub scalar_units: u64,
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.workload.matmul.is_none() {
        return (query.operation.0
            == crate::placement::fusion_operation_token(
                &crate::fusion::FusionKind::MatmulEpilogue,
            ))
        .then(|| {
            ProviderFeasibility::rejected(
                ProviderRejectionCode::UnknownResourceEstimate,
                "fusion.matmul.missing_admitted_route",
            )
        });
    }
    if wgpu && provider.matmul_geometry().is_none() {
        return Some(ProviderFeasibility::rejected(
            ProviderRejectionCode::UnknownResourceEstimate,
            "fusion.matmul.missing_kernel_geometry",
        ));
    }
    Some(match work(provider, wgpu, query) {
        Some(work)
            if provider
                .capability_snapshot()
                .max_allocation_bytes
                .is_some_and(|limit| work.largest_allocation > limit) =>
        {
            ProviderFeasibility::rejected(
                ProviderRejectionCode::ResourceLimit,
                "fusion.matmul.allocation_limit",
            )
        }
        Some(work) => ProviderFeasibility::supported(work.estimate),
        None => ProviderFeasibility::rejected(
            ProviderRejectionCode::InvalidShape,
            "fusion.matmul.invalid_admitted_route",
        ),
    })
}

#[cfg(feature = "wgpu")]
pub(crate) fn physical_work(
    provider: &dyn AccelProvider,
    query: &ProviderFeasibilityQuery,
) -> Option<Work> {
    work(provider, true, query)
}

fn finite_control(bits: u64, precision: ProviderPrecision) -> Option<f64> {
    let value = f64::from_bits(bits);
    (value.is_finite() && (precision == ProviderPrecision::F64 || f64::from(value as f32) == value))
        .then_some(value)
}

fn controls(work: &MatmulWorkload, precision: ProviderPrecision) -> Option<u64> {
    let mut count = 0;
    if let Some(bits) = work.affine.scale {
        finite_control(bits, precision)?;
        count += 1;
    }
    if let Some(offset) = work.affine.offset {
        let (MatmulOffset::Add(bits)
        | MatmulOffset::Subtract(bits)
        | MatmulOffset::SubtractFrom(bits)) = offset;
        finite_control(bits, precision)?;
        count += 1;
    }
    for bits in [work.clamp_min, work.clamp_max].into_iter().flatten() {
        finite_control(bits, precision)?;
        count += 1;
    }
    if let Some(bits) = work.exponent {
        let exponent = finite_control(bits, precision)?;
        let nonnegative = work
            .clamp_min
            .is_some_and(|bits| f64::from_bits(bits) >= 0.0)
            && work
                .clamp_max
                .is_none_or(|bits| f64::from_bits(bits) >= 0.0);
        if exponent.fract() != 0.0 && !nonnegative {
            return None;
        }
        // Explicit transcendental work prior, not a measured instruction bound.
        count += 32;
    }
    Some(count + u64::from(work.row.is_some()) + u64::from(work.column.is_some()))
}

/// Shared CPU/provider prior counts the same admitted operations. This is not
/// a measurement or an exact transcendental instruction count.
pub(crate) fn scalar_work_units(
    route: &MatmulWorkload,
    precision: ProviderPrecision,
) -> Option<u64> {
    route
        .rows
        .checked_mul(route.columns)?
        .checked_mul(
            route
                .inner
                .checked_mul(2)?
                .checked_add(controls(route, precision)?)?
                .checked_add(1)?, // output write, including k=0
        )?
        .checked_add(if route.diagonal {
            route.rows.min(route.columns)
        } else {
            0
        })
}

fn work(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<Work> {
    work_with_geometry(provider, wgpu, query, provider.matmul_geometry())
}

fn work_with_geometry(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
    geometry: Option<ProviderMatmulGeometry>,
) -> Option<Work> {
    if wgpu && geometry.is_none() {
        return None;
    }
    let route = query.workload.matmul.as_ref()?;
    if query.family != ProviderOperationFamily::Fusion
        || query.operation.0
            != crate::placement::fusion_operation_token(&crate::fusion::FusionKind::MatmulEpilogue)
    {
        return None;
    }
    let precision = provider.precision();
    if wgpu
        && precision == ProviderPrecision::F64
        && !provider.supports_shader_requirements(
            runmat_accelerate_api::ProviderShaderRequirements::F64IeeeWords,
        )
    {
        return None;
    }
    let width = match precision {
        ProviderPrecision::F32 => 4u64,
        ProviderPrecision::F64 => 8u64,
    };
    controls(route, precision)?;
    let a = query.inputs.get(route.left)?;
    let b = query.inputs.get(route.right)?;
    if a.shape != [route.rows, route.inner] || b.shape != [route.inner, route.columns] {
        return None;
    }
    let elements = route.rows.checked_mul(route.columns)?;
    let matrix_bytes = elements.checked_mul(width)?;
    let diagonal_elements = route.rows.min(route.columns);
    if route.diagonal && (route.rows == 1 || route.columns == 1) {
        return None;
    }
    let output_shape = if route.diagonal {
        vec![diagonal_elements, 1]
    } else {
        vec![route.rows, route.columns]
    };
    let [output] = query.outputs.as_slice() else {
        return None;
    };
    if output.shape != output_shape || output.residency != ProviderResidency::Device {
        return None;
    }
    let mut roles = vec![route.left, route.right];
    for (role, shape) in [
        (route.row, vec![route.rows, 1]),
        (route.column, vec![1, route.columns]),
    ] {
        if let Some(role) = role {
            if query.inputs.get(role.input)?.shape != shape {
                return None;
            }
            roles.push(role.input);
        }
    }
    if (0..query.inputs.len()).any(|index| !roles.contains(&index)) {
        return None;
    }
    for value in query.inputs.iter().chain(&query.outputs) {
        if value.element_type != ProviderElementType::from(precision)
            || value.storage != ProviderStorage::DenseReal
            || value.layout != ProviderLayout::ColumnMajorContiguous
            || !matches!(
                value.residency,
                ProviderResidency::Host | ProviderResidency::Device
            )
        {
            return None;
        }
        value.checked_byte_len()?;
    }
    let output_bytes = output.checked_byte_len()?.max(if wgpu { width } else { 0 });
    let mut uploaded = 0u64;
    for value in &query.inputs {
        if value.residency == ProviderResidency::Host {
            uploaded = uploaded.checked_add(value.checked_byte_len()?.max(if wgpu {
                width
            } else {
                0
            }))?;
        }
    }
    let nonempty = elements != 0;
    let diagonal_dispatch = route.diagonal && diagonal_elements != 0;
    let dispatches = u32::from(nonempty) + u32::from(diagonal_dispatch);
    let (matrix_scratch, uniform_bytes, largest_uniform) = if wgpu {
        #[cfg(feature = "wgpu")]
        {
            use crate::backend::wgpu::{config, params};
            let mut matrix_scratch = if route.diagonal {
                matrix_bytes.max(width)
            } else {
                0
            };
            let mut uniform_bytes = 0u64;
            let groups = geometry?.dispatch_groups(route.rows, route.columns, route.diagonal)?;
            if route.rows > u64::from(u32::MAX)
                || route.inner > u64::from(u32::MAX)
                || route.columns > u64::from(u32::MAX)
                || groups
                    .into_iter()
                    .any(|count| count > config::MAX_DISPATCH_WORKGROUPS)
            {
                return None;
            }
            if nonempty {
                uniform_bytes = u64::try_from(std::mem::size_of::<params::MatmulParams>())
                    .ok()?
                    .checked_add(match precision {
                        ProviderPrecision::F32 => {
                            std::mem::size_of::<params::MatmulEpilogueParamsF32>() as u64
                        }
                        ProviderPrecision::F64 => {
                            std::mem::size_of::<params::MatmulEpilogueParamsF64>() as u64
                        }
                    })?;
                matrix_scratch = matrix_scratch.checked_add(width)?; // shared dummy row/column buffer
            }
            if diagonal_dispatch {
                uniform_bytes = uniform_bytes
                    .checked_add(std::mem::size_of::<params::DiagExtractParams>() as u64)?;
            }
            let largest_uniform = if nonempty {
                (std::mem::size_of::<params::MatmulParams>() as u64).max(match precision {
                    ProviderPrecision::F32 => {
                        std::mem::size_of::<params::MatmulEpilogueParamsF32>() as u64
                    }
                    ProviderPrecision::F64 => {
                        std::mem::size_of::<params::MatmulEpilogueParamsF64>() as u64
                    }
                })
            } else {
                0
            };
            (matrix_scratch, uniform_bytes, largest_uniform)
        }
        #[cfg(not(feature = "wgpu"))]
        return None;
    } else {
        // The reference provider clones product data during epilogue, and
        // clones the full product during diagonal extraction. Those phases
        // do not overlap; include the larger phase's additional live bytes.
        let scales = route
            .row
            .map_or(Some(0), |_| route.rows.checked_mul(width))?
            .checked_add(
                route
                    .column
                    .map_or(Some(0), |_| route.columns.checked_mul(width))?,
            )?;
        let matrix_scratch = if route.diagonal {
            matrix_bytes
                .checked_mul(2)?
                .checked_add(scales.saturating_sub(output_bytes))?
        } else {
            matrix_bytes.checked_add(scales)?
        };
        (matrix_scratch, 0, 0)
    };
    let scratch_bytes = uploaded
        .checked_add(matrix_scratch)?
        .checked_add(uniform_bytes)?;
    let largest = query
        .inputs
        .iter()
        .chain(&query.outputs)
        .filter_map(ProviderRepresentation::checked_byte_len)
        .chain([matrix_bytes, largest_uniform])
        .max()?;
    Some(Work {
        largest_allocation: largest.max(if wgpu { width } else { 0 }),
        estimate: ProviderResourceEstimate {
            transient_bytes: Some(scratch_bytes),
            output_bytes: Some(output_bytes),
            dispatches: Some(dispatches),
        },
        #[cfg(any(test, feature = "wgpu"))]
        scratch_bytes,
        #[cfg(any(test, feature = "wgpu"))]
        queue_submissions: dispatches + u32::from(wgpu && diagonal_dispatch), // existing diagonal no-op warmup submission
        #[cfg(any(test, feature = "wgpu"))]
        allocations: 1
            + u32::from(route.diagonal)
            + u32::try_from(
                query
                    .inputs
                    .iter()
                    .filter(|input| input.residency == ProviderResidency::Host)
                    .count(),
            )
            .ok()?
            + if wgpu && nonempty {
                3 + u32::from(diagonal_dispatch)
            } else {
                0
            },
        #[cfg(any(test, feature = "wgpu"))]
        uniform_bytes,
        #[cfg(any(test, feature = "wgpu"))]
        scalar_units: scalar_work_units(route, precision)?,
    })
}
