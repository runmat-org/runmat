use super::*;

type Rejection = (ProviderRejectionCode, &'static str);
pub(crate) struct CoordinateWork {
    pub(crate) outputs: Vec<ProviderRepresentation>,
    pub(crate) scratch_bytes: u64,
    #[cfg(any(test, feature = "wgpu"))]
    pub(crate) extra_upload_bytes: u64,
    pub(crate) execution_elements: u64,
    #[cfg(feature = "wgpu")]
    pub(crate) allocations: u32,
    pub(crate) dispatches: u32,
    pub(crate) queue_submissions: u32,
}

enum Step {
    Unary(Vec<u64>),
    Multiply {
        left: Vec<u64>,
        right: Vec<u64>,
        shape: Vec<u64>,
    },
}

fn broadcast(left: &[u64], right: &[u64]) -> Result<Vec<u64>, Rejection> {
    let convert = |shape: &[u64]| {
        shape
            .iter()
            .map(|dimension| usize::try_from(*dimension))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                (
                    ProviderRejectionCode::ResourceLimit,
                    "pol2cart.dimension.overflow",
                )
            })
    };
    let shape = runmat_runtime::builtins::common::broadcast::broadcast_shapes(
        "pol2cart",
        &convert(left)?,
        &convert(right)?,
    )
    .map_err(|_| {
        (
            ProviderRejectionCode::InvalidShape,
            "pol2cart.broadcast.invalid",
        )
    })?;
    Ok(shape
        .into_iter()
        .map(|dimension| dimension as u64)
        .collect())
}
fn elements(shape: &[u64]) -> Result<u64, Rejection> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1u64, |n, d| n.checked_mul(*d))
        .ok_or((
            ProviderRejectionCode::ResourceLimit,
            "pol2cart.shape.overflow",
        ))
}
fn add(total: &mut u64, amount: u64) -> Result<(), Rejection> {
    *total = total.checked_add(amount).ok_or((
        ProviderRejectionCode::ResourceLimit,
        "pol2cart.work.overflow",
    ))?;
    Ok(())
}

pub(super) fn derive(
    precision: ProviderPrecision,
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Result<CoordinateWork, Rejection> {
    use ProviderRejectionCode as Code;
    let arity = query.inputs.len();
    let count = query.workload.requested_outputs.unwrap_or(1) as usize;
    if query.family != ProviderOperationFamily::Elementwise
        || !(2..=3).contains(&arity)
        || count > arity
    {
        return Err((Code::InvalidShape, "pol2cart.form.invalid"));
    }
    let element = ProviderElementType::from(precision);
    for input in &query.inputs {
        if input.element_type != element {
            return Err((
                Code::UnsupportedElementType,
                "pol2cart.exact_float_class.required",
            ));
        }
        if input.storage != ProviderStorage::DenseReal {
            return Err((Code::UnsupportedStorage, "pol2cart.real_dense.required"));
        }
        if input.layout != ProviderLayout::ColumnMajorContiguous {
            return Err((Code::UnsupportedLayout, "pol2cart.contiguous.required"));
        }
    }
    let theta = &query.inputs[0].shape;
    let rho = &query.inputs[1].shape;
    let pair = broadcast(theta, rho)?;
    let mut shape = vec![1, 1];
    for input in &query.inputs {
        shape = broadcast(&shape, &input.shape)?;
    }
    let output = ProviderRepresentation {
        element_type: element,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape.clone(),
        residency: ProviderResidency::Device,
    };
    let outputs = vec![output; count];
    if !query.outputs.is_empty() && query.outputs != outputs {
        return Err((Code::InvalidShape, "pol2cart.outputs.mismatch"));
    }
    let multiply = |left: &[u64], right: &[u64], output: &[u64]| Step::Multiply {
        left: left.to_vec(),
        right: right.to_vec(),
        shape: output.to_vec(),
    };
    // The current body computes both coordinates even for zero/one output.
    // Count that work rather than attributing an unimplemented optimization.
    let mut steps = vec![
        Step::Unary(theta.clone()),
        Step::Unary(theta.clone()),
        multiply(rho, theta, &pair),
        multiply(rho, theta, &pair),
    ];
    let expands_xy = pair != shape;
    let expands_z = count == 3 && query.inputs[2].shape != shape;
    let ones = if expands_xy || expands_z {
        Some(shape.clone())
    } else if count == 3 {
        Some(vec![1, 1])
    } else {
        None
    };
    if expands_xy {
        steps.push(multiply(&pair, &shape, &shape));
        steps.push(multiply(&pair, &shape, &shape));
    }
    if count == 3 {
        steps.push(multiply(
            &query.inputs[2].shape,
            ones.as_ref().expect("elevation multiplier"),
            &shape,
        ));
    }
    let width = element.byte_width();
    let extra_upload_bytes = ones
        .as_ref()
        .map(|shape| elements(shape))
        .transpose()?
        .unwrap_or(0)
        .checked_mul(width)
        .ok_or((Code::ResourceLimit, "pol2cart.ones.overflow"))?;
    let mut result = CoordinateWork {
        outputs,
        scratch_bytes: if wgpu && ones.is_some() {
            extra_upload_bytes.max(width)
        } else {
            extra_upload_bytes
        },
        #[cfg(any(test, feature = "wgpu"))]
        extra_upload_bytes,
        execution_elements: 0,
        #[cfg(feature = "wgpu")]
        allocations: steps.len() as u32 + u32::from(ones.is_some()),
        dispatches: 0,
        queue_submissions: u32::from(ones.is_some()),
    };
    for step in &steps {
        if let Step::Multiply { left, right, shape } = step {
            if broadcast(left, right)? != *shape {
                return Err((Code::InvalidShape, "pol2cart.compound.shape"));
            }
        }
        let output_shape = match step {
            Step::Unary(shape) | Step::Multiply { shape, .. } => shape,
        };
        let n = elements(output_shape)?;
        let bytes = (if wgpu { n.max(1) } else { n })
            .checked_mul(width)
            .ok_or((Code::ResourceLimit, "pol2cart.output.overflow"))?;
        add(&mut result.scratch_bytes, bytes)?;
        add(&mut result.execution_elements, n)?;
        if wgpu {
            #[cfg(feature = "wgpu")]
            account_wgpu(&mut result, step, n)?;
            #[cfg(not(feature = "wgpu"))]
            return Err((Code::ProviderUnavailable, "pol2cart.wgpu.unavailable"));
        } else {
            result.dispatches += 1;
            result.queue_submissions += 1;
        }
    }
    // Published outputs are already counted by dense feasibility.
    let published_elements = elements(&shape)?;
    let published = (if wgpu {
        published_elements.max(1)
    } else {
        published_elements
    })
    .checked_mul(width)
    .and_then(|n| n.checked_mul(count as u64))
    .ok_or((Code::ResourceLimit, "pol2cart.published.overflow"))?;
    result.scratch_bytes = result
        .scratch_bytes
        .checked_sub(published)
        .ok_or((Code::ResourceLimit, "pol2cart.accounting.invalid"))?;
    Ok(result)
}

#[cfg(feature = "wgpu")]
fn account_wgpu(work: &mut CoordinateWork, step: &Step, n: u64) -> Result<(), Rejection> {
    use crate::backend::wgpu::{config, params};
    use ProviderRejectionCode as Code;
    if ["RUNMAT_DISABLE_UNARY", "RUNMAT_DISABLE_BINARY"]
        .iter()
        .any(|key| std::env::var(key).is_ok())
    {
        return Err((Code::UnsupportedOperation, "pol2cart.disabled_hooks"));
    }
    if n > u64::from(u32::MAX) {
        return Err((Code::ResourceLimit, "pol2cart.dispatch.length"));
    }
    let chunk = u64::from(config::WORKGROUP_SIZE) * u64::from(config::MAX_DISPATCH_WORKGROUPS);
    let chunks = n.div_ceil(chunk) as u32;
    let (extra_queues, uniform) = match step {
        Step::Unary(_) => {
            let effective = u64::from(config::effective_workgroup_size());
            if effective == 0
                || n.min(chunk).div_ceil(effective) > u64::from(config::MAX_DISPATCH_WORKGROUPS)
            {
                return Err((Code::ResourceLimit, "pol2cart.unary.groups"));
            }
            (1, std::mem::size_of::<params::FusionParams>())
        }
        Step::Multiply { left, right, shape } => {
            if left != right && (shape.len() > params::BCAST_MAX_RANK || shape.len() < 2) {
                return Err((Code::UnsupportedRank, "pol2cart.broadcast.rank"));
            }
            if n == 0 {
                (0, 0)
            } else if left == right {
                (2, std::mem::size_of::<params::LenOpParams>())
            } else {
                if shape.len() > params::BCAST_MAX_RANK {
                    return Err((Code::UnsupportedRank, "pol2cart.broadcast.rank"));
                }
                if shape.len() < 2 {
                    return Err((Code::UnsupportedRank, "pol2cart.broadcast.row_shorthand"));
                }
                (0, std::mem::size_of::<params::BinaryBroadcastParams>())
            }
        }
    };
    work.dispatches += chunks;
    work.queue_submissions += chunks + extra_queues;
    work.allocations += u32::from(uniform > 0);
    add(&mut work.scratch_bytes, uniform as u64)
}
