//! Exact dense floating arithmetic hooks and their broadcast temporaries.
use super::*;
use runmat_builtins::BinaryArithmeticInferenceRule as Operation;

#[cfg(test)]
mod tests;

const OPERATIONS: &[Operation] = &[
    Operation::Add,
    Operation::Subtract,
    Operation::Multiply,
    Operation::RightDivide,
    Operation::LeftDivide,
];

fn identity(operation: Operation) -> &'static str {
    use runmat_builtins::*;
    match operation {
        Operation::Add => PLUS_CATALOG_ENTRY.identity.name,
        Operation::Subtract => MINUS_CATALOG_ENTRY.identity.name,
        Operation::Multiply => TIMES_CATALOG_ENTRY.identity.name,
        Operation::RightDivide => RDIVIDE_CATALOG_ENTRY.identity.name,
        Operation::LeftDivide => LDIVIDE_CATALOG_ENTRY.identity.name,
        Operation::Power => POWER_CATALOG_ENTRY.identity.name,
    }
}

#[derive(Clone, Copy)]
pub(super) enum HookProfile {
    InProcess,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.extend(
        OPERATIONS
            .iter()
            .map(|&operation| ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(identity(operation)),
                family: ProviderOperationFamily::Elementwise,
            }),
    );
    // Power additionally needs a value-domain proof: real bases and exponents
    // can require complex output. A real elem_pow hook alone is insufficient.
}

pub(super) fn feasibility(
    profile: HookProfile,
    precision: ProviderPrecision,
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    OPERATIONS
        .iter()
        .find(|&&operation| identity(operation) == query.operation.0)?;
    Some(evaluate(profile, precision, snapshot, query))
}

fn evaluate(
    profile: HookProfile,
    precision: ProviderPrecision,
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> ProviderFeasibility {
    let reject = ProviderFeasibility::rejected;
    if query.family != ProviderOperationFamily::Elementwise {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "arithmetic.family.mismatch",
        );
    }
    if query.inputs.len() != 2 || query.outputs.len() > 1 {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "arithmetic.operand_count",
        );
    }
    let element_type = ProviderElementType::from(precision);
    if query
        .inputs
        .iter()
        .chain(&query.outputs)
        .any(|v| v.element_type != element_type)
    {
        return reject(
            ProviderRejectionCode::UnsupportedElementType,
            "arithmetic.floating_precision.mismatch",
        );
    }
    if query
        .inputs
        .iter()
        .chain(&query.outputs)
        .any(|v| v.storage != ProviderStorage::DenseReal)
    {
        return reject(
            ProviderRejectionCode::UnsupportedStorage,
            "arithmetic.real_storage.required",
        );
    }
    let left = &query.inputs[0].shape;
    let right = &query.inputs[1].shape;
    let rank = left.len().max(right.len());
    if left != right && (left.len() < 2 || right.len() < 2) {
        // Runtime repmat currently retains the physical leading extent of a
        // rank-one handle; it cannot implement row-shorthand alignment safely.
        return reject(
            ProviderRejectionCode::UnsupportedRank,
            "arithmetic.broadcast.row_shorthand",
        );
    }
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu)
        && left != right
        && rank > crate::backend::wgpu::params::REPMAT_MAX_RANK
    {
        return reject(
            ProviderRejectionCode::UnsupportedRank,
            "arithmetic.broadcast.rank",
        );
    }
    let _ = profile;
    let mut shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        let a = left.get(axis).copied().unwrap_or(1);
        let b = right.get(axis).copied().unwrap_or(1);
        shape.push(if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "arithmetic.broadcast.invalid",
            );
        });
    }
    let Some(elements) = shape.iter().try_fold(1_u64, |n, d| n.checked_mul(*d)) else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "arithmetic.output_shape.overflow",
        );
    };
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu) && elements > u64::from(u32::MAX) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "arithmetic.dispatch.length",
        );
    }
    let _ = elements;
    if query.outputs.iter().any(|v| v.shape != shape) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "arithmetic.output_shape.mismatch",
        );
    }
    let mut accounted = query.clone();
    if accounted.outputs.is_empty() {
        let mut output = query.inputs[0].clone();
        output.shape = shape.clone();
        output.residency = ProviderResidency::Device;
        accounted.outputs.push(output);
    }
    let mut expansions = 0;
    for input in &query.inputs {
        if input
            .shape
            .iter()
            .enumerate()
            .any(|(axis, extent)| *extent != shape[axis])
            || shape
                .iter()
                .skip(input.shape.len())
                .any(|extent| *extent != 1)
        {
            let mut expanded = input.clone();
            expanded.shape = shape.clone();
            expanded.residency = ProviderResidency::Device;
            accounted.inputs.push(expanded);
            expansions += 1;
        }
    }
    match dense_feasibility(snapshot, &accounted) {
        ProviderFeasibility::Supported { mut estimate } => {
            estimate.dispatches = Some(1 + expansions);
            ProviderFeasibility::supported(estimate)
        }
        rejected => rejected,
    }
}
