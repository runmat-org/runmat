//! Typed real/complex component hooks, including constructor output allocation.
use super::*;
use runmat_builtins::NumericComponentRule;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum Operation {
    Component(NumericComponentRule),
    Construct,
}
const OPERATIONS: &[Operation] = &[
    Operation::Component(NumericComponentRule::Conjugate),
    Operation::Component(NumericComponentRule::RealPart),
    Operation::Component(NumericComponentRule::ImaginaryPart),
    Operation::Construct,
];
impl Operation {
    fn identity(self) -> &'static str {
        use runmat_builtins::*;
        match self {
            Self::Component(NumericComponentRule::Conjugate) => CONJ_CATALOG_ENTRY.identity.name,
            Self::Component(NumericComponentRule::RealPart) => REAL_CATALOG_ENTRY.identity.name,
            Self::Component(NumericComponentRule::ImaginaryPart) => {
                IMAG_CATALOG_ENTRY.identity.name
            }
            Self::Construct => COMPLEX_CATALOG_ENTRY.identity.name,
        }
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
            .map(|operation| ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(operation.identity()),
                family: ProviderOperationFamily::Elementwise,
            }),
    );
}

pub(super) fn feasibility(
    profile: HookProfile,
    precision: ProviderPrecision,
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    let operation = OPERATIONS
        .iter()
        .find(|operation| operation.identity() == query.operation.0)?;
    Some(evaluate(*operation, profile, precision, snapshot, query))
}

fn evaluate(
    operation: Operation,
    profile: HookProfile,
    precision: ProviderPrecision,
    snapshot: &ProviderCapabilitySnapshot,
    query: &ProviderFeasibilityQuery,
) -> ProviderFeasibility {
    let reject = ProviderFeasibility::rejected;
    if query.family != ProviderOperationFamily::Elementwise {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "component.family.mismatch",
        );
    }
    let valid_count = match operation {
        Operation::Construct => matches!(query.inputs.len(), 1 | 2),
        Operation::Component(_) => query.inputs.len() == 1,
    };
    if !valid_count || query.outputs.len() > 1 {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "component.operand_count",
        );
    }
    let real_type = ProviderElementType::from(precision);
    let complex_type = match precision {
        ProviderPrecision::F32 => ProviderElementType::ComplexF32,
        ProviderPrecision::F64 => ProviderElementType::ComplexF64,
    };
    for input in &query.inputs {
        let valid = match input.storage {
            ProviderStorage::DenseReal => input.element_type == real_type,
            ProviderStorage::DenseComplexInterleaved
                if matches!(operation, Operation::Component(_)) =>
            {
                input.element_type == complex_type
            }
            _ => false,
        };
        if !valid {
            return reject(
                ProviderRejectionCode::UnsupportedElementType,
                "component.representation.unsupported",
            );
        }
    }
    let mut output = query.inputs[0].clone();
    output.residency = ProviderResidency::Device;
    let output_complex = matches!(operation, Operation::Construct)
        || (matches!(
            operation,
            Operation::Component(NumericComponentRule::Conjugate)
        ) && output.storage == ProviderStorage::DenseComplexInterleaved);
    output.storage = if output_complex {
        ProviderStorage::DenseComplexInterleaved
    } else {
        ProviderStorage::DenseReal
    };
    output.element_type = if output_complex {
        complex_type
    } else {
        real_type
    };
    if query.inputs.len() == 2 {
        let left = &query.inputs[0].shape;
        let right = &query.inputs[1].shape;
        if left == right || count(right) == Some(1) {
            output.shape = left.clone();
        } else if count(left) == Some(1) {
            output.shape = right.clone();
        } else {
            return reject(
                ProviderRejectionCode::InvalidShape,
                "component.constructor.shape",
            );
        }
    }
    let Some(elements) = count(&output.shape) else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "component.output_shape.overflow",
        );
    };
    let Some(words) = elements.checked_mul(if output_complex { 2 } else { 1 }) else {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "component.output_storage.overflow",
        );
    };
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu) && words > u64::from(u32::MAX) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "component.dispatch.length",
        );
    }
    let _ = (profile, words);
    if query.outputs.iter().any(|actual| {
        actual.shape != output.shape
            || actual.element_type != output.element_type
            || actual.storage != output.storage
    }) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "component.output_representation.mismatch",
        );
    }
    let mut accounted = query.clone();
    if accounted.outputs.is_empty() {
        accounted.outputs.push(output);
    }
    #[cfg(feature = "wgpu")]
    if matches!(profile, HookProfile::Wgpu)
        && matches!(
            operation,
            Operation::Component(NumericComponentRule::RealPart)
        )
        && query.inputs[0].storage == ProviderStorage::DenseReal
    {
        // This hook registers another handle over the existing buffer; it
        // does not allocate an output payload or dispatch a kernel.
        accounted.outputs.clear();
    }
    let result = dense_feasibility(snapshot, &accounted);
    #[cfg(feature = "wgpu")]
    let result = {
        let mut result = result;
        if let ProviderFeasibility::Supported { estimate } = &mut result {
            if matches!(profile, HookProfile::Wgpu)
                && (elements == 0
                    || (matches!(
                        operation,
                        Operation::Component(NumericComponentRule::RealPart)
                    ) && query.inputs[0].storage == ProviderStorage::DenseReal))
            {
                estimate.dispatches = Some(0);
            }
        }
        result
    };
    result
}

fn count(shape: &[u64]) -> Option<u64> {
    shape
        .iter()
        .try_fold(1u64, |n, extent| n.checked_mul(*extent))
}
