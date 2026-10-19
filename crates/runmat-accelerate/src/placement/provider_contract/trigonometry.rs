//! Capabilities of the actual direct trigonometric runtime/provider routes.
//! Range-dependent inverse functions and host-computed degree/pi-scaled forms
//! need their own compound workload representation; these are not aliases for
//! an arbitrary real unary kernel.
use super::*;
use runmat_builtins::*;

mod atan2;
pub(crate) mod inverse;
pub(crate) mod pol2cart;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum DirectUnary {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Asinh,
    Atan,
    Tanh,
}

impl DirectUnary {
    fn entry(self) -> &'static BuiltinCatalogEntry {
        match self {
            Self::Sin => &SIN_CATALOG_ENTRY,
            Self::Cos => &COS_CATALOG_ENTRY,
            Self::Tan => &TAN_CATALOG_ENTRY,
            Self::Sinh => &SINH_CATALOG_ENTRY,
            Self::Cosh => &COSH_CATALOG_ENTRY,
            Self::Asinh => &ASINH_CATALOG_ENTRY,
            Self::Atan => &ATAN_CATALOG_ENTRY,
            Self::Tanh => &TANH_CATALOG_ENTRY,
        }
    }

    fn complex(self) -> bool {
        // sinh/cosh have complex provider hooks, but their current runtime
        // route intentionally gathers complex input before restoring its owner.
        matches!(self, Self::Sin | Self::Cos | Self::Tan)
    }
}

const OPERATIONS: &[DirectUnary] = &[
    DirectUnary::Sin,
    DirectUnary::Cos,
    DirectUnary::Tan,
    DirectUnary::Sinh,
    DirectUnary::Cosh,
    DirectUnary::Asinh,
    DirectUnary::Atan,
    DirectUnary::Tanh,
];

pub(super) fn append(snapshot: &mut ProviderCapabilitySnapshot) {
    snapshot.operations.push(pol2cart::operation());
    inverse::append(snapshot);
    snapshot.operations.push(atan2::operation());
    snapshot.operations.extend(
        OPERATIONS
            .iter()
            .map(|operation| ProviderCapabilityOperation {
                identity: ProviderOperationIdentity::new(operation.entry().identity.name),
                family: ProviderOperationFamily::Elementwise,
            }),
    );
}

pub(super) fn validate(
    provider: &(impl AccelProvider + ?Sized),
    wgpu: bool,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if let Some(result) = pol2cart::validate(provider, wgpu, query) {
        return Some(result);
    }
    if let Some(result) = inverse::validate(provider, wgpu, query) {
        return Some(result);
    }
    if let Some(result) = atan2::validate(provider, wgpu, query) {
        return Some(result);
    }
    let operation = OPERATIONS
        .iter()
        .find(|operation| operation.entry().identity.name == query.operation.0)?;
    let reject = |code, detail| Some(ProviderFeasibility::rejected(code, detail));
    if query.family != ProviderOperationFamily::Elementwise {
        return reject(
            ProviderRejectionCode::UnsupportedOperation,
            "trigonometry.family.mismatch",
        );
    }
    if query.inputs.len() != 1 || query.outputs.len() > 1 {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "trigonometry.operand_count",
        );
    }
    let input = &query.inputs[0];
    let real = ProviderElementType::from(provider.precision());
    let complex = match provider.precision() {
        ProviderPrecision::F32 => ProviderElementType::ComplexF32,
        ProviderPrecision::F64 => ProviderElementType::ComplexF64,
    };
    if !matches!(
        input.storage,
        ProviderStorage::DenseReal | ProviderStorage::DenseComplexInterleaved
    ) {
        return reject(
            ProviderRejectionCode::UnsupportedStorage,
            "trigonometry.dense.required",
        );
    }
    let supported = match input.storage {
        ProviderStorage::DenseReal => input.element_type == real,
        ProviderStorage::DenseComplexInterleaved => {
            operation.complex() && input.element_type == complex
        }
        ProviderStorage::Sparse => false,
    };
    if !supported {
        return reject(
            ProviderRejectionCode::UnsupportedElementType,
            "trigonometry.direct_representation.unsupported",
        );
    }
    let mut output = input.clone();
    output.residency = ProviderResidency::Device;
    if query.outputs.iter().any(|declared| declared != &output) {
        return reject(
            ProviderRejectionCode::InvalidShape,
            "trigonometry.output_representation.mismatch",
        );
    }
    let count = input.checked_element_count();
    if count.is_none() || (wgpu && count.is_some_and(|n| n > u64::from(u32::MAX))) {
        return reject(
            ProviderRejectionCode::ResourceLimit,
            "trigonometry.dispatch.length",
        );
    }
    // Omitted descriptors in the current automatic query still require one
    // materialized result, including calls whose language output is discarded.
    let mut complete = query.clone();
    complete.outputs = vec![output];
    let mut snapshot = dense_capability_snapshot(provider, 1, &[]);
    append(&mut snapshot);
    Some(dense_feasibility(&snapshot, &complete))
}
