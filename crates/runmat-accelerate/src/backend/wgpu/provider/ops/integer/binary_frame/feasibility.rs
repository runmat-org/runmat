//! Side-effect-free feasibility for the exact native integer division frame.

use super::*;
use runmat_accelerate_api::{
    ProviderElementType, ProviderFeasibility, ProviderFeasibilityQuery, ProviderLayout,
    ProviderOperationFamily, ProviderPrecision, ProviderRejectionCode, ProviderResidency,
    ProviderResourceEstimate, ProviderStorage,
};

#[cfg(test)]
#[path = "feasibility/tests.rs"]
mod tests;

pub(in super::super) fn integer_division_feasibility(
    provider: &WgpuProvider,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    feasibility_for_geometry(
        &provider.adapter_limits,
        provider.linear_launch,
        provider.precision(),
        query,
    )
}

fn feasibility_for_geometry(
    limits: &wgpu::Limits,
    launch: FrozenLinearLaunch,
    precision: ProviderPrecision,
    query: &ProviderFeasibilityQuery,
) -> Option<ProviderFeasibility> {
    if query.operation.0 != runmat_builtins::LDIVIDE_CATALOG_ENTRY.identity.name
        && query.operation.0 != runmat_builtins::RDIVIDE_CATALOG_ENTRY.identity.name
    {
        return None;
    }
    if !query
        .inputs
        .iter()
        .any(|input| integer_type(input.element_type).is_some())
    {
        return None;
    }
    let reject = |code, detail| ProviderFeasibility::rejected(code, detail);
    if query.family != ProviderOperationFamily::Elementwise {
        return Some(reject(
            ProviderRejectionCode::UnsupportedOperation,
            "integer_division.family",
        ));
    }
    if query.inputs.len() != 2 || query.outputs.len() > 1 {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "integer_division.arity",
        ));
    }
    let class = query.inputs[0].element_type;
    let Some(integer_type) = integer_type(class) else {
        return Some(reject(
            ProviderRejectionCode::UnsupportedElementType,
            "integer_division.class",
        ));
    };
    if query.inputs.iter().any(|input| {
        input.element_type != class
            || input.storage != ProviderStorage::DenseReal
            || input.layout != ProviderLayout::ColumnMajorContiguous
            || input.residency != ProviderResidency::Device
    }) {
        return Some(reject(
            ProviderRejectionCode::UnsupportedElementType,
            "integer_division.input_representation",
        ));
    }
    let Some((lhs_shape, lhs_len, lhs_bytes)) = input_geometry(&query.inputs[0], precision) else {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "integer_division.lhs_geometry",
        ));
    };
    let Some((rhs_shape, rhs_len, rhs_bytes)) = input_geometry(&query.inputs[1], precision) else {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "integer_division.rhs_geometry",
        ));
    };
    let Ok(geometry) = geometry::plan(
        limits,
        launch,
        "elem_div",
        integer_type,
        geometry::IntegerBinaryInputGeometry {
            shape: &lhs_shape,
            len: lhs_len,
            binding_bytes: lhs_bytes,
        },
        geometry::IntegerBinaryInputGeometry {
            shape: &rhs_shape,
            len: rhs_len,
            binding_bytes: rhs_bytes,
        },
    ) else {
        return Some(reject(
            ProviderRejectionCode::ResourceLimit,
            "integer_division.geometry",
        ));
    };
    if query.outputs.first().is_some_and(|output| {
        output.element_type != class
            || output.storage != ProviderStorage::DenseReal
            || output.layout != ProviderLayout::ColumnMajorContiguous
            || output.residency != ProviderResidency::Device
            || output.shape
                != geometry
                    .broadcast
                    .output_shape
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect::<Vec<_>>()
    }) {
        return Some(reject(
            ProviderRejectionCode::InvalidShape,
            "integer_division.output_representation",
        ));
    }
    // This query proves execution support and the output allocation. It does
    // not claim complete physical cost for the preceding scalar upload.
    Some(ProviderFeasibility::supported(ProviderResourceEstimate {
        transient_bytes: None,
        output_bytes: Some(geometry.output_bytes),
        dispatches: None,
    }))
}

fn input_geometry(
    input: &runmat_accelerate_api::ProviderRepresentation,
    precision: ProviderPrecision,
) -> Option<(Vec<usize>, usize, u64)> {
    let shape = input
        .shape
        .iter()
        .map(|&extent| usize::try_from(extent).ok())
        .collect::<Option<Vec<_>>>()?;
    let len = shape
        .iter()
        .try_fold(1usize, |product, &extent| product.checked_mul(extent))?;
    // Empty native uploads use the selected provider precision's one-element
    // buffer; live empty binary execution emits no shader or storage binding.
    let bytes = crate::backend::wgpu_storage::checked_upload_allocation_bytes(input, precision)?;
    Some((shape, len, bytes))
}

fn integer_type(class: ProviderElementType) -> Option<IntegerElementType> {
    Some(match class {
        ProviderElementType::I8 => IntegerElementType::I8,
        ProviderElementType::I16 => IntegerElementType::I16,
        ProviderElementType::I32 => IntegerElementType::I32,
        ProviderElementType::I64 => IntegerElementType::I64,
        ProviderElementType::U8 => IntegerElementType::U8,
        ProviderElementType::U16 => IntegerElementType::U16,
        ProviderElementType::U32 => IntegerElementType::U32,
        ProviderElementType::U64 => IntegerElementType::U64,
        _ => return None,
    })
}
