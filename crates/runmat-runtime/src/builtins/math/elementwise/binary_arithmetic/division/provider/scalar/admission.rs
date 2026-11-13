//! Pre-upload feasibility for native integer scalar division.

use super::super::DivisionContext;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, ProviderFeasibilityQuery, ProviderLayout,
    ProviderOperationFamily, ProviderOperationIdentity, ProviderRepresentation, ProviderResidency,
    ProviderStorage, ProviderWorkload,
};

pub(super) fn native_integer_division_supported(
    provider: &dyn AccelProvider,
    context: DivisionContext,
    resident: &GpuTensorHandle,
    scalar_is_numerator: bool,
) -> bool {
    let Some(integer_type) = runmat_accelerate_api::handle_integer_type(resident) else {
        return false;
    };
    let Some(shape) = resident
        .shape()
        .iter()
        .map(|&extent| u64::try_from(extent).ok())
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let class = integer_type.into();
    let resident_representation = ProviderRepresentation {
        element_type: class,
        storage: ProviderStorage::DenseReal,
        layout: if runmat_accelerate_api::handle_is_transposed(resident) {
            ProviderLayout::Strided
        } else {
            ProviderLayout::ColumnMajorContiguous
        },
        shape: shape.clone(),
        residency: ProviderResidency::Device,
    };
    let scalar_representation = ProviderRepresentation {
        element_type: class,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: vec![1, 1],
        residency: ProviderResidency::Device,
    };
    let inputs = if scalar_is_numerator {
        vec![scalar_representation, resident_representation.clone()]
    } else {
        vec![resident_representation.clone(), scalar_representation]
    };
    provider
        .query_feasibility(&ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(context.identity.name),
            family: ProviderOperationFamily::Elementwise,
            inputs,
            outputs: vec![ProviderRepresentation {
                layout: ProviderLayout::ColumnMajorContiguous,
                ..resident_representation
            }],
            workload: ProviderWorkload::default(),
        })
        .is_supported()
}
