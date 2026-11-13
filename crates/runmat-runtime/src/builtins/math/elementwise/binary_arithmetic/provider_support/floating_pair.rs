//! Direct mixed floating arithmetic admitted by the provider's canonical contract.
//! Inputs retain their original representation: Double intermediates are not
//! interchangeable with rounding both operands to the Single publication class.
use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorLayout, GpuTensorStorage, ProviderElementType,
    ProviderFeasibility, ProviderFeasibilityQuery, ProviderLayout, ProviderOperationFamily,
    ProviderOperationIdentity, ProviderPrecision, ProviderRepresentation, ProviderResidency,
    ProviderStorage, ProviderWorkload,
};
use runmat_builtins::BuiltinCatalogIdentity;
use runmat_types::{NumericClass, OperatorKind, RealFloatingRecipe};

use crate::builtins::common::{broadcast::BroadcastPlan, gpu_helpers};

pub(in crate::builtins::math::elementwise::binary_arithmetic) struct FloatingPairAdmission {
    recipe: RealFloatingRecipe,
    output_shape: Vec<usize>,
}

impl FloatingPairAdmission {
    pub(in crate::builtins::math::elementwise::binary_arithmetic) fn admit(
        identity: BuiltinCatalogIdentity,
        operator: OperatorKind,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
        provider: &dyn AccelProvider,
    ) -> Option<Self> {
        let left_rep = representation(left)?;
        let right_rep = representation(right)?;
        if left_rep.element_type == right_rep.element_type {
            return None;
        }
        let owner = super::common_owner(left, right)?;
        if !runmat_accelerate_api::same_provider_instance(owner, provider) {
            return None;
        }
        let class = |element| match element {
            ProviderElementType::F32 => Some(NumericClass::Single),
            ProviderElementType::F64 => Some(NumericClass::Double),
            _ => None,
        };
        let recipe = RealFloatingRecipe::for_classes(
            operator,
            class(left_rep.element_type)?,
            class(right_rep.element_type)?,
        )?;
        let output_shape = BroadcastPlan::new(left.shape(), right.shape())
            .ok()?
            .output_shape()
            .to_vec();
        let output = ProviderRepresentation {
            element_type: precision(recipe.result_class())?.into(),
            storage: ProviderStorage::DenseReal,
            layout: ProviderLayout::ColumnMajorContiguous,
            residency: ProviderResidency::Device,
            shape: output_shape
                .iter()
                .map(|&n| u64::try_from(n).ok())
                .collect::<Option<_>>()?,
        };
        let query = ProviderFeasibilityQuery {
            operation: ProviderOperationIdentity::new(identity.name),
            family: ProviderOperationFamily::Elementwise,
            inputs: vec![left_rep, right_rep],
            outputs: vec![output],
            workload: ProviderWorkload::default(),
        };
        // Supported describes this exact operation, original input classes and
        // broadcast shape. No repmat, pre-cast or scalar extraction follows it.
        matches!(
            provider.query_feasibility(&query),
            ProviderFeasibility::Supported { .. }
        )
        .then_some(Self {
            recipe,
            output_shape,
        })
    }

    pub(in crate::builtins::math::elementwise::binary_arithmetic) fn accepts(
        &self,
        output: &GpuTensorHandle,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
        provider: &dyn AccelProvider,
    ) -> bool {
        representation(output).is_some()
            && output.shape() == self.output_shape
            && output.device_id() == left.device_id()
            && gpu_helpers::exact_provider_for_handle(output)
                .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
            && !output.shares_allocation_with(left)
            && !output.shares_allocation_with(right)
            && runmat_accelerate_api::handle_precision(output)
                == precision(self.recipe.result_class())
    }
}

fn precision(class: NumericClass) -> Option<ProviderPrecision> {
    match class {
        NumericClass::Single => Some(ProviderPrecision::F32),
        NumericClass::Double => Some(ProviderPrecision::F64),
        _ => None,
    }
}

fn representation(handle: &GpuTensorHandle) -> Option<ProviderRepresentation> {
    let precision = runmat_accelerate_api::handle_precision(handle)?;
    if handle.allocation_lease().validate_live().is_err()
        || handle.view().layout() != GpuTensorLayout::Contiguous
        || handle.physical().storage() != GpuTensorStorage::Real
        || handle.descriptor().storage != Some(GpuTensorStorage::Real)
        || handle.physical().element_type().precision() != Some(precision)
        || handle.descriptor().element_type != Some(handle.physical().element_type())
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
        || gpu_helpers::expected_handle_numeric_element_type(handle).is_err()
        || !gpu_helpers::gpu_class_metadata_matches(handle, Some(precision), None, false)
    {
        return None;
    }
    Some(ProviderRepresentation {
        element_type: precision.into(),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        residency: ProviderResidency::Device,
        shape: handle
            .shape()
            .iter()
            .map(|&n| u64::try_from(n).ok())
            .collect::<Option<_>>()?,
    })
}
