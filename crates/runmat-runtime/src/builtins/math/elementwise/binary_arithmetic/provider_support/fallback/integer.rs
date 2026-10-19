use super::explicit_provider_required;
use crate::builtins::common::{gpu_helpers, tensor};
use crate::BuiltinResult;
use runmat_accelerate_api::{
    class_preserving_upload_query, GpuTensorHandle, GpuTensorStorage, ProviderElementType,
    ProviderFeasibility, ProviderLayout, ProviderRepresentation, ProviderResidency,
    ProviderStorage,
};
use runmat_builtins::{
    BuiltinCatalogEntry, BuiltinIntegerBackendRule, BuiltinIntegerClass,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerOutputClassRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_value::Value;

pub(super) fn integer_materialization(
    entry: &BuiltinCatalogEntry,
    left: &Value,
    right: &Value,
) -> Result<Option<(BuiltinIntegerClass, Vec<usize>)>, String> {
    use runmat_builtins::BuiltinPreparationOperation;
    // This checks the catalog's two-data-argument declaration. Integer calls
    // are not minted by the floating-pair preparation owner; the integer
    // descriptor and canonical runtime class validator below are their own
    // domain admission, not an inferred floating preparation receipt.
    let declared_binary_form = matches!(entry.placement.preparation,
        runmat_builtins::BuiltinOperandPreparation::Forms(forms)
            if forms.iter().any(|form| form.id == runmat_builtins::BuiltinPreparationFormId(0)
                && form.data_arguments == [0, 1]
                && matches!(form.operation,
                    BuiltinPreparationOperation::ElementwiseBinary
                        | BuiltinPreparationOperation::RealPower)));
    let operands = [left, right];
    let classes = operands.map(integer_class);
    let integer_input = classes.iter().any(Option::is_some);
    let represented = operands.iter().all(|value| match value {
        Value::Num(_) | Value::Int(_) | Value::Tensor(_) => true,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
                && !runmat_accelerate_api::handle_is_logical(handle)
        }
        _ => false,
    });
    if !(declared_binary_form && integer_input && represented) {
        return Ok(None);
    }
    crate::builtins::math::elementwise::integer_arithmetic::validate_integer_binary_inputs(
        left,
        right,
        scalar(left),
        scalar(right),
        entry.identity.name,
    )?;
    let (Some(left_shape), Some(right_shape)) = (shape(left), shape(right)) else {
        return Ok(None);
    };
    let plan = crate::builtins::common::broadcast::BroadcastPlan::new(left_shape, right_shape)?;
    let catalog_allows =
        entry.integer_capabilities.iter().any(|capability| {
            capability.backend == BuiltinIntegerBackendRule::GatherFallback
                && capability.computation_domain == BuiltinIntegerComputationDomain::ExactInteger
                && capability.output_class == BuiltinIntegerOutputClassRule::PreserveNondoubleInput
                && capability.overload == BuiltinIntegerOverloadKind::BroadcastCompatible
                && capability.inputs.len() == operands.len()
                && capability.inputs.iter().zip(classes).enumerate().all(
                    |(index, (input, class))| match class {
                        Some(class) => {
                            input.availability == BuiltinIntegerInputAvailability::Documented
                                && input.classes.contains(&class)
                        }
                        None => {
                            matches!(
                                operands[index],
                                Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_)
                            ) && scalar(operands[index])
                                && input.scalar_double == BuiltinIntegerScalarDoubleRule::Allowed
                        }
                    },
                )
        });
    let output_class = classes.into_iter().flatten().next();
    Ok(if catalog_allows {
        output_class.map(|class| (class, plan.output_shape().to_vec()))
    } else {
        None
    })
}

pub(super) fn certify_integer_publication(
    source: &GpuTensorHandle,
    class: BuiltinIntegerClass,
    shape: &[usize],
    builtin: &'static str,
) -> BuiltinResult<()> {
    let provider = gpu_helpers::exact_provider_for_handle(source)
        .ok_or_else(|| explicit_provider_required(builtin))?;
    let representation = ProviderRepresentation {
        element_type: provider_element_type(class),
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape: shape
            .iter()
            .map(|&extent| u64::try_from(extent))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| explicit_provider_required(builtin))?,
        residency: ProviderResidency::Device,
    };
    let query = class_preserving_upload_query(vec![representation])
        .ok_or_else(|| explicit_provider_required(builtin))?;
    let ProviderFeasibility::Supported { estimate } = provider.query_feasibility(&query) else {
        return Err(explicit_provider_required(builtin));
    };
    let Some(physical) = provider.upload_physical_evidence(&query) else {
        return Err(explicit_provider_required(builtin));
    };
    if estimate.output_bytes != Some(physical.output_requested_bytes) {
        return Err(explicit_provider_required(builtin));
    }
    Ok(())
}

fn provider_element_type(class: BuiltinIntegerClass) -> ProviderElementType {
    match class {
        BuiltinIntegerClass::Int8 => ProviderElementType::I8,
        BuiltinIntegerClass::Int16 => ProviderElementType::I16,
        BuiltinIntegerClass::Int32 => ProviderElementType::I32,
        BuiltinIntegerClass::Int64 => ProviderElementType::I64,
        BuiltinIntegerClass::Uint8 => ProviderElementType::U8,
        BuiltinIntegerClass::Uint16 => ProviderElementType::U16,
        BuiltinIntegerClass::Uint32 => ProviderElementType::U32,
        BuiltinIntegerClass::Uint64 => ProviderElementType::U64,
    }
}

pub(super) fn integer_class_from_catalog(class: BuiltinIntegerClass) -> runmat_types::IntegerClass {
    match class {
        BuiltinIntegerClass::Int8 => runmat_types::IntegerClass::Int8,
        BuiltinIntegerClass::Int16 => runmat_types::IntegerClass::Int16,
        BuiltinIntegerClass::Int32 => runmat_types::IntegerClass::Int32,
        BuiltinIntegerClass::Int64 => runmat_types::IntegerClass::Int64,
        BuiltinIntegerClass::Uint8 => runmat_types::IntegerClass::UInt8,
        BuiltinIntegerClass::Uint16 => runmat_types::IntegerClass::UInt16,
        BuiltinIntegerClass::Uint32 => runmat_types::IntegerClass::UInt32,
        BuiltinIntegerClass::Uint64 => runmat_types::IntegerClass::UInt64,
    }
}

fn integer_class(value: &Value) -> Option<BuiltinIntegerClass> {
    let class = match value {
        Value::Int(integer) => Some(integer.integer_class()),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .map(|storage| storage.integer_class()),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_class(handle),
        _ => None,
    }?;
    Some(match class {
        runmat_types::IntegerClass::Int8 => BuiltinIntegerClass::Int8,
        runmat_types::IntegerClass::Int16 => BuiltinIntegerClass::Int16,
        runmat_types::IntegerClass::Int32 => BuiltinIntegerClass::Int32,
        runmat_types::IntegerClass::Int64 => BuiltinIntegerClass::Int64,
        runmat_types::IntegerClass::UInt8 => BuiltinIntegerClass::Uint8,
        runmat_types::IntegerClass::UInt16 => BuiltinIntegerClass::Uint16,
        runmat_types::IntegerClass::UInt32 => BuiltinIntegerClass::Uint32,
        runmat_types::IntegerClass::UInt64 => BuiltinIntegerClass::Uint64,
    })
}

fn scalar(value: &Value) -> bool {
    match value {
        Value::Num(_) | Value::Int(_) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::GpuTensor(handle) => super::super::scalar::is_scalar_shape(handle.shape()),
        _ => false,
    }
}

fn shape(value: &Value) -> Option<&[usize]> {
    match value {
        Value::Num(_) | Value::Int(_) => Some(&[1, 1]),
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        _ => None,
    }
}
