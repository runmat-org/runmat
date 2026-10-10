//! Host evaluation is a domain decision, not a consequence of a missing hook.
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

use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

/// A pre-effect decline may leave automatically placed operands on the host.
/// An explicit operand needs a certified canonical materialization form and an
/// exact resident result; an unsupported provider hook alone proves neither.
pub(in crate::builtins::math::elementwise::binary_arithmetic) struct HostFallbackAdmission {
    borrowed: Vec<GpuTensorHandle>,
    restore_source: Option<GpuTensorHandle>,
    expected_integer: Option<(BuiltinIntegerClass, Vec<usize>)>,
    builtin: &'static str,
}

impl HostFallbackAdmission {
    pub(in crate::builtins::math::elementwise::binary_arithmetic) fn new(
        entry: &'static BuiltinCatalogEntry,
        left: &Value,
        right: &Value,
    ) -> BuiltinResult<Self> {
        let builtin = entry.identity.name;
        let resident = [left, right]
            .into_iter()
            .filter_map(|value| match value {
                Value::GpuTensor(handle) => Some(handle),
                _ => None,
            })
            .collect::<Vec<_>>();
        let borrowed = resident
            .iter()
            .map(|handle| (*handle).clone())
            .collect::<Vec<_>>();
        let explicit = resident
            .iter()
            .any(|handle| runmat_accelerate_api::handle_is_explicit(handle));
        let certified_integer = integer_materialization(entry, left, right).map_err(|detail| {
            build_runtime_error(detail)
                .with_builtin(builtin)
                .with_gpu_gather_retry(GpuGatherRetry::Never)
                .build()
        })?;
        if explicit && certified_integer.is_none() {
            return Err(explicit_provider_required(builtin));
        }
        if explicit {
            for handle in &resident {
                handle.allocation_lease().validate_live().map_err(|error| {
                    build_runtime_error(format!("{builtin}: resident source lease is unavailable"))
                        .with_builtin(builtin)
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .with_source(error)
                        .build()
                })?;
            }
        }
        let restore_source = if explicit {
            gpu_helpers::select_resident_output_source(borrowed.iter().cloned(), builtin)?
        } else {
            None
        };
        if let (Some(source), Some((class, shape))) = (&restore_source, &certified_integer) {
            certify_integer_publication(source, *class, shape, builtin)?;
        }
        Ok(Self {
            borrowed,
            restore_source,
            expected_integer: if explicit { certified_integer } else { None },
            builtin,
        })
    }

    pub(in crate::builtins::math::elementwise::binary_arithmetic) fn finish(
        self,
        host: Value,
    ) -> BuiltinResult<Value> {
        let Some(source) = &self.restore_source else {
            return Ok(host);
        };
        let host = match host {
            value @ Value::Int(_) => Value::Tensor(
                tensor::value_into_tensor_for(self.builtin, value).map_err(|error| {
                    build_runtime_error(error)
                        .with_builtin(self.builtin)
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build()
                })?,
            ),
            other => other,
        };
        if let Some((class, shape)) = &self.expected_integer {
            let Value::Tensor(result) = &host else {
                return Err(explicit_provider_required(self.builtin));
            };
            if result.shape() != shape
                || result
                    .integer_storage()
                    .map(|storage| storage.integer_class())
                    != Some(integer_class_from_catalog(*class))
            {
                return Err(explicit_provider_required(self.builtin));
            }
        }
        let protected = self.borrowed.iter().collect::<Vec<_>>();
        let output = gpu_helpers::restore_class_preserving_value_with_operands(
            source,
            host,
            self.builtin,
            gpu_helpers::RestoreOperandContext::new(&protected, &protected),
        )?;
        if !matches!(output, Value::GpuTensor(_)) {
            return Err(explicit_provider_required(self.builtin));
        }
        Ok(output)
    }
}

fn integer_materialization(
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

fn certify_integer_publication(
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

fn integer_class_from_catalog(class: BuiltinIntegerClass) -> runmat_types::IntegerClass {
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
        Value::GpuTensor(handle) => super::scalar::is_scalar_shape(handle.shape()),
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

fn explicit_provider_required(builtin: &'static str) -> crate::RuntimeError {
    build_runtime_error(format!(
        "{builtin}: explicit GPU input requires an admitted provider route"
    ))
    .with_builtin(builtin)
    .with_identifier("RunMat:gpu:ExplicitProviderRequired")
    .with_gpu_gather_retry(GpuGatherRetry::Never)
    .build()
}

#[cfg(test)]
#[path = "fallback_tests.rs"]
mod tests;
