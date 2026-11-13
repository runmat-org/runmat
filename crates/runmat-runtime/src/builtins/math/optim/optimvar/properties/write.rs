use super::*;
use runmat_value::optimization_variable::OptimizationVariableDomain;
use runmat_value::{HostNumericBuffer, NumericStorage};

pub(in crate::builtins::math::optim::optimvar) async fn set(
    view: &View,
    property: Property,
    rhs: &Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValue> {
    ensure_writable(view, property)?;
    let old = view.definition();
    let mut domain = old.domain();
    let mut lower = old.bound_storage(OptimizationBoundRole::Lower).clone();
    let mut upper = old.bound_storage(OptimizationBoundRole::Upper).clone();
    let mut index_names = None;
    match property {
        Property::Name => unreachable!("read-only admission"),
        Property::Type => {
            domain = OptimizationVariableDomain::parse(&super::super::parse::text(
                rhs,
                cooperation.runtime(),
            )?)
            .ok_or_else(|| invalid("unsupported optimization-variable Type"))?;
        }
        Property::IndexNames => {
            index_names =
                Some(super::super::parse::index_names(rhs, old.shape(), cooperation).await?);
        }
        Property::LowerBound | Property::UpperBound => {
            let role = if property == Property::LowerBound {
                OptimizationBoundRole::Lower
            } else {
                OptimizationBoundRole::Upper
            };
            let replacement =
                super::super::parse::bound(rhs, view.selection().shape(), cooperation).await?;
            if view.selection().element_count() == 0 {
                return RootedValue::in_optional_context(
                    Value::Object(view.reference().clone()),
                    cooperation.runtime(),
                );
            }
            let storage = if view.selection().is_whole() {
                replacement
            } else {
                // This is the actual new immutable bound generation. The
                // original remains available through every suspended check.
                let mut values =
                    runmat_value::exact_capacity_vec(old.element_count()).map_err(data_error)?;
                for position in 0..old.element_count() {
                    values.push(
                        old.bound_value(role, position)
                            .ok_or_else(|| invalid("incomplete original bounds"))?,
                    );
                    cooperation.completed(1).await?;
                }
                for position in 0..view.selection().element_count() {
                    let index = view
                        .selection()
                        .definition_index(position)
                        .ok_or_else(|| invalid("invalid reference index"))?;
                    values[index] = match &replacement {
                        OptimizationBoundStorage::Uniform(value) => *value,
                        OptimizationBoundStorage::PerElement(buffer) => {
                            buffer.as_f64_slice().expect("admitted double bounds")[position]
                        }
                    };
                    cooperation.completed(1).await?;
                }
                OptimizationBoundStorage::PerElement(
                    runmat_value::optimization_variable::OptimizationBoundArray::try_from_tensor(
                        runmat_value::Tensor::from_shared_host_buffer(
                            HostNumericBuffer::from_numeric_storage(NumericStorage::F64(values)),
                            old.shape().clone(),
                        )
                        .map_err(invalid)?,
                    )
                    .map_err(|_| invalid("replacement bounds must be full real double storage"))?,
                )
            };
            if role == OptimizationBoundRole::Lower {
                lower = storage;
            } else {
                upper = storage;
            }
        }
    }
    let labels = index_names.unwrap_or_else(|| old.index_names().clone());
    let candidate = OptimizationVariableCandidate {
        name: old.name_owner().clone(),
        domain,
        shape: old.shape().clone(),
        index_names: labels,
        lower,
        upper,
    };
    let data = super::super::construction::validate(candidate, cooperation).await?;
    let next = ObjectInstance::new(CLASS.owned())
        .with_optimization_variable_payload(OptimizationVariablePayload::Definition(data));
    let returned = RootedValue::in_optional_context(
        Value::Object(view.reference().clone()),
        cooperation.runtime(),
    )?;
    cooperation.completed(0).await?;
    view.publish(next)?;
    Ok(returned)
}

pub(in crate::builtins::math::optim::optimvar) fn ensure_writable(
    view: &View,
    property: Property,
) -> BuiltinResult<()> {
    if property == Property::Name
        || (matches!(property, Property::Type | Property::IndexNames)
            && view.selection().role() == OptimizationVariableReferenceRole::Reference)
    {
        let mut error = crate::runtime_descriptor_error_with_detail(
            "optimvar",
            &runmat_builtins::OPTIMVAR_ERROR_READ_ONLY,
            "array-wide property is read-only on this reference",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(error);
    }
    Ok(())
}
