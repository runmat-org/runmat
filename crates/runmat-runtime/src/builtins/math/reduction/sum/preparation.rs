//! Sum's already-admitted data boundary; controls never become provider operands.
use super::{
    apply_output_template, execute_raw, resolve_dims, sum_native_integer, InputMeta,
    ParsedArguments,
};
use crate::context::preparation::{
    execute_request, AdmittedBuiltinPreparation, AdmittedSumReduction, BuiltinPreparationAdmission,
};
use crate::context::RuntimeContext;
use crate::execution::RootedValue;
use crate::BuiltinResult;
use runmat_accelerate_api::{handle_integer_type, handle_is_logical, handle_precision};
use runmat_builtins::{BuiltinBindingIdentity, BuiltinOperandPreparation, SumOutputClass};
use runmat_types::{NumericClass, NumericDomain};
use runmat_value::Value;

#[cfg(test)]
#[path = "preparation/tests.rs"]
mod tests;

pub(super) fn eligible(value: &Value, output: SumOutputClass) -> bool {
    match value {
        Value::Int(_) => output == SumOutputClass::Native,
        Value::Tensor(_) => true,
        Value::GpuTensor(handle) => {
            handle_is_logical(handle)
                || handle_integer_type(handle).is_some()
                || handle_precision(handle).is_some()
        }
        _ => false,
    }
}

pub(super) async fn execute(
    runtime: Option<&RuntimeContext>,
    value: Value,
    parsed: ParsedArguments,
    input_meta: InputMeta,
    input_count: usize,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
) -> BuiltinResult<RootedValue> {
    let controls = controls(&value, &parsed)?;
    let entry = runmat_builtins::sum_invocation_entry(parsed.invocation);
    let BuiltinOperandPreparation::Forms([form]) = entry.placement.preparation else {
        unreachable!("sum invocation has one canonical data-only preparation form")
    };
    let [signature] = form.signatures else {
        unreachable!("sum invocation form has one canonical variadic signature")
    };
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: entry.identity,
                variant: form.binding_variant,
            },
            form: form.id,
            signature: *signature,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
    )?
    .with_sum_reduction(controls)?
    .with_prepared_call_observation(observation);
    execute_request(runtime, request, |mut operands| async move {
        let value = operands.remove(0);
        if parsed.output == SumOutputClass::Native {
            if let Some(result) = sum_native_integer(&value, &parsed)? {
                return RootedValue::new(result).map_err(|cause| cause.into_runtime_error());
            }
        }
        let raw = execute_raw(value, &parsed).await?;
        let published = apply_output_template(raw, &parsed.output, &input_meta).await?;
        RootedValue::new(published).map_err(|cause| cause.into_runtime_error())
    })
    .await
}

fn controls(value: &Value, parsed: &ParsedArguments) -> BuiltinResult<AdmittedSumReduction> {
    let scalar_shape = [1, 1];
    let (shape, class, domain) = match value {
        Value::Int(integer) => (
            scalar_shape.as_slice(),
            NumericClass::from_class_identity(&integer.numeric_dtype().class_identity())
                .expect("all integer scalar dtypes have canonical classes"),
            NumericDomain::Real,
        ),
        Value::Tensor(tensor) => (
            tensor.shape(),
            NumericClass::from_class_identity(&tensor.numeric_dtype().class_identity())
                .expect("all numeric tensor dtypes have canonical classes"),
            NumericDomain::Real,
        ),
        Value::GpuTensor(handle) => {
            let class = if handle_is_logical(handle) {
                NumericClass::Double
            } else if let Some(integer) = handle_integer_type(handle) {
                NumericClass::from(runmat_types::IntegerClass::from(integer))
            } else {
                match handle_precision(handle) {
                    Some(runmat_accelerate_api::ProviderPrecision::F32) => NumericClass::Single,
                    Some(runmat_accelerate_api::ProviderPrecision::F64) => NumericClass::Double,
                    None => unreachable!("eligible floating GPU input has a precision"),
                }
            };
            let domain = if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                NumericDomain::Complex
            } else {
                NumericDomain::Real
            };
            (handle.shape(), class, domain)
        }
        _ => unreachable!("only eligible data enters sum preparation"),
    };
    let resolved = resolve_dims(shape, &parsed.selection)?;
    let accumulation_class =
        if parsed.output == SumOutputClass::Native && class.integer_class().is_some() {
            class
        } else {
            NumericClass::Double
        };
    Ok(AdmittedSumReduction {
        invocation: parsed.invocation,
        axes: resolved.dims_in_bounds,
        missing: parsed.nan_mode,
        output: parsed.output,
        input_class: class,
        input_domain: domain,
        accumulation_class,
        publication_class: parsed.output.resolve_numeric_class(class),
    })
}
