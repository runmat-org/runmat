//! Sum's already-admitted data boundary; controls never become provider operands.
use super::{apply_output_template, execute_raw, resolve_dims, InputMeta, ParsedArguments};
use crate::context::preparation::{
    execute_current_request, AdmittedBuiltinPreparation, AdmittedSumReduction,
    BuiltinPreparationAdmission,
};
use crate::BuiltinResult;
use runmat_accelerate_api::{handle_integer_type, handle_is_logical, handle_precision};
use runmat_builtins::{
    BuiltinBindingIdentity, BuiltinPreparationFormId, SumOutputClass, SUM_CATALOG_ENTRY,
};
use runmat_types::{NumericClass, NumericDomain};
use runmat_value::Value;

#[cfg(test)]
mod tests;

pub(super) fn eligible(value: &Value) -> bool {
    match value {
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
    value: Value,
    parsed: ParsedArguments,
    input_meta: InputMeta,
    input_count: usize,
) -> BuiltinResult<Value> {
    let controls = controls(&value, &parsed)?;
    let request = AdmittedBuiltinPreparation::new(
        BuiltinPreparationAdmission {
            binding: BuiltinBindingIdentity {
                builtin: SUM_CATALOG_ENTRY.identity,
                variant: "default",
            },
            form: BuiltinPreparationFormId(0),
            // The final signature describes the parser's admitted, variadic
            // option grammar. All earlier signatures remain public forms.
            signature: 8,
            input_count,
            requested_outputs: crate::output_count::current_output_count(),
        },
        vec![value],
    )?
    .with_sum_reduction(controls)?;
    execute_current_request(request, |mut operands| async move {
        let value = operands.remove(0);
        let raw = execute_raw(value, &parsed).await?;
        apply_output_template(raw, &parsed.output, &input_meta).await
    })
    .await
}

fn controls(value: &Value, parsed: &ParsedArguments) -> BuiltinResult<AdmittedSumReduction> {
    let (shape, class, domain) = match value {
        Value::Tensor(tensor) => (
            tensor.shape.as_slice(),
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
        axes: resolved.dims_in_bounds,
        missing: parsed.nan_mode,
        output: parsed.output,
        input_class: class,
        input_domain: domain,
        accumulation_class,
        publication_class: parsed.output.resolve_numeric_class(class),
    })
}
