use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, NumericClass, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() != 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLY-ARITY",
            "poly requires one operand",
            0,
        ));
    }
    let input = request.arguments.first();
    if let Some(input) = input {
        if matches!(input.residency, ResidencyFact::Remote { .. }) {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLY-RESIDENCY",
                "poly cannot consume remote/distributed input",
                0,
            ));
        }
        if !matches!(
            input.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single | NumericClass::Double,
                ..
            }) | ValueKindFact::Unknown
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-POLY-CLASS",
                "poly requires single or double real/complex input",
                0,
            ));
        }
        if let Some(dims) = input.shape.known_dims() {
            if dims.len() > 2
                || (dims.len() == 2
                    && dims[0].is_some()
                    && dims[1].is_some()
                    && dims[0] != Some(1)
                    && dims[1] != Some(1)
                    && dims[0] != dims[1])
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-POLY-SHAPE",
                    "poly requires a vector or square matrix",
                    0,
                ));
            }
        }
    }
    let count = request.outputs.requested.known_count();
    if count.is_some_and(|count| count > 1) {
        diagnostics.push(argument_error(
            "RM-CATALOG-POLY-OUTPUTS",
            "poly returns one coefficient row",
            0,
        ));
    }
    let mut contract = CallContract::fixed(if count == Some(0) {
        Vec::new()
    } else {
        vec![output_fact(input)]
    });
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = count.is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn output_fact(input: Option<&ValueFact>) -> ValueFact {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.shape = ShapeFact::from(vec![Some(1), None]);
    output.storage = StorageFact::Dense;
    output.residency = ResidencyFact::Host;
    let Some(input) = input else { return output };
    if matches!(input.residency, ResidencyFact::Remote { .. }) {
        return output;
    }
    if let ValueKindFact::Numeric(numeric) = input.kind {
        if matches!(numeric.class, NumericClass::Single | NumericClass::Double) {
            output.kind = ValueKindFact::Numeric(numeric);
        }
    }
    if let Some(dims) = input.shape.known_dims() {
        let degree = match dims.as_slice() {
            [] => Some(1),
            [value] => *value,
            [rows, cols] => match (*rows, *cols) {
                (Some(rows), Some(cols)) if rows == 1 || cols == 1 => rows.checked_mul(cols),
                (Some(rows), Some(cols)) if rows == cols => Some(rows),
                _ => None,
            },
            _ => None,
        };
        if let Some(length) = degree.and_then(|degree| degree.checked_add(1)) {
            output.shape = ShapeFact::from(vec![Some(1), Some(length)]);
        }
    }
    output
}
