use crate::catalog::inference::{argument_error, finish_fixed, numeric_kind};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    CallInference, CallRequest, DynamicReason, NumericClass, NumericDomain, ShapeFact, StorageFact,
    ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(2..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-COLON-ARITY",
            "colon takes two or three scalar operands",
            0,
        ));
    }
    let mut integer_class = None;
    for (index, argument) in request.arguments.iter().enumerate() {
        if argument
            .shape
            .element_count()
            .is_some_and(|count| count != 1)
        {
            diagnostics.push(argument_error(
                "RM-COLON-SCALAR",
                "colon operands must be scalar",
                index,
            ));
        }
        if let Some(class) = argument
            .numeric()
            .and_then(|numeric| numeric.class.integer_class())
        {
            let class = NumericClass::from(class);
            if integer_class.is_some_and(|previous| previous != class) {
                diagnostics.push(argument_error(
                    "RM-COLON-INTEGER-CLASS",
                    "typed integer operands must share one integer class",
                    index,
                ));
            }
            integer_class = Some(class);
        }
    }
    let char_bounds = request.arguments.len() >= 2
        && request.arguments[0].kind == ValueKindFact::Character
        && request
            .arguments
            .last()
            .is_some_and(|value| value.kind == ValueKindFact::Character);
    let kind = if let Some(class) = integer_class {
        numeric_kind(class, NumericDomain::Real)
    } else if char_bounds {
        ValueKindFact::Character
    } else {
        numeric_kind(NumericClass::Double, NumericDomain::Real)
    };
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = kind;
    output.shape = ShapeFact::from(vec![Some(1), None]);
    output.storage = StorageFact::Unknown;
    finish_fixed(entry, request, output, diagnostics)
}
