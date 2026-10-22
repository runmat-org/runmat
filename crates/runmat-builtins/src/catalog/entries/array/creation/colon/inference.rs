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
    output.shape = ShapeFact::from(vec![Some(1), exact_double_length(request)]);
    output.storage = StorageFact::Unknown;
    finish_fixed(entry, request, output, diagnostics)
}

// Integer-valued quotients in this range are unaffected by the runtime's
// endpoint tolerance. Other floating endpoints retain a dynamic length rather
// than treating an approximate quotient as a proven extent.
fn exact_double_length(request: &CallRequest) -> Option<usize> {
    use runmat_types::{LiteralValue, NumericFact};
    if !request.arguments.iter().all(|argument| {
        argument.kind
            == ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
    }) {
        return None;
    }
    let (start, step, end) = match request.literals.literal_args.as_slice() {
        [LiteralValue::Number(start), LiteralValue::Number(end)] => (*start, 1.0, *end),
        [LiteralValue::Number(start), LiteralValue::Number(step), LiteralValue::Number(end)] => {
            (*start, *step, *end)
        }
        _ => return None,
    };
    if !start.is_finite() || !step.is_finite() || !end.is_finite() || step == 0.0 {
        return None;
    }
    let quotient = (end - start) / step;
    if !quotient.is_finite()
        || quotient < 0.0
        || quotient.fract() != 0.0
        || quotient >= (1u64 << 52) as f64
        || quotient >= usize::MAX as f64
    {
        return None;
    }
    (quotient as usize).checked_add(1)
}
