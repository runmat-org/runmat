use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, NumericClass, NumericDomain,
    NumericFact, ResidencyFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

const FIELDS: &[&str] = &[
    "RiseTime",
    "TransientTime",
    "SettlingTime",
    "SettlingMin",
    "SettlingMax",
    "Overshoot",
    "Undershoot",
    "Peak",
    "PeakTime",
    "SteadyStateValue",
];

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let args = &request.arguments;
    if args.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-STEPINFO-ARITY",
            "stepinfo requires a model or response samples",
            0,
        ));
    }
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 1)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-STEPINFO-OUTPUTS",
            "stepinfo has one struct output",
            0,
        ));
    }

    let is_model = args
        .first()
        .is_some_and(|input| matches!(input.kind, ValueKindFact::Object(_)));
    if let Some(input) = args.first() {
        let allowed = match &input.kind {
            ValueKindFact::Unknown => true,
            ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
                Some(class) => {
                    class == &standard::TRANSFER_FUNCTION.owned()
                        && (!object.properties_complete
                            || [
                                "Numerator",
                                "Denominator",
                                "Ts",
                                "InputDelay",
                                "OutputDelay",
                            ]
                            .iter()
                            .all(|key| object.properties.contains_key(*key)))
                }
                None => object.class.is_none(),
            },
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Logical => vector(input),
            _ => false,
        };
        if !allowed
            || (is_model && input.shape.element_count().is_some_and(|count| count != 1))
            || matches!(input.residency, ResidencyFact::Remote { .. })
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-STEPINFO-INPUT",
                "input must be a scalar tf model or real numeric response vector",
                0,
            ));
        }
    }

    let option_start = if is_model {
        1
    } else {
        let mut end = 1;
        for (index, argument) in args.iter().enumerate().take(3).skip(1) {
            if matches!(
                argument.kind,
                ValueKindFact::String | ValueKindFact::Character
            ) {
                break;
            }
            if matches!(argument.kind, ValueKindFact::Unknown) {
                break;
            }
            if index == 1 && !real_vector(argument) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STEPINFO-TIME",
                    "t must be a real numeric vector",
                    index,
                ));
            }
            if index == 2 && !real_scalar(argument) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STEPINFO-FINAL",
                    "yfinal must be a real scalar",
                    index,
                ));
            }
            end = index + 1;
        }
        end
    };
    if !is_model && option_start >= 2 {
        if let (Some(y), Some(t)) = (args.first(), args.get(1)) {
            if let (Some(ny), Some(nt)) = (y.shape.element_count(), t.shape.element_count()) {
                if ny != nt {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-STEPINFO-LENGTH",
                        "y and t must have equal sample counts",
                        1,
                    ));
                }
            }
        }
    }
    if args.len().saturating_sub(option_start) % 2 != 0 {
        diagnostics.push(argument_error(
            "RM-CATALOG-STEPINFO-OPTIONS",
            "options require name-value pairs",
            args.len().saturating_sub(1),
        ));
    }
    for index in (option_start..args.len()).step_by(2) {
        if index + 1 >= args.len() {
            break;
        }
        if !matches!(
            args[index].kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-CATALOG-STEPINFO-OPTION-NAME",
                "option name must be text",
                index,
            ));
        }
        if let Some(name) = request.literals.literal_string_at(index) {
            if !name.trim().eq_ignore_ascii_case("SettlingTimeThreshold") {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STEPINFO-OPTION-NAME",
                    "unsupported option",
                    index,
                ));
            }
        }
        if request
            .literals
            .numeric_at(index + 1)
            .is_some_and(|value| !value.is_finite() || value <= 0.0 || value >= 1.0)
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-STEPINFO-THRESHOLD",
                "SettlingTimeThreshold must lie strictly between zero and one",
                index + 1,
            ));
        }
        if !matches!(args[index + 1].kind, ValueKindFact::Unknown) && !real_scalar(&args[index + 1])
        {
            diagnostics.push(argument_error(
                "RM-CATALOG-STEPINFO-THRESHOLD",
                "SettlingTimeThreshold must be a real scalar",
                index + 1,
            ));
        }
    }

    let double = || {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::Scalar,
            StorageFact::Scalar,
        )
    };
    let fields = FIELDS
        .iter()
        .map(|field| ((*field).to_string(), double()))
        .collect::<BTreeMap<_, _>>();
    let output = ValueFact::proven(
        ValueKindFact::Struct(StructFact::scalar(fields, true)),
        ShapeFact::Scalar,
        StorageFact::Scalar,
    );
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn vector(input: &ValueFact) -> bool {
    input
        .shape
        .known_dims()
        .is_none_or(|dims| dims.iter().filter(|dim| dim.is_some_and(|n| n > 1)).count() <= 1)
}

fn real_vector(input: &ValueFact) -> bool {
    matches!(
        input.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) && vector(input)
}

fn real_scalar(input: &ValueFact) -> bool {
    matches!(
        input.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) && input.shape.element_count().is_none_or(|count| count == 1)
}
