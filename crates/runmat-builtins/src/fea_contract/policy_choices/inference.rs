//! Static argument checks for time-sampled and callable-backed signals.
//!
//! Diagnostics are raised only for facts proven incompatible with the runtime
//! admission rules; unknown kinds, shapes, and option names never error. Codes
//! shared with fixed-arity policy constructors keep their meaning.

use crate::{
    catalog::{argument_error, finish_fixed},
    fea_contract::{
        constructor_inference::host_text,
        policy_types::{
            classes::{SIGNAL_DOMAIN, SIGNAL_EXTRAPOLATION, SIGNAL_INTERPOLATION, SIGNAL_SAMPLING},
            inference::{policy_object, ARITY, NUMERIC, RESIDENT, SHAPE, TEXT, TYPED},
        },
    },
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, DimensionFact, InferenceDiagnostic, NumericDomain, ResidencyFact,
    ShapeFact, StaticClassIdentity, ValueFact, ValueKindFact,
};

pub const CALLABLE: &str = "RM-CATALOG-FEA-SIGNAL-CALLABLE";
pub const OPTION: &str = "RM-CATALOG-FEA-SIGNAL-OPTION";
pub const ARGUMENTS: &str = "RM-CATALOG-FEA-SIGNAL-ARGUMENTS";

/// Ordinate layout of one signal class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignalShape {
    Scalar,
    Vector3,
}

/// `fea.<Quantity>Signal.fromTimeSamples(id, times, values, interpolation, extrapolation)`.
pub fn infer_time_samples(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: StaticClassIdentity,
    shape: SignalShape,
) -> CallInference {
    let arguments = &request.arguments;
    let mut diagnostics = Vec::new();
    if arguments.len() != 5 {
        diagnostics.push(argument_error(
            ARITY,
            format!(
                "{} requires id, times, values, interpolation, and extrapolation",
                entry.identity.name
            ),
            arguments.len().min(5),
        ));
    }
    let times = arguments.get(1).map(|fact| fact.shape.element_count());
    for (index, fact) in arguments.iter().enumerate().take(5) {
        let failure = if resident(fact) {
            Some((
                RESIDENT,
                "signal constructor arguments are host values; gather device data explicitly",
            ))
        } else {
            match index {
                0 => (!host_text(fact)).then_some((TEXT, "signal id must be a text scalar")),
                1 => real_numeric(fact).or_else(|| {
                    (!vector_shape(fact) || matches!(count(fact), Some(count) if count < 2))
                        .then_some((SHAPE, "times must be a vector of at least two samples"))
                }),
                2 => real_numeric(fact).or_else(|| sample_shape(fact, times.flatten(), shape)),
                3 => (!possible_object(fact, SIGNAL_INTERPOLATION)).then_some((
                    TYPED,
                    "interpolation must be a fea.SignalInterpolation policy",
                )),
                _ => (!possible_object(fact, SIGNAL_EXTRAPOLATION)).then_some((
                    TYPED,
                    "extrapolation must be a fea.SignalExtrapolation policy",
                )),
            }
        };
        if let Some((code, message)) = failure {
            diagnostics.push(argument_error(code, message, index));
        }
    }
    finish_fixed(entry, request, policy_object(class), diagnostics)
}

/// `fea.<Quantity>Signal.fromFunction(id, callable, domain, sampling, Name, Value, ...)`.
pub fn infer_function_signal(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    class: StaticClassIdentity,
) -> CallInference {
    let arguments = &request.arguments;
    let mut diagnostics = Vec::new();
    if arguments.len() < 4 || !(arguments.len() - 4).is_multiple_of(2) {
        diagnostics.push(argument_error(
            ARITY,
            format!(
                "{} requires id, callable, domain, sampling, and complete Name/Value pairs",
                entry.identity.name
            ),
            arguments.len(),
        ));
    }
    for (index, fact) in arguments.iter().enumerate().take(4) {
        let failure = if resident(fact) {
            Some((
                RESIDENT,
                "signal constructor arguments are host values; gather device data explicitly",
            ))
        } else {
            match index {
                0 => (!host_text(fact)).then_some((TEXT, "signal id must be a text scalar")),
                1 => (!matches!(
                    fact.kind,
                    ValueKindFact::Unknown | ValueKindFact::Callable(_)
                ))
                .then_some((
                    CALLABLE,
                    "the generator must be a function handle or closure",
                )),
                2 => (!possible_object(fact, SIGNAL_DOMAIN))
                    .then_some((TYPED, "domain must be a fea.SignalDomain value")),
                _ => (!possible_object(fact, SIGNAL_SAMPLING))
                    .then_some((TYPED, "sampling must be a fea.SignalSampling value")),
            }
        };
        if let Some((code, message)) = failure {
            diagnostics.push(argument_error(code, message, index));
        }
    }
    let mut seen = Vec::new();
    for index in (4..arguments.len()).step_by(2) {
        diagnostics.extend(function_option(request, index, &mut seen));
    }
    finish_fixed(entry, request, policy_object(class), diagnostics)
}

fn function_option(
    request: &CallRequest,
    index: usize,
    seen: &mut Vec<String>,
) -> Vec<InferenceDiagnostic> {
    let mut diagnostics = Vec::new();
    let name = &request.arguments[index];
    if resident(name) || !host_text(name) {
        diagnostics.push(argument_error(
            OPTION,
            "fromFunction option names must be text scalars",
            index,
        ));
    }
    let value = request.arguments.get(index + 1);
    if value.is_some_and(resident) {
        diagnostics.push(argument_error(
            RESIDENT,
            "fromFunction option values are host values",
            index + 1,
        ));
    }
    let Some(literal) = request.literals.literal_string_at(index) else {
        return diagnostics;
    };
    let key = literal.to_ascii_lowercase();
    if seen.contains(&key) {
        diagnostics.push(argument_error(
            OPTION,
            format!("duplicate fromFunction option `{literal}`"),
            index,
        ));
    }
    let failure = match (key.as_str(), value) {
        (_, None) => None,
        ("arguments", Some(value)) => {
            (!matches!(value.kind, ValueKindFact::Unknown | ValueKindFact::Cell(_))).then_some((
                ARGUMENTS,
                "Arguments must be a cell array of callable arguments",
            ))
        }
        ("interpolation", Some(value)) => (!possible_object(value, SIGNAL_INTERPOLATION))
            .then_some((
                TYPED,
                "Interpolation must be a fea.SignalInterpolation policy",
            )),
        ("extrapolation", Some(value)) => (!possible_object(value, SIGNAL_EXTRAPOLATION))
            .then_some((
                TYPED,
                "Extrapolation must be a fea.SignalExtrapolation policy",
            )),
        _ => {
            diagnostics.push(argument_error(
                OPTION,
                format!("unsupported fromFunction option `{literal}`; expected Arguments, Interpolation, or Extrapolation"),
                index,
            ));
            None
        }
    };
    if let Some((code, message)) = failure {
        diagnostics.push(argument_error(code, message, index + 1));
    }
    seen.push(key);
    diagnostics
}

fn resident(fact: &ValueFact) -> bool {
    matches!(fact.residency, ResidencyFact::Device { .. })
}

fn count(fact: &ValueFact) -> Option<usize> {
    fact.shape.element_count()
}

fn real_numeric(fact: &ValueFact) -> Option<(&'static str, &'static str)> {
    let real = match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Numeric(numeric) => numeric.domain == NumericDomain::Real,
        _ => false,
    };
    (!real).then_some((NUMERIC, "expected a real numeric array"))
}

/// False only when the shape is proven to have two non-singleton extents.
fn vector_shape(fact: &ValueFact) -> bool {
    known_extents(fact).filter(|extent| *extent != 1).count() <= 1
}

fn known_extents(fact: &ValueFact) -> impl Iterator<Item = usize> + '_ {
    let dims = match &fact.shape {
        ShapeFact::Shaped { dims } => dims.as_slice(),
        _ => &[],
    };
    dims.iter().filter_map(|dim| match dim {
        DimensionFact::Known(extent) => Some(*extent),
        _ => None,
    })
}

fn sample_shape(
    fact: &ValueFact,
    times: Option<usize>,
    shape: SignalShape,
) -> Option<(&'static str, &'static str)> {
    let samples = count(fact);
    let mismatch = match shape {
        SignalShape::Scalar => {
            !vector_shape(fact)
                || matches!((samples, times), (Some(samples), Some(times)) if samples != times)
        }
        SignalShape::Vector3 => {
            let rank_two = matches!(&fact.shape, ShapeFact::Shaped { dims } if dims.len() == 2);
            let extents: Vec<_> = known_extents(fact).collect();
            let no_component_axis = rank_two && extents.len() == 2 && !extents.contains(&3);
            no_component_axis
                || matches!((samples, times), (Some(samples), Some(times)) if samples != 3 * times)
        }
    };
    mismatch.then_some((
        SHAPE,
        "values must hold one sample per time in the signal's shape",
    ))
}

fn possible_object(fact: &ValueFact, class: StaticClassIdentity) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown => true,
        ValueKindFact::Object(object) => {
            count(fact).is_none_or(|count| count == 1)
                && object
                    .runtime_class
                    .as_ref()
                    .is_none_or(|actual| actual.is(class))
        }
        _ => false,
    }
}
