use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, LiteralValue,
    NumericClass, NumericDomain, NumericFact, ResidencyFact, ShapeFact, StorageFact, ValueFact,
    ValueKindFact,
};

/// Sampling of a generated cosine-sum window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowSampling {
    Symmetric,
    Periodic,
}

/// Class of the generated window, independent of the length's class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowOutputType {
    Double,
    Single,
}

/// The two public option roles. Decode text once at the language boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowOption {
    Sampling(WindowSampling),
    OutputType(WindowOutputType),
}

impl WindowOption {
    pub fn parse(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("symmetric") {
            Some(Self::Sampling(WindowSampling::Symmetric))
        } else if text.eq_ignore_ascii_case("periodic") {
            Some(Self::Sampling(WindowSampling::Periodic))
        } else if text.eq_ignore_ascii_case("double") {
            Some(Self::OutputType(WindowOutputType::Double))
        } else if text.eq_ignore_ascii_case("single") {
            Some(Self::OutputType(WindowOutputType::Single))
        } else {
            None
        }
    }
}

/// Shared static contract for the three cosine-sum window generators.
pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let name = entry.identity.name;
    if !(1..=3).contains(&request.arguments.len()) {
        diagnostics.push(argument_error(
            "RM-CATALOG-WINDOW-ARITY",
            format!("{name} expects a length and at most two options"),
            request.arguments.len().saturating_sub(1),
        ));
    }

    let mut length = None;
    if let Some(input) = request.arguments.first() {
        let admitted = matches!(
            input.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        );
        if !admitted || input.shape.element_count().is_some_and(|count| count != 1) {
            diagnostics.push(argument_error(
                "RM-CATALOG-WINDOW-LENGTH",
                format!("{name} length must be a nonnegative real scalar"),
                0,
            ));
        }
        match request.literals.literal_args.first() {
            Some(LiteralValue::Bool(value)) => length = Some(usize::from(*value)),
            Some(LiteralValue::Integer { text, .. }) => {
                length = text.parse::<usize>().ok();
                if length.is_none() {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-WINDOW-LENGTH",
                        "window length is outside the supported size range",
                        0,
                    ));
                }
            }
            Some(literal) => {
                if let Some(number) = runmat_types::LiteralContext::numeric_from_literal(literal) {
                    let rounded = number.round();
                    if !number.is_finite()
                        || number < 0.0
                        || rounded > usize::MAX as f64
                        || (usize::BITS == 64 && rounded == usize::MAX as f64)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-WINDOW-LENGTH",
                            "window length must be finite, nonnegative and representable",
                            0,
                        ));
                    } else {
                        length = Some(rounded as usize);
                    }
                }
            }
            None => {}
        }
    }

    let mut output_type = Some(WindowOutputType::Double);
    for index in 1..request.arguments.len().min(3) {
        let value = &request.arguments[index];
        let bad_geometry = match value.kind {
            ValueKindFact::Character => match &value.shape {
                ShapeFact::Shaped { dims } => dims.first().is_some_and(
                    |dim| matches!(dim, runmat_types::DimensionFact::Known(rows) if *rows != 1),
                ),
                _ => false,
            },
            ValueKindFact::String => value.shape.element_count().is_some_and(|count| count != 1),
            ValueKindFact::Unknown => false,
            _ => true,
        };
        if bad_geometry {
            diagnostics.push(argument_error(
                "RM-CATALOG-WINDOW-OPTION",
                "window options must be scalar text",
                index,
            ));
            continue;
        }
        match request.literals.literal_string_at(index) {
            Some(text) => match WindowOption::parse(&text) {
                Some(WindowOption::Sampling(_)) if index == 1 => {}
                Some(WindowOption::OutputType(kind))
                    if index == 1 && request.arguments.len() == 2 =>
                {
                    output_type = Some(kind)
                }
                Some(WindowOption::OutputType(kind)) if index == 2 => output_type = Some(kind),
                _ => diagnostics.push(argument_error(
                    "RM-CATALOG-WINDOW-OPTION",
                    "window option is in the wrong position",
                    index,
                )),
            },
            None => output_type = None,
        }
    }
    let output = if diagnostics.is_empty() {
        output_type.map_or_else(
            || ValueFact::unknown(DynamicReason::RuntimeValue),
            |kind| {
                let mut output = ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: match kind {
                            WindowOutputType::Double => NumericClass::Double,
                            WindowOutputType::Single => NumericClass::Single,
                        },
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::from(vec![length, Some(1)]),
                    StorageFact::Dense,
                );
                output.residency = ResidencyFact::Unknown;
                output
            },
        )
    } else {
        ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inferred = infer_call(&contract, request);
    inferred.diagnostics.extend(diagnostics);
    inferred
}
