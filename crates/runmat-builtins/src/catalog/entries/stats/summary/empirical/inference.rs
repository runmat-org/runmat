//! Sound public output facts for empirical statistics and their graphics sink.
use super::options::{EcdfBounds, EcdfFunction, EcdfOptionName, EmpiricalBuiltin};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, EffectKind,
    InferenceDiagnostic, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ResidencyFact, ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn error(code: &'static str, message: &'static str) -> InferenceDiagnostic {
    InferenceDiagnostic::error(code, message)
}

fn text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn double(shape: ShapeFact, residency: ResidencyFact) -> ValueFact {
    let storage = match shape.element_count() {
        Some(1) => StorageFact::Scalar,
        Some(_) => StorageFact::Dense,
        None => StorageFact::Unknown,
    };
    let mut fact = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    );
    fact.residency = residency;
    fact
}

fn unknown() -> ValueFact {
    ValueFact::unknown(DynamicReason::RuntimeValue)
}

fn cannot_be_axes_handle(value: &ValueFact) -> bool {
    value.shape.element_count().is_some_and(|count| count != 1)
}

fn known_bad_data(
    builtin: EmpiricalBuiltin,
    value: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> bool {
    let bad_kind = !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    );
    if bad_kind {
        diagnostics.push(error(
            "INPUT",
            "sample data must be real numeric or logical",
        ));
    }
    let bad_shape = match &value.shape {
        ShapeFact::Shaped { dims } if dims.len() >= 2 => matches!((&dims[0], &dims[1]), (runmat_types::DimensionFact::Known(rows), runmat_types::DimensionFact::Known(cols)) if *rows > 1 && *cols > 1)
            || builtin == EmpiricalBuiltin::Cdfplot
                && dims.iter().skip(2).any(
                    |dim| matches!(dim, runmat_types::DimensionFact::Known(extent) if *extent > 1),
                ),
        _ => false,
    };
    if bad_shape {
        diagnostics.push(error("SHAPE", "sample data must be a vector"));
    }
    if value.shape.element_count() == Some(0) {
        diagnostics.push(error("EMPTY", "sample data must be nonempty"));
    }
    bad_kind || bad_shape || value.shape.element_count() == Some(0)
}

fn validate_ecdf_options(
    request: &CallRequest,
    sample_index: usize,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let first_option = sample_index + 1;
    if request.arguments.len() <= first_option {
        return;
    }
    if !(request.arguments.len() - first_option).is_multiple_of(2) {
        diagnostics.push(error("OPTION", "ecdf options require name-value pairs"));
        return;
    }
    let mut censoring_count = None;
    let mut frequency_count = None;
    for index in (first_option..request.arguments.len()).step_by(2) {
        let Some(name) = request.literals.literal_args.get(index) else {
            continue;
        };
        let Some(name) = text(name) else {
            if !matches!(name, LiteralValue::Unknown)
                || request.arguments.get(index).is_some_and(|fact| {
                    matches!(
                        fact.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Logical
                    )
                })
            {
                diagnostics.push(error("OPTION", "ecdf option names must be text"));
            }
            continue;
        };
        let Some(option) = EcdfOptionName::from_public(name) else {
            diagnostics.push(error("OPTION", "unsupported ecdf option"));
            continue;
        };
        if option.is_interval_only() {
            diagnostics.push(error(
                "UNSUPPORTED",
                "interval-censored controls are unsupported",
            ));
            continue;
        }
        let Some(value) = request.literals.literal_args.get(index + 1) else {
            continue;
        };
        match option {
            EcdfOptionName::Function => {
                if let Some(text) = text(value) {
                    if EcdfFunction::from_public(text).is_none() {
                        diagnostics.push(error("FUNCTION", "unsupported ecdf Function value"));
                    }
                } else if request.arguments.get(index + 1).is_some_and(|fact| {
                    matches!(
                        fact.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Logical
                    )
                }) || !matches!(value, LiteralValue::Unknown)
                    && !matches!(
                        request.arguments.get(index + 1).map(|fact| &fact.kind),
                        Some(ValueKindFact::String | ValueKindFact::Character)
                    )
                {
                    diagnostics.push(error("FUNCTION", "ecdf Function must be text"));
                }
            }
            EcdfOptionName::Bounds => {
                if let Some(text) = text(value) {
                    if EcdfBounds::from_public(text).is_none() {
                        diagnostics.push(error("BOUNDS", "unsupported ecdf Bounds value"));
                    }
                } else if request.arguments.get(index + 1).is_some_and(|fact| {
                    matches!(
                        fact.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Logical
                    )
                }) || !matches!(value, LiteralValue::Unknown)
                    && !matches!(
                        request.arguments.get(index + 1).map(|fact| &fact.kind),
                        Some(ValueKindFact::String | ValueKindFact::Character)
                    )
                {
                    diagnostics.push(error("BOUNDS", "ecdf Bounds must be text"));
                }
            }
            EcdfOptionName::Alpha => {
                if let Some(number) = LiteralContext::numeric_from_literal(value) {
                    if !(number.is_finite() && number > 0.0 && number < 1.0) {
                        diagnostics.push(error("ALPHA", "ecdf Alpha must be in (0,1)"));
                    }
                } else if request.arguments.get(index + 1).is_some_and(|fact| {
                    !matches!(
                        fact.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Unknown
                    ) || matches!(
                        fact.kind,
                        ValueKindFact::Numeric(NumericFact {
                            domain: NumericDomain::Complex,
                            ..
                        })
                    ) || fact.shape.element_count().is_some_and(|n| n != 1)
                }) {
                    diagnostics.push(error("ALPHA", "ecdf Alpha must be numeric"));
                }
            }
            EcdfOptionName::Censoring | EcdfOptionName::Frequency => {
                let count = request
                    .arguments
                    .get(index + 1)
                    .and_then(|fact| fact.shape.element_count());
                match option {
                    EcdfOptionName::Censoring => censoring_count = count,
                    EcdfOptionName::Frequency => frequency_count = count,
                    _ => unreachable!(),
                }
                if request.arguments.get(index + 1).is_some_and(|fact| {
                    !matches!(
                        fact.kind,
                        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown
                    )
                        || matches!(fact.kind, ValueKindFact::Numeric(NumericFact { domain: NumericDomain::Complex, .. }))
                        || matches!(&fact.shape, ShapeFact::Shaped { dims } if dims.len() >= 2 && matches!((&dims[0], &dims[1]), (runmat_types::DimensionFact::Known(rows), runmat_types::DimensionFact::Known(cols)) if *rows > 1 && *cols > 1))
                }) {
                    diagnostics.push(error("OPTION", "ecdf observation controls must be numeric"));
                }
            }
            _ => unreachable!("interval-only options handled above"),
        }
    }
    if let Some(sample_count) = request
        .arguments
        .get(sample_index)
        .and_then(|fact| fact.shape.element_count())
    {
        if censoring_count.is_some_and(|count| count != sample_count) {
            diagnostics.push(error(
                "CENSORING",
                "Censoring must match the sample element count",
            ));
        }
        if frequency_count.is_some_and(|count| count != sample_count) {
            diagnostics.push(error(
                "FREQUENCY",
                "Frequency must match the sample element count",
            ));
        }
    }
}

pub(super) fn ecdf(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let even_form = request.arguments.len() >= 2 && request.arguments.len().is_multiple_of(2);
    let ambiguous_axes = even_form
        && request
            .arguments
            .first()
            .is_some_and(|fact| !cannot_be_axes_handle(fact));
    let sample_index = usize::from(ambiguous_axes);
    if request.arguments.is_empty() {
        diagnostics.push(error("ARITY", "ecdf requires sample data"));
    } else if even_form && !ambiguous_axes {
        diagnostics.push(error("OPTION", "ecdf options require name-value pairs"));
    }
    if let Some(value) = request.arguments.get(sample_index) {
        known_bad_data(EmpiricalBuiltin::Ecdf, value, &mut diagnostics);
    }
    validate_ecdf_options(request, sample_index, &mut diagnostics);
    let bad = !diagnostics.is_empty() || ambiguous_axes;
    let residency = match request
        .arguments
        .get(sample_index)
        .map(|value| &value.residency)
    {
        Some(ResidencyFact::Host) => ResidencyFact::Host,
        _ => ResidencyFact::Unknown,
    };
    let values = (0..4)
        .map(|_| {
            if bad {
                unknown()
            } else {
                double(ShapeFact::from(vec![None, Some(1)]), residency.clone())
            }
        })
        .collect();
    let mut contract = CallContract::fixed(values);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(double(ShapeFact::Scalar, ResidencyFact::Host)));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    if request
        .outputs
        .requested
        .known_count()
        .is_some_and(|count| count > 0)
    {
        contract.effects.0.remove(&EffectKind::UserInterface);
        contract.effects.0.remove(&EffectKind::HostCallback);
        contract
            .capabilities
            .0
            .remove(&runmat_types::CapabilityRequirement::UserInterface);
    }
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn cdfplot(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if !(1..=2).contains(&request.arguments.len()) {
        diagnostics.push(error("ARITY", "cdfplot requires data and optional axes"));
    }
    let ambiguous_axes = request.arguments.len() == 2
        && request
            .arguments
            .first()
            .is_some_and(|fact| !cannot_be_axes_handle(fact));
    if request.arguments.len() == 2 && !ambiguous_axes {
        diagnostics.push(error(
            "AXES",
            "cdfplot requires a scalar axes handle before data",
        ));
    }
    let sample = request.arguments.last();
    if let Some(sample) = sample {
        known_bad_data(EmpiricalBuiltin::Cdfplot, sample, &mut diagnostics);
    }
    let bad = !diagnostics.is_empty() || ambiguous_axes;
    let scalar = if bad {
        unknown()
    } else {
        double(ShapeFact::Scalar, ResidencyFact::Host)
    };
    let stats = if bad {
        unknown()
    } else {
        let fields = ["min", "max", "mean", "median", "std"]
            .into_iter()
            .map(|name| {
                (
                    name.to_owned(),
                    double(ShapeFact::Scalar, ResidencyFact::Host),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut fact = ValueFact::proven(
            ValueKindFact::Struct(StructFact::scalar(fields, true)),
            ShapeFact::Scalar,
            StorageFact::Scalar,
        );
        fact.residency = ResidencyFact::Host;
        fact
    };
    let mut contract = CallContract::fixed(vec![scalar, stats]);
    contract.maximum_outputs = None;
    contract.variadic_output = Some(Box::new(double(ShapeFact::Scalar, ResidencyFact::Host)));
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
