//! Conservative ttest2 output facts; runtime value-dependent tests remain dynamic.
use super::options::{Ttest2OptionName, Ttest2Tail, Ttest2Variance};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, DynamicReason, InferenceDiagnostic,
    LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact, StorageFact,
    StructFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn text(value: &LiteralValue) -> Option<&str> {
    match value {
        LiteralValue::Character(text)
        | LiteralValue::String(text)
        | LiteralValue::Keyword(text) => Some(text),
        _ => None,
    }
}

fn known_nontext(value: Option<&ValueFact>) -> bool {
    value.is_some_and(|value| {
        matches!(
            &value.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    })
}

fn known_nonnumeric(value: Option<&ValueFact>) -> bool {
    value.is_some_and(|value| {
        matches!(
            &value.kind,
            ValueKindFact::Character
                | ValueKindFact::String
                | ValueKindFact::Cell(_)
                | ValueKindFact::Struct(_)
                | ValueKindFact::Object(_)
        )
    })
}

fn double(shape: ShapeFact) -> ValueFact {
    let storage = if shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else if shape.element_count().is_some() {
        StorageFact::Dense
    } else {
        StorageFact::Unknown
    };
    ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        storage,
    )
}

fn selected_dim(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<Option<usize>> {
    if !(request.arguments.len() - 2).is_multiple_of(2) {
        diagnostics.push(InferenceDiagnostic::error(
            "OPTION",
            "ttest2 options require name-value pairs",
        ));
        return None;
    }
    let mut selected = Some(None);
    for index in (2..request.arguments.len()).step_by(2) {
        let name = request.literals.literal_args.get(index)?;
        let Some(name) = text(name) else {
            if !matches!(name, LiteralValue::Unknown) || known_nontext(request.arguments.get(index))
            {
                diagnostics.push(InferenceDiagnostic::error(
                    "OPTION",
                    "ttest2 option name must be text",
                ));
            }
            return None;
        };
        let Some(option) = Ttest2OptionName::from_public(name) else {
            diagnostics.push(InferenceDiagnostic::error(
                "OPTION",
                "unsupported ttest2 option",
            ));
            return None;
        };
        let value = request.literals.literal_args.get(index + 1);
        match option {
            Ttest2OptionName::Dim => {
                let value = value?;
                let Some(number) = LiteralContext::numeric_from_literal(value) else {
                    if !matches!(value, LiteralValue::Unknown)
                        || known_nonnumeric(request.arguments.get(index + 1))
                    {
                        diagnostics.push(InferenceDiagnostic::error(
                            "DIMENSION",
                            "ttest2 Dim must be numeric",
                        ));
                    }
                    return None;
                };
                let Ok(dim) = runmat_types::decode_shape_dimension(number) else {
                    diagnostics.push(InferenceDiagnostic::error(
                        "DIMENSION",
                        "ttest2 Dim must be a positive integer",
                    ));
                    return None;
                };
                if dim == 0 {
                    diagnostics.push(InferenceDiagnostic::error(
                        "DIMENSION",
                        "ttest2 Dim must be a positive integer",
                    ));
                    return None;
                }
                selected = Some(Some(dim - 1));
            }
            Ttest2OptionName::Alpha => {
                if let Some(value) = value {
                    if let Some(number) = LiteralContext::numeric_from_literal(value) {
                        if !(number.is_finite() && number > 0.0 && number < 1.0) {
                            diagnostics.push(InferenceDiagnostic::error(
                                "ALPHA",
                                "ttest2 Alpha must be in (0,1)",
                            ));
                        }
                    } else if !matches!(value, LiteralValue::Unknown)
                        || known_nonnumeric(request.arguments.get(index + 1))
                    {
                        diagnostics.push(InferenceDiagnostic::error(
                            "ALPHA",
                            "ttest2 Alpha must be numeric",
                        ));
                    }
                }
            }
            Ttest2OptionName::Tail | Ttest2OptionName::VarType => {
                let value = value?;
                if let Some(value) = text(value) {
                    let valid = match option {
                        Ttest2OptionName::Tail => Ttest2Tail::from_public(value).is_some(),
                        Ttest2OptionName::VarType => Ttest2Variance::from_public(value).is_some(),
                        _ => unreachable!(),
                    };
                    if !valid {
                        diagnostics.push(InferenceDiagnostic::error(
                            "OPTION",
                            "unsupported ttest2 option value",
                        ));
                    }
                } else if !matches!(value, LiteralValue::Unknown)
                    || known_nontext(request.arguments.get(index + 1))
                {
                    diagnostics.push(InferenceDiagnostic::error(
                        "OPTION",
                        "ttest2 option value must be text",
                    ));
                }
            }
        }
    }
    selected
}

fn output_shapes(
    request: &CallRequest,
    dim: Option<Option<usize>>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> (ShapeFact, ShapeFact) {
    let (Some(x), Some(y), Some(dim)) = (request.arguments.first(), request.arguments.get(1), dim)
    else {
        return (ShapeFact::Unknown, ShapeFact::Unknown);
    };
    let (Some(mut xs), Some(mut ys)) = (x.shape.known_dims(), y.shape.known_dims()) else {
        return (ShapeFact::Unknown, ShapeFact::Unknown);
    };
    if xs.iter().any(Option::is_none) || ys.iter().any(Option::is_none) {
        return (ShapeFact::Unknown, ShapeFact::Unknown);
    }
    let vector = |dims: &[Option<usize>]| {
        dims.iter().all(|v| v.unwrap_or(0) > 0)
            && dims.iter().filter(|v| v.unwrap_or(0) > 1).count() <= 1
    };
    if dim.is_none() && vector(&xs) && vector(&ys) {
        return (
            ShapeFact::from(vec![Some(1), Some(1)]),
            ShapeFact::from(vec![Some(2), Some(1)]),
        );
    }
    let dim = dim.unwrap_or_else(|| xs.iter().position(|v| *v != Some(1)).unwrap_or(0));
    let rank = xs.len().max(ys.len()).max(2);
    // A source-controlled literal cannot grow compiler metadata without
    // bound. Runtime owns fallible padding beyond the known input rank.
    if dim >= rank {
        return (ShapeFact::Unknown, ShapeFact::Unknown);
    }
    xs.resize(rank, Some(1));
    ys.resize(rank, Some(1));
    if xs
        .iter()
        .zip(&ys)
        .enumerate()
        .any(|(axis, (x, y))| axis != dim && x != y)
    {
        diagnostics.push(InferenceDiagnostic::error(
            "SHAPE",
            "ttest2 sample dimensions must agree outside the test dimension",
        ));
        return (ShapeFact::Unknown, ShapeFact::Unknown);
    }
    xs[dim] = Some(1);
    let mut interval = vec![Some(2)];
    interval.extend(
        xs.iter()
            .enumerate()
            .filter_map(|(axis, value)| (axis != dim).then_some(*value)),
    );
    if interval.len() == 1 {
        interval.push(Some(1));
    }
    (ShapeFact::from(xs), ShapeFact::from(interval))
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(InferenceDiagnostic::error(
            "ARITY",
            "ttest2 requires two samples",
        ));
    }
    for sample in request.arguments.iter().take(2) {
        if !matches!(
            &sample.kind,
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            }) | ValueKindFact::Logical
                | ValueKindFact::Unknown
        ) {
            diagnostics.push(InferenceDiagnostic::error(
                "INPUT",
                "ttest2 samples must be real numeric or logical arrays",
            ));
        }
    }
    let dim = if request.arguments.len() >= 2 {
        selected_dim(request, &mut diagnostics)
    } else {
        None
    };
    let (result_shape, ci_shape) = output_shapes(request, dim, &mut diagnostics);
    let mut h = ValueFact::proven(
        ValueKindFact::Logical,
        result_shape.clone(),
        StorageFact::Unknown,
    );
    h.storage = if result_shape.element_count() == Some(1) {
        StorageFact::Scalar
    } else if result_shape.element_count().is_some() {
        StorageFact::Dense
    } else {
        StorageFact::Unknown
    };
    let p = double(result_shape.clone());
    let ci = double(ci_shape.clone());
    let mut fields = BTreeMap::new();
    fields.insert("tstat".into(), p.clone());
    fields.insert("df".into(), p.clone());
    // Welch's sd has an extra interval axis; without a known Vartype use a
    // dynamic field rather than claiming equal-variance geometry.
    fields.insert("sd".into(), ValueFact::unknown(DynamicReason::RuntimeValue));
    let stats = ValueFact::scalar(ValueKindFact::Struct(StructFact::scalar(fields, true)));
    let outputs = if diagnostics.is_empty() {
        vec![h, p, ci, stats]
    } else {
        vec![ValueFact::unknown(DynamicReason::RuntimeValue); 4]
    };
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}
