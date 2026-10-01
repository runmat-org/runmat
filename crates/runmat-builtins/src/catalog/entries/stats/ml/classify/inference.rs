//! Output facts preserve the label family; learned class count remains dynamic.
use super::{DiscriminantType, PriorKeyword};
use crate::BuiltinCatalogEntry;
use runmat_types::{
    infer_call, CallContract, CallInference, CallRequest, CellFact, DimensionFact, DynamicReason,
    InferenceDiagnostic, LiteralValue, NumericClass, NumericDomain, NumericFact, ShapeFact,
    StorageFact, ValueFact, ValueKindFact,
};

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    if request.outputs.requested.known_count() == Some(0) {
        let mut contract = CallContract::fixed(Vec::new());
        contract.effects = entry.contract.effect_set();
        contract.capabilities = entry.contract.capability_set();
        let mut result = infer_call(&contract, request);
        if request.arguments.len() < 3 || request.arguments.len() > 5 {
            result.diagnostics.push(InferenceDiagnostic::error(
                "RM.CLASSIFY.ARITY",
                "classify requires sample, training, group, and at most type and prior",
            ));
        }
        return result;
    }
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 || request.arguments.len() > 5 {
        diagnostics.push(InferenceDiagnostic::error(
            "RM.CLASSIFY.ARITY",
            "classify requires sample, training, group, and at most type and prior",
        ));
    }
    for index in [0, 1] {
        if let Some(fact) = request.arguments.get(index) {
            if !matches!(
                fact.kind,
                ValueKindFact::Numeric(NumericFact {
                    domain: NumericDomain::Real,
                    ..
                }) | ValueKindFact::Logical
                    | ValueKindFact::Unknown
            ) {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM.CLASSIFY.INPUT",
                    "predictors must be real numeric or logical",
                ));
            }
            if matches!(&fact.shape, ShapeFact::Shaped { dims } if dims.len() > 2) {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM.CLASSIFY.RANK",
                    "predictors must be 2-D matrices",
                ));
            }
        }
    }
    if let (Some(sample), Some(training)) = (request.arguments.first(), request.arguments.get(1)) {
        if let (Some(left), Some(right)) = (columns(&sample.shape), columns(&training.shape)) {
            if left != right {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM.CLASSIFY.COLUMNS",
                    "sample and training must have matching columns",
                ));
            }
        }
        if columns(&sample.shape) == Some(0) || rows(&training.shape) == Some(0) {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.CLASSIFY.EMPTY",
                "sample and training must be nonempty numeric matrices",
            ));
        }
    }
    if let (Some(training), Some(group)) = (request.arguments.get(1), request.arguments.get(2)) {
        let group_count = if matches!(group.kind, ValueKindFact::Character) {
            rows(&group.shape)
        } else {
            group.shape.element_count()
        };
        if let (Some(expected), Some(actual)) = (rows(&training.shape), group_count) {
            if expected != actual {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM.CLASSIFY.GROUP",
                    "group length must equal training rows",
                ));
            }
        }
        if matches!(&group.shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(value) if *value > 1)).count() > 1)
            && matches!(group.kind, ValueKindFact::Numeric(_))
        {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.CLASSIFY.GROUP",
                "group must be a label vector",
            ));
        }
    }
    if let Some(literal) = request.literals.literal_args.get(3) {
        match literal {
            LiteralValue::String(text)
            | LiteralValue::Character(text)
            | LiteralValue::Keyword(text)
                if DiscriminantType::from_public(text).is_none() =>
            {
                diagnostics.push(InferenceDiagnostic::error(
                    "RM.CLASSIFY.TYPE",
                    "unsupported discriminant type",
                ))
            }
            LiteralValue::Number(_)
            | LiteralValue::Real { .. }
            | LiteralValue::Integer { .. }
            | LiteralValue::Bool(_) => diagnostics.push(InferenceDiagnostic::error(
                "RM.CLASSIFY.TYPE",
                "discriminant type must be text",
            )),
            _ => {}
        }
    }
    if let Some(kind) = request.arguments.get(3).map(|fact| &fact.kind) {
        if !matches!(
            kind,
            ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
        ) {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.CLASSIFY.TYPE",
                "discriminant type must be text",
            ));
        }
    }
    if let Some(prior) = request.arguments.get(4) {
        diagnose_prior(
            prior,
            request.literals.literal_args.get(4),
            &mut diagnostics,
        );
    }
    let sample_rows = request.arguments.first().and_then(|fact| rows(&fact.shape));
    let label = request.arguments.get(2).map_or_else(
        || ValueFact::unknown(DynamicReason::RuntimeValue),
        |fact| label_output(fact, sample_rows, &mut diagnostics),
    );
    let numerical = |shape: ShapeFact| {
        ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            shape,
            StorageFact::Dense,
        )
    };
    let mut contract = CallContract::fixed(vec![
        label,
        ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })),
        numerical(ShapeFact::from(vec![sample_rows, None])),
        numerical(ShapeFact::from(vec![sample_rows, Some(1)])),
        ValueFact::proven(
            cell_kind(),
            ShapeFact::from(vec![None, None]),
            StorageFact::Unknown,
        ),
    ]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    if !diagnostics.is_empty() {
        for output in &mut result.outputs {
            *output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    }
    result.diagnostics.extend(diagnostics);
    result
}

fn diagnose_prior(
    prior: &ValueFact,
    literal: Option<&LiteralValue>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    let invalid = match &prior.kind {
        ValueKindFact::Unknown => false,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) => {
            prior.shape.element_count() == Some(1)
                || matches!(&prior.shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(value) if *value > 1)).count() > 1)
                || literal.is_some_and(known_invalid_probability)
        }
        ValueKindFact::String | ValueKindFact::Character => match literal {
            Some(
                LiteralValue::String(text)
                | LiteralValue::Character(text)
                | LiteralValue::Keyword(text),
            ) => PriorKeyword::from_public(text).is_none(),
            _ => false,
        },
        ValueKindFact::Struct(structure) => {
            let groups = structure
                .fields
                .get("group")
                .or_else(|| structure.fields.get("Group"));
            let probabilities = structure
                .fields
                .get("prob")
                .or_else(|| structure.fields.get("Prob"));
            (structure.fields_complete && (groups.is_none() || probabilities.is_none()))
                || groups.is_some_and(|value| definitely_invalid_prior_groups(&value.kind))
                || probabilities.is_some_and(|value| definitely_invalid_prior_values(value))
        }
        _ => true,
    };
    if invalid {
        diagnostics.push(InferenceDiagnostic::error(
            "RM.CLASSIFY.PRIOR",
            "prior must be empirical text, a numeric vector, or a group/prob struct",
        ));
    }
}

fn definitely_invalid_prior_groups(kind: &ValueKindFact) -> bool {
    !matches!(
        kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Logical
            | ValueKindFact::String
            | ValueKindFact::Character
            | ValueKindFact::Cell(_)
    )
}

fn definitely_invalid_prior_values(fact: &ValueFact) -> bool {
    !matches!(
        fact.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    ) || matches!(&fact.shape, ShapeFact::Shaped { dims } if dims.iter().filter(|dim| matches!(dim, DimensionFact::Known(value) if *value > 1)).count() > 1)
}

fn known_invalid_probability(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Number(value) => !value.is_finite() || *value <= 0.0,
        LiteralValue::Real { text, .. } | LiteralValue::Integer { text, .. } => text
            .parse::<f64>()
            .is_ok_and(|value| !value.is_finite() || value <= 0.0),
        _ => false,
    }
}

fn rows(shape: &ShapeFact) -> Option<usize> {
    match shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } => dims.first().and_then(known_dimension),
        _ => None,
    }
}

fn columns(shape: &ShapeFact) -> Option<usize> {
    match shape {
        ShapeFact::Scalar => Some(1),
        ShapeFact::Shaped { dims } => dims
            .get(1)
            .and_then(known_dimension)
            .or_else(|| (dims.len() == 1).then_some(1)),
        _ => None,
    }
}

fn known_dimension(dimension: &DimensionFact) -> Option<usize> {
    match dimension {
        DimensionFact::Known(value) => Some(*value),
        DimensionFact::Unknown | DimensionFact::Symbolic(_) => None,
    }
}

fn label_output(
    group: &ValueFact,
    sample_rows: Option<usize>,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let kind = match &group.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            class,
        }) => ValueKindFact::Numeric(NumericFact {
            class: *class,
            domain: NumericDomain::Real,
        }),
        ValueKindFact::Logical => ValueKindFact::Logical,
        ValueKindFact::String => ValueKindFact::String,
        ValueKindFact::Character => ValueKindFact::Character,
        ValueKindFact::Cell(_) => cell_kind(),
        ValueKindFact::Unknown => ValueKindFact::Unknown,
        _ => {
            diagnostics.push(InferenceDiagnostic::error(
                "RM.CLASSIFY.GROUP",
                "group labels must be numeric, logical, or text",
            ));
            ValueKindFact::Unknown
        }
    };
    let width = if matches!(kind, ValueKindFact::Character) {
        None
    } else {
        Some(1)
    };
    let shape = ShapeFact::from(vec![sample_rows, width]);
    if matches!(kind, ValueKindFact::Unknown) {
        let mut value = ValueFact::unknown(DynamicReason::RuntimeValue);
        value.shape = shape;
        value
    } else {
        let storage = if matches!(kind, ValueKindFact::Cell(_)) {
            StorageFact::Unknown
        } else {
            StorageFact::Dense
        };
        ValueFact::proven(kind, shape, storage)
    }
}

fn cell_kind() -> ValueKindFact {
    ValueKindFact::Cell(CellFact {
        element: Box::new(ValueFact::unknown(DynamicReason::RuntimeValue)),
        elements: Vec::new(),
        elements_complete: false,
    })
}
