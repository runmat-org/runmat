use super::super::{
    parse_stratify_boolean_alias, parse_stratify_numeric_boolean, CvPartitionMethod,
    CvPartitionOption, MaskSelectionKeyword, PartitionCountGrammar,
};
use crate::{catalog::inference::argument_error, BuiltinCatalogEntry};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, DynamicReason, EffectKind,
    InferenceDiagnostic, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
    ObjectFact, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

fn literal_positive(value: &LiteralValue) -> Option<Result<usize, ()>> {
    match value {
        LiteralValue::Unknown => None,
        LiteralValue::Integer { text, .. } => {
            Some(text.parse::<usize>().ok().filter(|n| *n > 0).ok_or(()))
        }
        LiteralValue::Number(_) | LiteralValue::Real { .. } => Some(
            LiteralContext::numeric_from_literal(value)
                .and_then(|n| PartitionCountGrammar::CvPartition.positive_from_float(n))
                .ok_or(()),
        ),
        _ => Some(Err(())),
    }
}

fn source_text(request: &CallRequest, index: usize) -> Option<String> {
    request.literals.literal_string_at(index)
}

fn holdout_literal_valid(literal: &LiteralValue, n: Option<usize>) -> Option<bool> {
    match literal {
        LiteralValue::Unknown => None,
        LiteralValue::Integer { text, .. } => Some(
            text.parse::<usize>()
                .ok()
                .is_some_and(|count| count > 0 && n.is_none_or(|n| count < n)),
        ),
        LiteralValue::Bool(value) => Some(*value && n.is_none_or(|n| n >= 2)),
        LiteralValue::Number(_) | LiteralValue::Real { .. } => Some(
            LiteralContext::numeric_from_literal(literal).is_some_and(|value| {
                (value > 0.0 && value < 1.0 && n.is_none_or(|n| n >= 2))
                    || (value.is_finite()
                        && PartitionCountGrammar::CvPartition
                            .positive_from_float(value)
                            .is_some_and(|count| n.is_none_or(|n| count < n)))
            }),
        ),
        _ => Some(false),
    }
}

fn custom_payload_possible(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown | ValueKindFact::Logical => true,
        ValueKindFact::Numeric(numeric) => {
            numeric.domain == NumericDomain::Real && !matches!(fact.storage, StorageFact::Scalar)
        }
        _ => false,
    }
}

fn observation_input_possible(fact: &ValueFact) -> bool {
    match &fact.kind {
        ValueKindFact::Unknown | ValueKindFact::Character | ValueKindFact::String => true,
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => fact
            .shape
            .known_dims()
            .is_none_or(|dims| dims.iter().flatten().filter(|n| **n > 1).count() <= 1),
        ValueKindFact::Logical => fact
            .shape
            .known_dims()
            .is_none_or(|dims| dims.iter().flatten().filter(|n| **n > 1).count() <= 1),
        _ => false,
    }
}

fn stratify_literal_valid(request: &CallRequest, index: usize) -> Option<bool> {
    let fact = request.arguments.get(index)?;
    let scalar = fact.shape.element_count().is_none_or(|count| count == 1);
    match &fact.kind {
        ValueKindFact::Unknown => None,
        ValueKindFact::Logical => Some(scalar),
        ValueKindFact::Numeric(numeric) if numeric.domain == NumericDomain::Real => {
            if !scalar {
                return Some(false);
            }
            request
                .literals
                .literal_args
                .get(index)
                .and_then(|literal| match literal {
                    LiteralValue::Unknown => None,
                    literal => Some(
                        LiteralContext::numeric_from_literal(literal)
                            .and_then(parse_stratify_numeric_boolean)
                            .is_some(),
                    ),
                })
        }
        ValueKindFact::String => {
            if !scalar {
                return Some(false);
            }
            source_text(request, index).map(|text| parse_stratify_boolean_alias(&text).is_some())
        }
        ValueKindFact::Character => {
            if fact
                .shape
                .known_dims()
                .is_some_and(|dims| dims.first().copied().flatten().is_some_and(|rows| rows > 1))
            {
                return Some(false);
            }
            source_text(request, index).map(|text| parse_stratify_boolean_alias(&text).is_some())
        }
        _ => Some(false),
    }
}

fn positive_index_literal(value: &LiteralValue) -> Option<bool> {
    match value {
        LiteralValue::Unknown => None,
        LiteralValue::Integer { text, .. } => {
            Some(text.parse::<usize>().ok().is_some_and(|n| n > 0))
        }
        LiteralValue::Number(_) | LiteralValue::Real { .. } => Some(
            LiteralContext::numeric_from_literal(value)
                .and_then(|n| runmat_types::decode_shape_dimension(n).ok())
                .is_some_and(|n| n > 0),
        ),
        LiteralValue::Bool(value) => Some(*value),
        LiteralValue::Vector(values) => Some(
            !values.is_empty()
                && values
                    .iter()
                    .all(|value| positive_index_literal(value).unwrap_or(true)),
        ),
        _ => None,
    }
}

fn constructor_object() -> ValueFact {
    let scalar_double = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let mut properties = BTreeMap::new();
    properties.insert("NumObservations".into(), scalar_double.clone());
    properties.insert("NumTestSets".into(), scalar_double);
    // A one-test-set partition publishes each size as a scalar; multiple
    // test sets publish a row tensor. The method alone cannot prove which
    // representation is selected for a dynamic partition count.
    let mut size = ValueFact::unknown(DynamicReason::RuntimeValue);
    size.kind = ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    });
    properties.insert("TestSize".into(), size.clone());
    properties.insert("TrainSize".into(), size);
    properties.insert("Type".into(), ValueFact::scalar(ValueKindFact::String));
    for property in ["IsCustom", "IsGrouped", "IsStratified"] {
        properties.insert(property.into(), ValueFact::scalar(ValueKindFact::Logical));
    }
    ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(standard::CVPARTITION.owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    )
}

fn append_diagnostic(
    diagnostics: &mut Vec<InferenceDiagnostic>,
    code: &'static str,
    message: &'static str,
    index: usize,
) {
    diagnostics.push(argument_error(code, message, index));
}

fn constructor_method(
    request: &CallRequest,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> Option<CvPartitionMethod> {
    let first = source_text(request, 0).and_then(|text| CvPartitionMethod::parse(&text));
    if first == Some(CvPartitionMethod::Custom) {
        if request.arguments.len() != 2 {
            append_diagnostic(
                diagnostics,
                "RM-CATALOG-CVPARTITION-ARITY",
                "CustomPartition requires one test-set argument",
                1,
            );
        }
        return first;
    }
    let method = source_text(request, 1).and_then(|text| CvPartitionMethod::parse(&text));
    if request.arguments.len() < 2 {
        append_diagnostic(
            diagnostics,
            "RM-CATALOG-CVPARTITION-ARITY",
            "observation data and partition method are required",
            0,
        );
    } else if source_text(request, 1).is_some() && method.is_none() {
        append_diagnostic(
            diagnostics,
            "RM-CATALOG-CVPARTITION-METHOD",
            "unsupported partition method",
            1,
        );
    }
    method
}

pub(super) fn infer_constructor(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let method = constructor_method(request, &mut diagnostics);
    let custom_first = source_text(request, 0).and_then(|text| CvPartitionMethod::parse(&text))
        == Some(CvPartitionMethod::Custom);
    let n = request
        .literals
        .literal_args
        .first()
        .and_then(literal_positive)
        .and_then(Result::ok);
    if method == Some(CvPartitionMethod::Custom) {
        let payload_index = usize::from(custom_first);
        if request
            .arguments
            .get(payload_index)
            .is_some_and(|fact| !custom_payload_possible(fact))
        {
            append_diagnostic(
                &mut diagnostics,
                "RM-CATALOG-CVPARTITION-CUSTOM",
                "custom test sets must be a logical array or real numeric array",
                payload_index,
            );
        }
    } else if request
        .arguments
        .first()
        .is_some_and(|fact| !observation_input_possible(fact))
    {
        append_diagnostic(
            &mut diagnostics,
            "RM-CATALOG-CVPARTITION-OBSERVATIONS",
            "observation input must be a positive count or vector of labels",
            0,
        );
    }
    let option_start = match method {
        Some(CvPartitionMethod::Custom) => request.arguments.len(),
        Some(CvPartitionMethod::KFold | CvPartitionMethod::Holdout) => 3,
        Some(CvPartitionMethod::Leaveout | CvPartitionMethod::Resubstitution) => 2,
        None => request.arguments.len(),
    };
    if method
        .is_some_and(|kind| matches!(kind, CvPartitionMethod::KFold | CvPartitionMethod::Holdout))
    {
        if request.arguments.len() < 3 {
            append_diagnostic(
                &mut diagnostics,
                "RM-CATALOG-CVPARTITION-ARITY",
                "partition value is required",
                2,
            );
        } else if method == Some(CvPartitionMethod::Holdout) {
            if request
                .literals
                .literal_args
                .get(2)
                .and_then(|value| holdout_literal_valid(value, n))
                == Some(false)
            {
                append_diagnostic(
                    &mut diagnostics,
                    "RM-CATALOG-CVPARTITION-CONTROL",
                    "Holdout must be a fraction in (0,1) or a count in [1,n)",
                    2,
                );
            }
        } else if let Some(Err(())) = request
            .literals
            .literal_args
            .get(2)
            .and_then(literal_positive)
        {
            append_diagnostic(
                &mut diagnostics,
                "RM-CATALOG-CVPARTITION-CONTROL",
                "KFold count must be a positive integer",
                2,
            );
        }
    }
    if option_start < request.arguments.len() {
        if !(request.arguments.len() - option_start).is_multiple_of(2) {
            append_diagnostic(
                &mut diagnostics,
                "RM-CATALOG-CVPARTITION-OPTIONS",
                "options must be paired",
                option_start,
            );
        }
        for index in (option_start..request.arguments.len()).step_by(2) {
            if let Some(name) = source_text(request, index) {
                match CvPartitionOption::parse(&name) {
                    Some(CvPartitionOption::Stratify) => {
                        if stratify_literal_valid(request, index + 1) == Some(false) {
                            append_diagnostic(&mut diagnostics, "RM-CATALOG-CVPARTITION-STRATIFY", "Stratify must be a logical scalar or supported RunMat boolean alias", index + 1);
                        }
                    }
                    None => append_diagnostic(
                        &mut diagnostics,
                        "RM-CATALOG-CVPARTITION-OPTION",
                        "unsupported option",
                        index,
                    ),
                }
            }
        }
    }
    if let Some(CvPartitionMethod::KFold) = method {
        let k = request
            .literals
            .literal_args
            .get(2)
            .and_then(literal_positive)
            .and_then(Result::ok);
        if k.is_some_and(|k| k < 2 || n.is_some_and(|n| k >= n)) {
            append_diagnostic(
                &mut diagnostics,
                "RM-CATALOG-CVPARTITION-KFOLD",
                "KFold count must be in [2,n)",
                2,
            );
        }
    }
    let fact = if diagnostics.is_empty() && method.is_some() {
        constructor_object()
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let mut contract = CallContract::fixed(vec![fact]);
    contract.effects = entry.contract.effect_set();
    if matches!(
        method,
        Some(
            CvPartitionMethod::Leaveout
                | CvPartitionMethod::Resubstitution
                | CvPartitionMethod::Custom
        )
    ) {
        contract.effects.0.remove(&EffectKind::Randomness);
    }
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

fn mask_output(request: &CallRequest) -> ValueFact {
    let mut output = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![None, None]),
        StorageFact::Dense,
    );
    // An index's owning provider can receive the restored logical mask. This
    // is not a host-only output contract, even though selection executes there.
    if request
        .arguments
        .get(1)
        .is_some_and(|fact| matches!(&fact.residency, ResidencyFact::Device { .. }))
    {
        output.residency = request.arguments[1].residency.clone();
    } else if request
        .arguments
        .get(1)
        .is_some_and(|fact| matches!(&fact.residency, ResidencyFact::Unknown))
    {
        output.residency = ResidencyFact::Unknown;
    }
    output
}

fn infer_mask(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() || request.arguments.len() > 2 {
        append_diagnostic(
            &mut diagnostics,
            "RM-CATALOG-CVPARTITION-MASK-ARITY",
            "mask access requires a cvpartition object and optional index",
            0,
        );
    }
    if let Some(value) = request.arguments.first() {
        match &value.kind {
            ValueKindFact::Object(object)
                if object
                    .runtime_class
                    .as_ref()
                    .is_some_and(|class| class.is(standard::CVPARTITION)) => {}
            ValueKindFact::Unknown => {}
            _ => append_diagnostic(
                &mut diagnostics,
                "RM-CATALOG-CVPARTITION-MASK-OBJECT",
                "first argument must be a cvpartition object",
                0,
            ),
        }
    }
    if let Some(index) = request.arguments.get(1) {
        if let Some(text) = source_text(request, 1) {
            if MaskSelectionKeyword::parse(&text).is_none() {
                append_diagnostic(
                    &mut diagnostics,
                    "RM-CATALOG-CVPARTITION-MASK-INDEX",
                    "text index must be 'all'",
                    1,
                );
            }
        } else if !matches!(
            &index.kind,
            ValueKindFact::Unknown
                | ValueKindFact::Numeric(_)
                | ValueKindFact::Logical
                | ValueKindFact::Character
                | ValueKindFact::String
        ) {
            append_diagnostic(
                &mut diagnostics,
                "RM-CATALOG-CVPARTITION-MASK-INDEX",
                "index must be numeric, logical, or 'all'",
                1,
            );
        }
        if request
            .literals
            .literal_args
            .get(1)
            .and_then(positive_index_literal)
            == Some(false)
        {
            append_diagnostic(
                &mut diagnostics,
                "RM-CATALOG-CVPARTITION-MASK-INDEX",
                "index must contain positive one-based integers",
                1,
            );
        }
    }
    let known_object = request.arguments.first().is_some_and(|value| matches!(&value.kind,
        ValueKindFact::Object(object) if object.runtime_class.as_ref().is_some_and(|class| class.is(standard::CVPARTITION))));
    let fact = if diagnostics.is_empty() && known_object {
        mask_output(request)
    } else {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let mut contract = CallContract::fixed(vec![fact]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.diagnostics.extend(diagnostics);
    result
}

pub(super) fn infer_test(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_mask(request, entry)
}

pub(super) fn infer_training(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_mask(request, entry)
}
