use super::schema::{
    DeepLearningDenseOperationDimension, DeepLearningLayerKind as LayerKind,
    DeepLearningLayerOption, DeepLearningNormalizationAxes,
};
use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain,
    NumericFact, ObjectFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::{BTreeMap, BTreeSet};

fn number() -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }))
}

fn text() -> ValueFact {
    ValueFact::scalar(ValueKindFact::String)
}

fn numeric_possible(value: &ValueFact, scalar: bool) -> bool {
    matches!(
        &value.kind,
        ValueKindFact::Unknown
            | ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
    ) && (!scalar || value.shape.element_count().is_none_or(|n| n == 1))
}

fn invalid_dimension_literal(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Integer { text, .. } => text.parse::<i128>().is_ok_and(|value| {
            let magnitude = value.unsigned_abs();
            if magnitude == 0 || magnitude > u64::MAX as u128 || value < 0 {
                return true;
            }
            let significant_bits = u128::BITS - magnitude.leading_zeros();
            significant_bits > f64::MANTISSA_DIGITS
                && magnitude.trailing_zeros() < significant_bits - f64::MANTISSA_DIGITS
        }),
        LiteralValue::Real { text, .. } => text.parse::<f64>().is_ok_and(|value| {
            !value.is_finite() || value < 1.0 || value.fract() != 0.0 || value > usize::MAX as f64
        }),
        LiteralValue::Number(value) => {
            !value.is_finite() || *value < 1.0 || value.fract() != 0.0 || *value > usize::MAX as f64
        }
        LiteralValue::Vector(values) => {
            values.is_empty() || values.iter().any(invalid_dimension_literal)
        }
        _ => false,
    }
}

fn invalid_nonnegative_factor_literal(literal: &LiteralValue) -> bool {
    match literal {
        LiteralValue::Vector(values) => values.iter().any(invalid_nonnegative_factor_literal),
        LiteralValue::Matrix(rows) => rows
            .iter()
            .flatten()
            .any(invalid_nonnegative_factor_literal),
        _ => LiteralContext::numeric_from_literal(literal)
            .is_some_and(|value| !value.is_finite() || value < 0.0),
    }
}

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry, kind: LayerKind) -> CallInference {
    let mut diagnostics = Vec::new();
    let required = kind.required_sizes();
    if request.arguments.len() < required {
        diagnostics.push(argument_error(
            "RM-DL-LAYER-ARITY",
            "missing required layer size",
            request.arguments.len(),
        ));
    }
    for index in 0..required.min(request.arguments.len()) {
        let scalar = !matches!(kind, LayerKind::FeatureInput | LayerKind::SequenceInput);
        if !numeric_possible(&request.arguments[index], scalar)
            || request.arguments[index].shape.element_count() == Some(0)
            || request
                .literals
                .literal_args
                .get(index)
                .is_some_and(invalid_dimension_literal)
        {
            diagnostics.push(argument_error(
                "RM-DL-LAYER-SIZE",
                "layer size must contain positive integers",
                index,
            ));
        }
    }
    let mut options_start = Some(required);
    if matches!(kind, LayerKind::Elu)
        && request
            .arguments
            .first()
            .is_some_and(|arg| matches!(arg.kind, ValueKindFact::Unknown))
    {
        // A dynamic first input can be either alpha or the first option name.
        // Neither interpretation is proven at analysis time.
        options_start = None;
    } else if matches!(kind, LayerKind::Elu)
        && request.arguments.first().is_some_and(|arg| {
            matches!(arg.kind, ValueKindFact::Numeric(_)) && numeric_possible(arg, true)
        })
    {
        options_start = Some(1);
        if request
            .literals
            .numeric_at(0)
            .is_some_and(|n| !n.is_finite())
        {
            diagnostics.push(argument_error(
                "RM-DL-ELU-ALPHA",
                "alpha must be a finite numeric scalar",
                0,
            ));
        }
    }
    let mut positional_dimension = None;
    if matches!(kind, LayerKind::FullyConnected) {
        if let Some(dimension) = request.arguments.get(1) {
            let documented_text = request.literals.literal_string_at(1).is_some_and(|value| {
                DeepLearningDenseOperationDimension::from_name(&value).is_some()
            });
            if documented_text || matches!(dimension.kind, ValueKindFact::Numeric(_)) {
                options_start = Some(2);
                positional_dimension = Some(dimension.clone());
                if !documented_text
                    && (!numeric_possible(dimension, true)
                        || request
                            .literals
                            .literal_args
                            .get(1)
                            .is_some_and(invalid_dimension_literal))
                {
                    diagnostics.push(argument_error(
                        "RM-DL-LAYER-DIMENSION",
                        "operation dimension must be spatial-channel or a positive integer",
                        1,
                    ));
                }
            } else if matches!(dimension.kind, ValueKindFact::Unknown) {
                // Could be a dynamic option name or the positional dimension.
                options_start = None;
            }
        }
    }
    if let Some(start) = options_start {
        if request.arguments.len() >= start && !(request.arguments.len() - start).is_multiple_of(2)
        {
            diagnostics.push(argument_error(
                "RM-DL-LAYER-OPTIONS",
                "layer name-value options must be paired",
                request.arguments.len().saturating_sub(1),
            ));
        }
    }
    let mut supplied_options = BTreeMap::new();
    let mut seen_options = BTreeSet::new();
    if let Some(dimension) = positional_dimension {
        seen_options.insert(DeepLearningLayerOption::OperationDimension);
        supplied_options.insert("OperationDimension".into(), dimension);
    }
    for index in
        (options_start.unwrap_or(request.arguments.len())..request.arguments.len()).step_by(2)
    {
        let name = &request.arguments[index];
        if !matches!(
            name.kind,
            ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
        ) {
            diagnostics.push(argument_error(
                "RM-DL-LAYER-OPTION-NAME",
                "option names must be text",
                index,
            ));
        }
        let known_option = request.literals.literal_string_at(index).and_then(|name| {
            let option = kind.option(&name);
            if option.is_none() {
                diagnostics.push(argument_error(
                    "RM-DL-LAYER-OPTION-UNKNOWN",
                    "option is not accepted by this layer constructor",
                    index,
                ));
            }
            option
        });
        if let Some(option) = known_option {
            if !seen_options.insert(option) {
                diagnostics.push(argument_error(
                    "RM-DL-LAYER-OPTION-DUPLICATE",
                    "layer option is specified more than once",
                    index,
                ));
            }
            if let Some(value) = request.arguments.get(index + 1) {
                supplied_options.insert(option.canonical_name().into(), value.clone());
                if let Some(gate_count) = kind
                    .recurrent_gate_count()
                    .filter(|_| option.is_recurrent_gate_factor())
                {
                    let scalar = value.shape.element_count() == Some(1);
                    let documented_row = value
                        .shape
                        .known_dims()
                        .is_some_and(|dims| dims.as_slice() == [Some(1), Some(gate_count)]);
                    if !numeric_possible(value, false)
                        || value.shape.element_count() == Some(0)
                        || (value.shape.element_count().is_some() && !scalar && !documented_row)
                        || request
                            .literals
                            .literal_args
                            .get(index + 1)
                            .is_some_and(invalid_nonnegative_factor_literal)
                    {
                        diagnostics.push(argument_error(
                            "RM-DL-LAYER-OPTION-VALUE",
                            "recurrent gate factor must contain finite nonnegative numeric values in a scalar or 1-by-4/1-by-8 row vector",
                            index + 1,
                        ));
                    }
                }
                if matches!(
                    option,
                    DeepLearningLayerOption::OutputMode
                        | DeepLearningLayerOption::NormalizationDimension
                        | DeepLearningLayerOption::StateActivationFunction
                        | DeepLearningLayerOption::GateActivationFunction
                ) && !matches!(
                    value.kind,
                    ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
                ) {
                    diagnostics.push(argument_error(
                        "RM-DL-LAYER-OPTION-VALUE",
                        "option value must be one of the documented text choices",
                        index + 1,
                    ));
                }
                if let (Some(text), Some(choices)) = (
                    request.literals.literal_string_at(index + 1),
                    option.text_values_for(kind),
                ) {
                    if !choices
                        .iter()
                        .any(|choice| text.eq_ignore_ascii_case(choice))
                    {
                        diagnostics.push(argument_error(
                            "RM-DL-LAYER-OPTION-VALUE",
                            "option text is not accepted by this layer",
                            index + 1,
                        ));
                    }
                }
                if matches!(kind, LayerKind::LayerNormalization)
                    && matches!(option, DeepLearningLayerOption::OperationDimension)
                    && request
                        .literals
                        .literal_string_at(index + 1)
                        .is_some_and(|text| {
                            DeepLearningNormalizationAxes::from_name(&text).is_none()
                        })
                {
                    diagnostics.push(argument_error(
                        "RM-DL-LAYER-OPTION-VALUE",
                        "normalization axis is not accepted by this layer",
                        index + 1,
                    ));
                }
            }
        }
        if matches!(
            known_option,
            Some(DeepLearningLayerOption::Name | DeepLearningLayerOption::Description)
        ) {
            if let Some(value) = request.arguments.get(index + 1) {
                if !matches!(
                    value.kind,
                    ValueKindFact::Unknown | ValueKindFact::String | ValueKindFact::Character
                ) {
                    diagnostics.push(argument_error(
                        "RM-DL-LAYER-OPTION-VALUE",
                        "Name and Description must be text",
                        index + 1,
                    ));
                }
            }
        }
    }
    let mut properties = BTreeMap::from([
        ("Type".into(), text()),
        ("Name".into(), text()),
        ("Description".into(), text()),
    ]);
    match kind {
        LayerKind::FullyConnected => {
            properties.insert("OutputSize".into(), number());
        }
        LayerKind::FeatureInput | LayerKind::SequenceInput => {
            properties.insert(
                "InputSize".into(),
                ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    ShapeFact::from(vec![Some(1), None]),
                    StorageFact::Dense,
                ),
            );
        }
        LayerKind::Elu => {
            properties.insert("Alpha".into(), number());
        }
        LayerKind::Lstm | LayerKind::Bilstm => {
            properties.insert("NumHiddenUnits".into(), number());
        }
        LayerKind::Convolution1D => {
            properties.insert("FilterSize".into(), number());
            properties.insert("NumFilters".into(), number());
        }
        _ => {}
    }
    properties.extend(supplied_options);
    let result = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(kind.class().owned()),
            properties,
            properties_complete: false,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    finish_fixed(entry, request, result, diagnostics)
}

macro_rules! layer_inference {
    ($function:ident, $kind:ident) => {
        pub(super) fn $function(
            request: &CallRequest,
            entry: &BuiltinCatalogEntry,
        ) -> CallInference {
            infer(request, entry, LayerKind::$kind)
        }
    };
}

layer_inference!(fully_connected, FullyConnected);
layer_inference!(feature_input, FeatureInput);
layer_inference!(sequence_input, SequenceInput);
layer_inference!(relu, Relu);
layer_inference!(elu, Elu);
layer_inference!(softmax, Softmax);
layer_inference!(classification, ClassificationOutput);
layer_inference!(regression, RegressionOutput);
layer_inference!(layer_norm, LayerNormalization);
layer_inference!(global_pool, GlobalAveragePooling1D);
layer_inference!(lstm, Lstm);
layer_inference!(bilstm, Bilstm);
layer_inference!(convolution, Convolution1D);
