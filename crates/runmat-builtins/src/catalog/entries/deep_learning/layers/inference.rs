use crate::{
    catalog::inference::{argument_error, finish_fixed},
    BuiltinCatalogEntry,
};
use runmat_types::{
    CallInference, CallRequest, LiteralValue, NumericClass, NumericDomain, NumericFact, ObjectFact,
    ShapeFact, StaticClassIdentity, StorageFact, ValueFact, ValueKindFact,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
enum LayerKind {
    FullyConnected,
    FeatureInput,
    SequenceInput,
    Relu,
    Elu,
    Softmax,
    Classification,
    Regression,
    LayerNorm,
    GlobalPool,
    Lstm,
    Bilstm,
    Convolution,
}

impl LayerKind {
    fn class(self) -> StaticClassIdentity {
        StaticClassIdentity::new(match self {
            Self::FullyConnected => "nnet.cnn.layer.FullyConnectedLayer",
            Self::FeatureInput => "nnet.cnn.layer.FeatureInputLayer",
            Self::SequenceInput => "nnet.cnn.layer.SequenceInputLayer",
            Self::Relu => "nnet.cnn.layer.ReLULayer",
            Self::Elu => "nnet.cnn.layer.ELULayer",
            Self::Softmax => "nnet.cnn.layer.SoftmaxLayer",
            Self::Classification => "nnet.cnn.layer.ClassificationOutputLayer",
            Self::Regression => "nnet.cnn.layer.RegressionOutputLayer",
            Self::LayerNorm => "nnet.cnn.layer.LayerNormalizationLayer",
            Self::GlobalPool => "nnet.cnn.layer.GlobalAveragePooling1DLayer",
            Self::Lstm => "nnet.cnn.layer.LSTMLayer",
            Self::Bilstm => "nnet.cnn.layer.BiLSTMLayer",
            Self::Convolution => "nnet.cnn.layer.Convolution1DLayer",
        })
    }

    fn required_sizes(self) -> usize {
        match self {
            Self::FullyConnected
            | Self::FeatureInput
            | Self::SequenceInput
            | Self::Lstm
            | Self::Bilstm => 1,
            Self::Convolution => 2,
            _ => 0,
        }
    }
}

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
        if request
            .literals
            .literal_string_at(index)
            .is_some_and(|name| matches!(name.as_str(), "name" | "description"))
        {
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
        LayerKind::Convolution => {
            properties.insert("FilterSize".into(), number());
            properties.insert("NumFilters".into(), number());
        }
        _ => {}
    }
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
layer_inference!(classification, Classification);
layer_inference!(regression, Regression);
layer_inference!(layer_norm, LayerNorm);
layer_inference!(global_pool, GlobalPool);
layer_inference!(lstm, Lstm);
layer_inference!(bilstm, Bilstm);
layer_inference!(convolution, Convolution);
