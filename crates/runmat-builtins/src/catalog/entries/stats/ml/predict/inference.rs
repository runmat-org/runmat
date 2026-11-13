use crate::{
    catalog::inference::{
        argument_error,
        deep_learning::{infer_network_output, NetworkInferenceCaller},
    },
    BuiltinCatalogEntry,
};
use runmat_types::{
    infer_call, standard, CallContract, CallInference, CallRequest, DynamicReason, NumericClass,
    NumericDomain, NumericFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
};

#[derive(Clone, Copy, Eq, PartialEq)]
enum ModelKind {
    Linear,
    Tree,
    Classifier,
    Network { supports_state: bool },
    Unknown,
    Unsupported,
}

fn model_kind(value: Option<&ValueFact>) -> ModelKind {
    let Some(value) = value else {
        return ModelKind::Unknown;
    };
    match &value.kind {
        ValueKindFact::Unknown => ModelKind::Unknown,
        ValueKindFact::Object(object) => match object.runtime_class.as_ref() {
            Some(class) if class.is(standard::LINEAR_MODEL) => ModelKind::Linear,
            Some(class) if class.is(standard::CLASSIFICATION_TREE) => ModelKind::Tree,
            Some(class) if class.is(standard::CLASSIFICATION_LINEAR) => ModelKind::Classifier,
            Some(class) if class.is(standard::DL_NETWORK) => ModelKind::Network {
                supports_state: true,
            },
            Some(class)
                if class.is(standard::SERIES_NETWORK) || class.is(standard::DAG_NETWORK) =>
            {
                ModelKind::Network {
                    supports_state: false,
                }
            }
            Some(_) => ModelKind::Unsupported,
            None => ModelKind::Unknown,
        },
        _ => ModelKind::Unsupported,
    }
}

fn numeric(class: NumericClass) -> ValueFact {
    let mut output = ValueFact::unknown(DynamicReason::RuntimeValue);
    output.kind = ValueKindFact::Numeric(NumericFact {
        class,
        domain: NumericDomain::Real,
    });
    output.shape = ShapeFact::Unknown;
    output.storage = StorageFact::Dense;
    output
}

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 2 {
        diagnostics.push(argument_error(
            "RM-PREDICT-ARITY",
            "predict requires a model and predictor data",
            request.arguments.len(),
        ));
    }
    let model = model_kind(request.arguments.first());
    if model == ModelKind::Unsupported {
        diagnostics.push(argument_error(
            "RM-PREDICT-MODEL",
            "predict requires a supported fitted model or network object",
            0,
        ));
    }
    let max_outputs = match model {
        ModelKind::Linear
        | ModelKind::Classifier
        | ModelKind::Network {
            supports_state: true,
        } => 2,
        ModelKind::Tree => 4,
        ModelKind::Network {
            supports_state: false,
        }
        | ModelKind::Unsupported => 1,
        ModelKind::Unknown => 4,
    };
    let requested = request.outputs.requested.known_count();
    if requested.is_some_and(|count| count > max_outputs) {
        diagnostics.push(argument_error(
            "RM-PREDICT-OUTPUTS",
            "requested outputs exceed the selected model's supported outputs",
            0,
        ));
    }
    let first = match model {
        ModelKind::Linear => numeric(NumericClass::Double),
        ModelKind::Tree | ModelKind::Classifier | ModelKind::Unknown | ModelKind::Unsupported => {
            ValueFact::unknown(DynamicReason::RuntimeValue)
        }
        ModelKind::Network { .. } => {
            let output = infer_network_output(
                request.arguments.first(),
                request.arguments.get(1),
                NetworkInferenceCaller::Predict,
            );
            diagnostics.extend(output.diagnostics());
            output.output
        }
    };
    let mut outputs = Vec::new();
    let count = requested.unwrap_or(1).min(max_outputs);
    if count != 0 {
        outputs.push(first);
    }
    for index in 1..count {
        outputs.push(match (model, index) {
            (ModelKind::Linear, 1) | (ModelKind::Tree, 1..=3) | (ModelKind::Classifier, 1) => {
                numeric(NumericClass::Double)
            }
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        });
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = requested.is_none() || model == ModelKind::Unknown;
    result.diagnostics.extend(diagnostics);
    result
}
