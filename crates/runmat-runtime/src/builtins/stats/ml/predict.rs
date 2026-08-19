//! Generic model prediction dispatcher.

use runmat_builtins::{
    BuiltinExtensionDescriptor, BuiltinExtensionMode, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::classification_linear::{
    predict_classification_linear_object, CLASSIFICATION_LINEAR_CLASS,
};
use super::classification_tree::{predict_classification_tree_object, CLASSIFICATION_TREE_CLASS};
use super::linear_model::{predict_invalid, predict_linear_model_dispatch, predict_type};

const PREDICT_INTEGER_PREDICTORS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "predict-integer-statistical-predictors",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "predict accepts typed-integer statistical-model predictors as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PredictIntegerStatisticalPredictorsExtension"),
    };
const PREDICT_INTEGER_CONTROLS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "predict-integer-statistical-controls",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "predict accepts typed-integer statistical-model controls as a RunMat extension",
    error_identifier: Some("RunMat:compatibility:PredictIntegerStatisticalControlsExtension"),
};
pub const PREDICT_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    PREDICT_INTEGER_PREDICTORS_EXTENSION,
    PREDICT_INTEGER_CONTROLS_EXTENSION,
];
const PREDICT_INTEGER_STATS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xnew for implemented statistical models",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The implemented linear-regression, classification-tree, and linear-classifier references document single, double, or table predictors. RunMat admits typed integer matrices only at a checked model-specific floating boundary.",
    }];
const PREDICT_INTEGER_CONTROL_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Alpha, Subtrees, or other numeric statistical controls",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Typed integer numeric controls are independently gated; unsupported general options remain ordinary surface gaps rather than silently admitted integer forms.",
    }];
const PREDICT_INTEGER_DEEP_LEARNING_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X for supported deep-learning network objects",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The public deep-learning predict surface explicitly includes all eight integer input classes; the selected network determines output class and conversion behavior.",
    }];
pub const PREDICT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "ypred = predict(statistical_model,integer_Xnew,___)",
        inputs: &PREDICT_INTEGER_STATS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Statistical integer predictors are gated before gather and model dispatch and must be exactly representable at the binary64 prediction boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "ypred = predict(statistical_model,Xnew,integer_control,___)",
        inputs: &PREDICT_INTEGER_CONTROL_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Numeric statistical controls are separately admitted and checked before the selected model parser runs.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Y = predict(deep_learning_network,integer_X,___)",
        inputs: &PREDICT_INTEGER_DEEP_LEARNING_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FunctionSpecific,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::FunctionSpecific,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Deep-learning integer admission is documented and is intentionally not covered by the statistical-model compatibility gates.",
    },
];

#[runtime_builtin(
    name = "predict",
    category = "stats/ml",
    summary = "Predict responses from a fitted model.",
    keywords = "predict,fitlm,linear model,classification,deep learning,prediction",
    type_resolver(predict_type),
    descriptor(crate::builtins::stats::ml::linear_model::PREDICT_DESCRIPTOR),
    extensions(crate::builtins::stats::ml::predict::PREDICT_EXTENSIONS),
    integer_capabilities(crate::builtins::stats::ml::predict::PREDICT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::ml::predict"
)]
pub(crate) async fn predict_builtin(
    model: Value,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let statistical_model = matches!(
        &model,
        Value::Object(object)
            if object.class_name == "LinearModel"
                || object.class_name == CLASSIFICATION_TREE_CLASS
                || object.class_name == CLASSIFICATION_LINEAR_CLASS
    );
    if statistical_model {
        crate::builtins::common::validation::reject_typed_complex_integer(&xnew, "predict")?;
        for control in &rest {
            crate::builtins::common::validation::reject_typed_complex_integer(control, "predict")?;
        }
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            &xnew,
            &PREDICT_INTEGER_PREDICTORS_EXTENSION,
            "predict",
            "predictor",
        )
        .await?;
        for control in &rest {
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                control,
                &PREDICT_INTEGER_CONTROLS_EXTENSION,
                "predict",
                "control",
            )
            .await?;
        }
    }
    let model = gather(model)
        .await
        .map_err(|err| predict_invalid(err.message))?;
    let xnew = gather(xnew)
        .await
        .map_err(|err| predict_invalid(err.message))?;
    let rest = gather_all(rest)
        .await
        .map_err(|err| predict_invalid(err.message))?;
    let output = match model {
        Value::Object(object)
            if crate::builtins::deep_learning::model::is_deep_learning_network_object(&object) =>
        {
            crate::builtins::deep_learning::model::predict_deep_learning_object(object, xnew, rest)?
        }
        Value::Object(object) if object.class_name == CLASSIFICATION_TREE_CLASS => {
            predict_classification_tree_object(object, xnew, rest)?
        }
        Value::Object(object) if object.class_name == CLASSIFICATION_LINEAR_CLASS => {
            predict_classification_linear_object(object, xnew, rest)?
        }
        other => predict_linear_model_dispatch(other, xnew, rest)?,
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![output[0].clone()])),
        Some(out_count) if out_count > output.len() => {
            Err(predict_invalid("predict: too many output arguments"))
        }
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count, output,
        )),
        None => Ok(output
            .into_iter()
            .next()
            .expect("predict dispatch always returns at least one output")),
    }
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}
