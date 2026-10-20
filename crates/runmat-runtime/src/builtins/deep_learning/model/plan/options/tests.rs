use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use runmat_builtins::{
    CONVOLUTION_1D_LAYER_CLASS, FEATURE_INPUT_LAYER_CLASS, FULLY_CONNECTED_LAYER_CLASS,
    LAYER_NORMALIZATION_LAYER_CLASS, LSTM_LAYER_CLASS, RELU_LAYER_CLASS,
};
use runmat_types::StaticClassIdentity;
use runmat_value::ObjectInstance;

fn layer(class: StaticClassIdentity, properties: Vec<(&str, Value)>) -> Value {
    let mut object = ObjectInstance::new(class.owned());
    for (name, value) in properties {
        object
            .replace_dl_property(name, value)
            .expect("fixture property");
    }
    Value::Object(object)
}

#[test]
fn defaults_and_metadata_do_not_require_an_execution_consumer() {
    let layers = [
        layer(
            FEATURE_INPUT_LAYER_CLASS,
            vec![
                ("InputSize", Value::Num(3.0)),
                ("Normalization", Value::String("none".into())),
                ("Name", Value::String("input".into())),
            ],
        ),
        layer(
            CONVOLUTION_1D_LAYER_CLASS,
            vec![
                ("FilterSize", Value::Num(3.0)),
                ("NumFilters", Value::Num(4.0)),
                ("Padding", Value::Num(0.0)),
                ("Stride", Value::Num(1.0)),
            ],
        ),
        layer(
            LAYER_NORMALIZATION_LAYER_CLASS,
            vec![("Epsilon", Value::Num(1.0e-5))],
        ),
        layer(
            LSTM_LAYER_CLASS,
            vec![
                ("NumHiddenUnits", Value::Num(4.0)),
                ("OutputMode", Value::String("sequence".into())),
                ("HasStateInputs", Value::Bool(false)),
            ],
        ),
    ];
    let plan = LayerOperationalPlan::inspect(&layers, "forward").unwrap();
    assert!(plan
        .requirements()
        .iter()
        .all(|requirement| requirement.status != OptionRequirementStatus::NeedsConsumer));
    plan.require_consumed("forward", |_| false).unwrap();
}

#[test]
fn convolution_same_padding_is_not_misclassified_as_the_default() {
    let layers = [layer(
        CONVOLUTION_1D_LAYER_CLASS,
        vec![("Padding", Value::String("same".into()))],
    )];
    let plan = LayerOperationalPlan::inspect(&layers, "forward").unwrap();
    assert!(plan.require_consumed("forward", |_| false).is_err());
    plan.require_consumed("forward", |requirement| {
        requirement.option == DeepLearningLayerOption::Padding
    })
    .unwrap();
}

#[test]
fn nondefault_options_require_actual_route_consumption() {
    let layers = [
        layer(
            FULLY_CONNECTED_LAYER_CLASS,
            vec![
                ("OutputSize", Value::Num(2.0)),
                ("Weights", Value::Num(2.0)),
                ("Bias", Value::Num(0.0)),
            ],
        ),
        layer(
            CONVOLUTION_1D_LAYER_CLASS,
            vec![("Stride", Value::Num(2.0))],
        ),
    ];
    let plan = LayerOperationalPlan::inspect(&layers, "forward").unwrap();
    let error = plan
        .require_consumed("forward", |requirement| {
            requirement.kind == DeepLearningLayerKind::FullyConnected
                && matches!(
                    requirement.option,
                    DeepLearningLayerOption::Weights | DeepLearningLayerOption::Bias
                )
        })
        .expect_err("unconsumed Conv1D stride");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    assert!(error.to_string().contains("Stride"));
    plan.require_consumed("forward", |requirement| {
        matches!(
            requirement.option,
            DeepLearningLayerOption::Weights
                | DeepLearningLayerOption::Bias
                | DeepLearningLayerOption::Stride
        )
    })
    .unwrap();
}

#[test]
fn unknown_properties_and_duplicate_aliases_are_not_silent() {
    let unknown = [layer(
        RELU_LAYER_CLASS,
        vec![("Padding", Value::String("same".into()))],
    )];
    assert!(LayerOperationalPlan::inspect(&unknown, "forward").is_err());

    let duplicate = [layer(
        FULLY_CONNECTED_LAYER_CLASS,
        vec![
            ("WeightLearnRateFactor", Value::Num(1.0)),
            ("weightslearnratefactor", Value::Num(2.0)),
        ],
    )];
    assert!(LayerOperationalPlan::inspect(&duplicate, "forward").is_err());
}

#[test]
fn explicit_nondefault_normalization_control_remains_a_requirement() {
    let layers = [layer(
        LAYER_NORMALIZATION_LAYER_CLASS,
        vec![
            ("OperationDimension", Value::String("batch-excluded".into())),
            ("Epsilon", Value::Num(1.0e-4)),
        ],
    )];
    let plan = LayerOperationalPlan::inspect(&layers, "forward").unwrap();
    assert_eq!(
        plan.requirements()
            .iter()
            .filter(|requirement| requirement.status == OptionRequirementStatus::NeedsConsumer)
            .count(),
        2
    );
    assert!(plan.require_consumed("forward", |_| false).is_err());
}

#[test]
fn callable_normalization_requires_a_consumer_without_invocation_during_inspection() {
    let layers = [layer(
        FEATURE_INPUT_LAYER_CLASS,
        vec![
            ("InputSize", Value::Num(2.0)),
            (
                "Normalization",
                Value::FunctionHandle("__missing_normalization_callback".into()),
            ),
        ],
    )];
    let plan = LayerOperationalPlan::inspect(&layers, "forward").unwrap();
    let requirement = plan
        .requirements()
        .iter()
        .find(|requirement| requirement.option == DeepLearningLayerOption::Normalization)
        .unwrap();
    assert_eq!(requirement.status, OptionRequirementStatus::NeedsConsumer);
    assert!(plan.require_consumed("forward", |_| false).is_err());
}
