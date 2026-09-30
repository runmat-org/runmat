use crate::*;

pub const PREDICT_INTEGER_PREDICTORS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "predict-integer-statistical-predictors",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "predict accepts typed-integer statistical-model predictors as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PredictIntegerStatisticalPredictorsExtension"),
    };
pub const PREDICT_INTEGER_CONTROLS_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "predict-integer-statistical-controls",
        mode: BuiltinExtensionMode::RunMatOnly,
        description:
            "predict accepts typed-integer statistical-model controls as a RunMat extension",
        error_identifier: Some("RunMat:compatibility:PredictIntegerStatisticalControlsExtension"),
    };

pub(super) const EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    PREDICT_INTEGER_PREDICTORS_EXTENSION,
    PREDICT_INTEGER_CONTROLS_EXTENSION,
];
