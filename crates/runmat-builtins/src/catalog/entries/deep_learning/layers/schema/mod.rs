//! Pure identities and constructor-option admission for exposed layers.
//!
//! Runtime values and learned parameter payloads remain in `runmat-runtime`;
//! this schema is usable by static analysis without linking an executor.

mod axes;
mod identity;
mod options;
mod properties;

pub use axes::*;
pub use identity::DeepLearningLayerKind;
pub use options::{
    DeepLearningConvolutionPaddingMode, DeepLearningConvolutionPaddingValueMode,
    DeepLearningDenseOperationDimension, DeepLearningLayerOption, DeepLearningNormalizationAxes,
    DeepLearningRecurrentGateActivation, DeepLearningRecurrentOutputMode,
    DeepLearningRecurrentStateActivation, LayerOptionDefault, LayerOptionExecution,
};
pub use properties::DeepLearningLayerStructuralProperty;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_exposed_constructor_round_trips_through_one_class_identity() {
        let names = [
            "featureInputLayer",
            "sequenceInputLayer",
            "fullyConnectedLayer",
            "convolution1dLayer",
            "lstmLayer",
            "bilstmLayer",
            "layerNormalizationLayer",
            "globalAveragePooling1dLayer",
            "reluLayer",
            "eluLayer",
            "softmaxLayer",
            "classificationLayer",
            "regressionLayer",
        ];
        for (name, kind) in names.into_iter().zip(DeepLearningLayerKind::ALL) {
            assert_eq!(DeepLearningLayerKind::from_constructor(name), Some(kind));
            assert_eq!(
                DeepLearningLayerKind::from_identity(&kind.class().owned()),
                Some(kind)
            );
        }
    }

    #[test]
    fn option_inventory_is_layer_specific_and_canonical() {
        use DeepLearningLayerKind as K;
        use DeepLearningLayerOption as O;
        assert_eq!(K::SequenceInput.option("MINlength"), Some(O::MinLength));
        assert_eq!(
            K::Convolution1D.option("paddingvalue"),
            Some(O::PaddingValue)
        );
        assert_eq!(
            K::Lstm.option("RecurrentWeights"),
            Some(O::RecurrentWeights)
        );
        assert_eq!(
            K::LayerNormalization.option("OperationDimension"),
            Some(O::OperationDimension)
        );
        assert_eq!(
            K::ClassificationOutput.option("ClassWeights"),
            Some(O::ClassWeights)
        );
        assert_eq!(
            K::RegressionOutput.option("ResponseNames"),
            Some(O::ResponseNames)
        );
        assert_eq!(
            K::FullyConnected.option("WeightsLearnRateFactor"),
            Some(O::WeightLearnRateFactor)
        );
        assert_eq!(
            O::WeightLearnRateFactor.canonical_name(),
            "WeightLearnRateFactor"
        );
        assert_eq!(K::Relu.option("Padding"), None);
        assert_eq!(K::Convolution1D.option("MinLength"), None);
        assert_eq!(K::FeatureInput.option("invented"), None);
        assert_eq!(
            O::Name.execution_for(K::Lstm),
            LayerOptionExecution::Metadata
        );
        assert_eq!(
            O::Weights.execution_for(K::FullyConnected),
            LayerOptionExecution::Executable
        );
        assert_eq!(
            O::Weights.execution_for(K::Convolution1D),
            LayerOptionExecution::PendingExecution
        );
    }

    #[test]
    fn dense_operation_dimension_is_distinct_from_normalization_axes() {
        use DeepLearningDenseOperationDimension as D;
        assert_eq!(D::from_name("SPATIAL-CHANNEL"), Some(D::SpatialChannel));
        assert_eq!(D::from_name("batch-excluded"), None);
        assert_eq!(D::NAMES, &[D::SpatialChannel.canonical_name()]);
        assert_eq!(
            DeepLearningLayerOption::OperationDimension
                .text_values_for(DeepLearningLayerKind::FullyConnected),
            Some(D::NAMES)
        );
        assert!(DeepLearningNormalizationAxes::from_name("batch-excluded").is_some());
    }

    #[test]
    fn execution_modes_round_trip_through_constructor_choices() {
        use DeepLearningLayerKind as K;
        use DeepLearningLayerOption as O;

        for (option, names) in [
            (O::Padding, DeepLearningConvolutionPaddingMode::NAMES),
            (
                O::PaddingValue,
                DeepLearningConvolutionPaddingValueMode::NAMES,
            ),
        ] {
            assert_eq!(option.text_values_for(K::Convolution1D), Some(names));
        }
        for mode in DeepLearningConvolutionPaddingMode::ALL {
            assert_eq!(
                DeepLearningConvolutionPaddingMode::from_name(
                    &mode.canonical_name().to_ascii_uppercase()
                ),
                Some(mode)
            );
        }
        for mode in DeepLearningConvolutionPaddingValueMode::ALL {
            assert_eq!(
                DeepLearningConvolutionPaddingValueMode::from_name(
                    &mode.canonical_name().to_ascii_uppercase()
                ),
                Some(mode)
            );
        }
        for kind in [K::Lstm, K::Bilstm] {
            assert_eq!(
                O::OutputMode.text_values_for(kind),
                Some(DeepLearningRecurrentOutputMode::NAMES)
            );
            assert_eq!(
                O::StateActivationFunction.text_values_for(kind),
                Some(DeepLearningRecurrentStateActivation::NAMES)
            );
            assert_eq!(
                O::GateActivationFunction.text_values_for(kind),
                Some(DeepLearningRecurrentGateActivation::NAMES)
            );
        }
        for mode in DeepLearningRecurrentOutputMode::ALL {
            assert_eq!(
                DeepLearningRecurrentOutputMode::from_name(
                    &mode.canonical_name().to_ascii_uppercase()
                ),
                Some(mode)
            );
        }
        for mode in DeepLearningRecurrentStateActivation::ALL {
            assert_eq!(
                DeepLearningRecurrentStateActivation::from_name(
                    &mode.canonical_name().to_ascii_uppercase()
                ),
                Some(mode)
            );
        }
        for mode in DeepLearningRecurrentGateActivation::ALL {
            assert_eq!(
                DeepLearningRecurrentGateActivation::from_name(
                    &mode.canonical_name().to_ascii_uppercase()
                ),
                Some(mode)
            );
        }
        assert_eq!(
            DeepLearningConvolutionPaddingMode::from_name("unknown"),
            None
        );
        assert_eq!(
            DeepLearningConvolutionPaddingValueMode::from_name("unknown"),
            None
        );
        assert_eq!(DeepLearningRecurrentOutputMode::from_name("unknown"), None);
        assert_eq!(
            DeepLearningRecurrentStateActivation::from_name("unknown"),
            None
        );
        assert_eq!(
            DeepLearningRecurrentGateActivation::from_name("unknown"),
            None
        );
    }

    #[test]
    fn read_only_structural_properties_are_distinct_from_options() {
        use DeepLearningLayerKind as K;
        use DeepLearningLayerStructuralProperty as P;
        for property in P::ALL {
            assert_eq!(P::from_name(property.canonical_name()), Some(*property));
        }
        assert_eq!(P::from_name("inputsize"), None);
        assert!(
            P::from_name("InputSize").is_some_and(|property| property.belongs_to(K::FeatureInput))
        );
        assert!(!P::from_name("InputSize")
            .is_some_and(|property| property.belongs_to(K::FullyConnected)));
        assert!(K::FullyConnected.option("InputSize").is_some());
        assert!(P::from_name("NumHiddenUnits").is_some_and(|property| property.belongs_to(K::Lstm)));
        assert!(!P::from_name("NumHiddenUnits")
            .is_some_and(|property| property.belongs_to(K::Convolution1D)));
    }
}
