//! Pure identities and constructor-option admission for exposed layers.
//!
//! Runtime values and learned parameter payloads remain in `runmat-runtime`;
//! this schema is usable by static analysis without linking an executor.

mod identity;
mod options;

pub use identity::DeepLearningLayerKind;
pub use options::{
    DeepLearningLayerOption, DeepLearningNormalizationAxes, LayerOptionDefault,
    LayerOptionExecution,
};

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
}
