//! Pure caller and terminal-target admission for shared network training.

use runmat_builtins::{DeepLearningAxisLabel, DeepLearningLayerKind as K};
use runmat_value::{NumericDType, Tensor, Value};

use crate::builtins::deep_learning::{
    deep_learning_error, model::plan::axes::AxisPlan, string_array,
};
use crate::BuiltinResult;

use super::super::{
    targets::{is_label_target, label_strings},
    LossKind,
};
use super::{labels::LabelTargetPlan, loss::PredictionDomain, target::NumericTargetPlan};

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

#[derive(Clone, Copy)]
pub(in crate::builtins::deep_learning::supervised) enum TrainingCaller {
    TrainNetwork,
    Trainnet,
}

impl TrainingCaller {
    pub(super) const fn function(self) -> &'static str {
        match self {
            Self::TrainNetwork => "trainNetwork",
            Self::Trainnet => "trainnet",
        }
    }
}

pub(super) enum TrainingTarget<'a> {
    Numeric {
        plan: NumericTargetPlan<'a>,
        loss: LossKind,
        channels: usize,
        function: &'static str,
    },
    Labels(LabelTargetPlan),
}

impl<'a> TrainingTarget<'a> {
    pub(super) fn admit(
        caller: TrainingCaller,
        declared_loss: Option<LossKind>,
        layers: &[Value],
        target: &'a Value,
        output: &AxisPlan,
    ) -> BuiltinResult<Self> {
        let function = caller.function();
        let loss = match declared_loss {
            Some(loss) => loss,
            None => infer_loss(layers, is_label_target(target), function)?,
        };
        let labels = if loss == LossKind::Classification {
            label_strings(target, function)?
        } else {
            None
        };
        match (loss, labels) {
            (LossKind::Classification, Some(labels)) => Ok(Self::Labels(
                LabelTargetPlan::from_labels(labels, output, function)?,
            )),
            (LossKind::Regression, Some(_)) => {
                Err(fail(function, "regression targets must be numeric arrays"))
            }
            (loss, None) => Ok(Self::Numeric {
                plan: NumericTargetPlan::admit(target, output, function)?,
                loss,
                channels: output
                    .axis_size(DeepLearningAxisLabel::Channel)
                    .ok_or_else(|| fail(function, "network output has no channel axis"))?,
                function,
            }),
        }
    }

    pub(super) const fn loss(&self) -> LossKind {
        match self {
            Self::Numeric { loss, .. } => *loss,
            Self::Labels(_) => LossKind::Classification,
        }
    }

    pub(super) fn classes_value(&self) -> BuiltinResult<Option<Value>> {
        match self {
            Self::Labels(labels) => labels.classes_value().map(Some),
            Self::Numeric {
                loss: LossKind::Classification,
                channels,
                function,
                ..
            } => string_array(
                (1..=*channels).map(|class| class.to_string()).collect(),
                vec![1, *channels],
                function,
            )
            .map(Some),
            Self::Numeric { .. } => Ok(None),
        }
    }

    pub(super) async fn prepare(self, dtype: NumericDType) -> BuiltinResult<Tensor> {
        match self {
            Self::Numeric { plan, .. } => plan.prepare(dtype).await,
            Self::Labels(labels) => labels.one_hot(dtype),
        }
    }
}

fn infer_loss(
    layers: &[Value],
    has_labels: bool,
    function: &'static str,
) -> BuiltinResult<LossKind> {
    let terminal = layers.iter().rev().find_map(|layer| match layer {
        Value::Object(layer) => {
            K::from_identity(layer.class_name()).filter(|kind| kind.is_output())
        }
        _ => None,
    });
    match terminal {
        Some(K::ClassificationOutput) => Ok(LossKind::Classification),
        Some(K::RegressionOutput) => Ok(LossKind::Regression),
        Some(_) => unreachable!("other output layer kind"),
        None if has_labels => Ok(LossKind::Classification),
        None if !layers.is_empty() => Ok(LossKind::Regression),
        None => Err(fail(function, "network has no layers")),
    }
}

pub(super) fn prediction_domain(layers: &[Value]) -> PredictionDomain {
    let final_numerical = layers
        .iter()
        .rev()
        .filter_map(|layer| match layer {
            Value::Object(layer) => K::from_identity(layer.class_name()),
            _ => None,
        })
        .find(|kind| !kind.is_output());
    if final_numerical == Some(K::Softmax) {
        PredictionDomain::Probabilities
    } else {
        PredictionDomain::Logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_builtins::DeepLearningAxisLayout;
    use runmat_value::{ObjectInstance, StringArray};

    fn axes(format: &str, shape: &[usize]) -> AxisPlan {
        AxisPlan::new(
            DeepLearningAxisLayout::parse(format, shape.len()).unwrap(),
            shape,
        )
        .unwrap()
    }

    #[test]
    fn classification_labels_and_numeric_probabilities_have_distinct_admission() {
        let output = axes("CB", &[2, 3]);
        let labels = Value::StringArray(
            StringArray::new(vec!["b".into(), "a".into(), "b".into()], vec![3, 1]).unwrap(),
        );
        let admitted = TrainingTarget::admit(
            TrainingCaller::Trainnet,
            Some(LossKind::Classification),
            &[],
            &labels,
            &output,
        )
        .unwrap();
        assert!(matches!(admitted, TrainingTarget::Labels(_)));
        assert_eq!(admitted.loss(), LossKind::Classification);

        let numeric = Value::Tensor(
            Tensor::from_f32(vec![0.2, 0.8, 0.3, 0.7, 0.4, 0.6], vec![3, 2]).unwrap(),
        );
        let admitted = TrainingTarget::admit(
            TrainingCaller::Trainnet,
            Some(LossKind::Classification),
            &[],
            &numeric,
            &output,
        )
        .unwrap();
        assert!(matches!(admitted, TrainingTarget::Numeric { .. }));
    }

    #[test]
    fn terminal_output_identity_sets_loss_without_string_class_comparison() {
        let output = axes("CB", &[2, 2]);
        let target = Value::Tensor(Tensor::new(vec![0., 1., 1., 0.], vec![2, 2]).unwrap());
        let layers = vec![Value::Object(ObjectInstance::new(
            runmat_builtins::CLASSIFICATION_OUTPUT_LAYER_CLASS.owned(),
        ))];
        assert_eq!(
            TrainingTarget::admit(
                TrainingCaller::TrainNetwork,
                None,
                &layers,
                &target,
                &output,
            )
            .unwrap()
            .loss(),
            LossKind::Classification
        );
    }
}
