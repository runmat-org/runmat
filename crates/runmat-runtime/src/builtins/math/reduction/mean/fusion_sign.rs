//! A sign theorem for the existing, owner-resolved ordinary MEAN stages.
//! Parsing controls and determining stage order remain in fusion_contract.

use crate::builtins::common::spec::{
    FusionReductionContract, FusionReductionCount, FusionReductionEmpty, FusionReductionOperation,
};
use runmat_types::{NumericClass, NumericDomain, StorageFact, ValueFact};

pub(super) fn preserves_nonnegative_or_nan(
    input: &ValueFact,
    output: &ValueFact,
    stages: &[FusionReductionContract],
) -> bool {
    let Some(input_numeric) = input.numeric() else {
        return false;
    };
    if input_numeric.domain != NumericDomain::Real
        || !matches!(
            input_numeric.class,
            NumericClass::Single | NumericClass::Double
        )
    {
        return false;
    }
    let Some(mut shape) = input
        .shape
        .known_dims()
        .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
    else {
        return false;
    };
    let mut class = input_numeric.class;
    if stages.is_empty() {
        return false;
    }
    for stage in stages {
        let mut reduced_shape = shape.clone();
        for &axis in &stage.axes {
            let Some(extent) = reduced_shape.get_mut(axis) else {
                return false;
            };
            *extent = 1;
        }
        let scalarized = reduced_shape
            .iter()
            .try_fold(1usize, |total, &extent| total.checked_mul(extent))
            == Some(1)
            && stage.output_shape == [1, 1];
        let counted = match stage.nan_mode {
            runmat_types::ReductionNaN::Include => stage.count == FusionReductionCount::SliceLength,
            runmat_types::ReductionNaN::Omit => stage.count == FusionReductionCount::NonNan,
        };
        if stage.operation != FusionReductionOperation::Mean
            || stage.input_class != class
            || stage.accumulator_class != NumericClass::Double
            || !matches!(
                stage.output_class,
                NumericClass::Single | NumericClass::Double
            )
            || stage.input_shape != shape
            || stage.empty != FusionReductionEmpty::NaN
            || !counted
            || stage.axes.is_empty()
            || stage
                .axes
                .iter()
                .any(|&axis| shape.get(axis).is_none_or(|&extent| extent == 0))
            || stage.output_shape.contains(&0)
            || !(stage.output_shape == reduced_shape || scalarized)
        {
            return false;
        }
        class = stage.output_class;
        shape.clone_from(&stage.output_shape);
    }
    if output
        .shape
        .known_dims()
        .and_then(|dims| dims.into_iter().collect::<Option<Vec<_>>>())
        != Some(shape)
    {
        return false;
    }
    output
        .numeric()
        .is_none_or(|numeric| numeric.domain == NumericDomain::Real && numeric.class == class)
}

/// The final public MEAN publication owns this carrier, independently of
/// the sign theorem. A one-element Double mean is a Num; Single publication
/// reconstructs a typed tensor even when its public shape is [1, 1].
pub(super) fn ordinary_output_storage(
    input: &ValueFact,
    output: &ValueFact,
    stages: &[FusionReductionContract],
) -> Option<StorageFact> {
    if !preserves_nonnegative_or_nan(input, output, stages) {
        return None;
    }
    let last = stages.last()?;
    let elements = last
        .output_shape
        .iter()
        .try_fold(1_usize, |total, extent| total.checked_mul(*extent))?;
    Some(
        if last.output_class == NumericClass::Double && elements == 1 {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::spec::FusionReductionContext;
    use crate::value_fact::value_fact;
    use runmat_value::{Tensor, Value};

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn ordered_vecdim_requires_real_double_scratch_and_nonempty_extents() {
        let input = value_fact(&Value::Tensor(
            Tensor::from_f32(vec![0.0; 40], vec![2, 4, 5]).unwrap(),
        ));
        let axes = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap());
        let controls = [axes, Value::String("native".into())];
        let stages = futures::executor::block_on(
            super::super::fusion_contract::DECLARATION.resolve_stages(FusionReductionContext {
                input: &input,
                controls: &controls,
                requested_outputs: 1,
            }),
        )
        .unwrap()
        .unwrap();
        assert_eq!(stages.len(), 2);
        let output = value_fact(&Value::Tensor(
            Tensor::from_f32(vec![0.0; 2], vec![2, 1, 1]).unwrap(),
        ));
        assert!(preserves_nonnegative_or_nan(&input, &output, &stages));
        let mut wrong = stages.clone();
        wrong[1].axes = vec![0];
        assert!(!preserves_nonnegative_or_nan(&input, &output, &wrong));
        let mut wrong = stages.clone();
        wrong[0].accumulator_class = NumericClass::Single;
        assert!(!preserves_nonnegative_or_nan(&input, &output, &wrong));
        let mut wrong = stages.clone();
        wrong[1].count = FusionReductionCount::None;
        assert!(!preserves_nonnegative_or_nan(&input, &output, &wrong));
        let wrong_class = value_fact(&Value::Tensor(
            Tensor::new(vec![0.0; 2], vec![2, 1, 1]).unwrap(),
        ));
        assert!(!preserves_nonnegative_or_nan(&input, &wrong_class, &stages));
        let omitted = [
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
            Value::String("omitnan".into()),
            Value::String("native".into()),
        ];
        let omit_stages = futures::executor::block_on(
            super::super::fusion_contract::DECLARATION.resolve_stages(FusionReductionContext {
                input: &input,
                controls: &omitted,
                requested_outputs: 1,
            }),
        )
        .unwrap()
        .unwrap();
        assert!(preserves_nonnegative_or_nan(&input, &output, &omit_stages));
        let owner = &super::super::fusion_contract::DECLARATION;
        assert_eq!(
            futures::executor::block_on(owner.prove_nonnegative_or_nan(
                FusionReductionContext {
                    input: &input,
                    controls: &controls,
                    requested_outputs: 1,
                },
                &output,
            ))
            .unwrap(),
            Some(
                crate::builtins::common::spec::FusionReductionSignProjection {
                    class: NumericClass::Single,
                    storage: StorageFact::Dense,
                }
            )
        );
        let double_output = value_fact(&Value::Num(0.0));
        let double_controls = [Value::String("all".into()), Value::String("double".into())];
        assert_eq!(
            futures::executor::block_on(owner.prove_nonnegative_or_nan(
                FusionReductionContext {
                    input: &input,
                    controls: &double_controls,
                    requested_outputs: 1,
                },
                &double_output,
            ))
            .unwrap(),
            Some(
                crate::builtins::common::spec::FusionReductionSignProjection {
                    class: NumericClass::Double,
                    storage: StorageFact::Scalar,
                }
            )
        );
        let singleton_input = value_fact(&Value::Tensor(
            Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap(),
        ));
        let singleton_output = value_fact(&Value::Tensor(
            Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap(),
        ));
        assert_eq!(
            futures::executor::block_on(owner.prove_nonnegative_or_nan(
                FusionReductionContext {
                    input: &singleton_input,
                    controls: &[],
                    requested_outputs: 1,
                },
                &singleton_output,
            ))
            .unwrap(),
            Some(
                crate::builtins::common::spec::FusionReductionSignProjection {
                    class: NumericClass::Single,
                    storage: StorageFact::Dense,
                }
            )
        );
        let out_of_range = [Value::Num(4.0), Value::String("native".into())];
        assert!(futures::executor::block_on(owner.prove_nonnegative_or_nan(
            FusionReductionContext {
                input: &input,
                controls: &out_of_range,
                requested_outputs: 1,
            },
            &output,
        ))
        .unwrap()
        .is_none());
        let weighted = [
            Value::String("weights".into()),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
        ];
        assert!(futures::executor::block_on(owner.prove_nonnegative_or_nan(
            FusionReductionContext {
                input: &input,
                controls: &weighted,
                requested_outputs: 1,
            },
            &output,
        ))
        .unwrap()
        .is_none());
        let empty = value_fact(&Value::Tensor(
            Tensor::from_f32(vec![], vec![2, 0, 5]).unwrap(),
        ));
        let empty_stages = futures::executor::block_on(
            super::super::fusion_contract::DECLARATION.resolve_stages(FusionReductionContext {
                input: &empty,
                controls: &controls,
                requested_outputs: 1,
            }),
        )
        .unwrap()
        .unwrap();
        assert!(!preserves_nonnegative_or_nan(
            &empty,
            &output,
            &empty_stages
        ));
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn ordinary_mean_edges_remain_nonnegative_or_nan_without_gpu_claims() {
        let scalar = |value: Value| match value {
            Value::Num(value) => value,
            Value::Tensor(tensor) => tensor.materialize_f64()[0],
            other => panic!("unexpected ordinary mean output: {other:?}"),
        };
        let single = Value::Tensor(
            Tensor::from_f32(vec![f32::NAN, -0.0, f32::INFINITY], vec![1, 3]).unwrap(),
        );
        let included = futures::executor::block_on(super::super::mean_builtin(
            single.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        assert!(scalar(included).is_nan());
        let omitted = futures::executor::block_on(super::super::mean_builtin(
            single,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "omitnan".into(),
            )]),
        ))
        .unwrap();
        assert_eq!(scalar(omitted), f64::INFINITY);
        let signed_zero = futures::executor::block_on(super::super::mean_builtin(
            Value::Tensor(Tensor::from_f32(vec![-0.0, -0.0], vec![1, 2]).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        assert_eq!(scalar(signed_zero), 0.0);
        let overflow = futures::executor::block_on(super::super::mean_builtin(
            Value::Tensor(Tensor::new(vec![f64::MAX, f64::MAX], vec![1, 2]).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        assert_eq!(scalar(overflow), f64::INFINITY);
    }
}
