use super::*;
use crate::builtins::common::spec::{
    checked_reduction_elements, FusionReductionContext, FusionReductionContract,
    FusionReductionControls, FusionReductionCount, FusionReductionDeclaration,
    FusionReductionEmpty, FusionReductionOperation, FusionReductionPrepareFuture,
    FusionReductionTies, PreparedFusionReduction,
};
use runmat_types::NumericClass;

pub(super) const DECLARATION: FusionReductionDeclaration = FusionReductionDeclaration {
    revision: 3,
    prepare,
};
fn prepare(controls: FusionReductionControls<'_>) -> FusionReductionPrepareFuture<'_> {
    Box::pin(async move {
        if !controls.supported() {
            return Ok(None);
        }
        let parsed = parse_arguments(controls.values).await?;
        if parsed.weights.is_some() {
            return Ok(None);
        }
        Ok(Some(
            Box::new(Prepared(parsed)) as Box<dyn PreparedFusionReduction>
        ))
    })
}

struct Prepared(ParsedArguments);
impl PreparedFusionReduction for Prepared {
    fn resolve(
        &self,
        input: &runmat_types::ValueFact,
    ) -> BuiltinResult<Option<FusionReductionContract>> {
        let Some(mut stages) = self.resolve_stages(input)? else {
            return Ok(None);
        };
        if stages.len() != 1 {
            return Ok(None);
        }
        Ok(stages.pop())
    }

    fn resolve_stages(
        &self,
        input: &runmat_types::ValueFact,
    ) -> BuiltinResult<Option<Vec<FusionReductionContract>>> {
        let Some((input_class, shape)) = (FusionReductionContext {
            input,
            controls: &[],
            requested_outputs: 1,
        })
        .real_floating_input() else {
            return Ok(None);
        };
        let parsed = &self.0;
        let dimensions = match &parsed.axes {
            MeanAxes::Default => vec![Some(default_dimension_from_shape(&shape))],
            MeanAxes::Dim(dim) => vec![Some(*dim)],
            MeanAxes::Vec(dims) => {
                let mut dims = dims.clone();
                dims.sort_unstable();
                dims.dedup();
                dims.into_iter().map(Some).collect()
            }
            MeanAxes::All => vec![None],
        };
        let mut current_class = input_class;
        let mut current_shape = shape;
        let mut stages = Vec::with_capacity(dimensions.len());
        for (index, dimension) in dimensions.iter().enumerate() {
            // Vecdim is one public call: the host implementation keeps Double
            // scratch values between axes and applies the output template
            // only after its final stage. Nested calls publish separately.
            let final_output_class = (index + 1 == dimensions.len()).then_some(match parsed.output {
                OutputTemplate::Double => NumericClass::Double,
                OutputTemplate::Default | OutputTemplate::Native => input_class,
            });
            let Some(stage) = stage_contract(
                current_class,
                current_shape,
                *dimension,
                final_output_class,
                parsed.nan_mode,
            ) else {
                return Ok(None);
            };
            current_class = stage.output_class;
            current_shape = stage.output_shape.clone();
            stages.push(stage);
        }
        Ok(Some(stages))
    }
}

fn stage_contract(
    input_class: NumericClass,
    shape: Vec<usize>,
    dimension: Option<usize>,
    final_output_class: Option<NumericClass>,
    nan_mode: ReductionNaN,
) -> Option<FusionReductionContract> {
    let scalar = is_scalar_shape(&shape);
    let (axes, mut output_shape, identity) = if dimension.is_none() && !scalar {
        ((0..shape.len()).collect(), vec![1, 1], false)
    } else {
        let dim = dimension.unwrap_or(1);
        if scalar {
            (vec![0], vec![1, 1], true)
        } else if let Some(output) = reduction_shape(&shape, dim) {
            (vec![dim - 1], output, false)
        } else {
            (vec![], shape.clone(), true)
        }
    };
    let count = checked_reduction_elements(&output_shape)?;
    let raw_double = !identity || scalar || input_class == NumericClass::Double;
    // An out-of-rank axis returns the input tensor unchanged; it does not
    // allocate a Double scratch tensor. A scalar identity does materialize
    // Double in the host reduction owner. The public template is applied
    // once after the final axis, including an all-identity vecdim call.
    let output_class = final_output_class.unwrap_or(if raw_double {
        NumericClass::Double
    } else {
        input_class
    });
    // mean_host converts any one-element Double scratch tensor into Num,
    // regardless of its rank. Final Single publication then reconstructs
    // a [1,1] tensor from that scalar; a final Double result remains Num.
    // Intermediate stages retain their tensor geometry until the next axis.
    if count == 1 && final_output_class.is_some() {
        output_shape = vec![1, 1];
    }
    Some(FusionReductionContract {
        operation: FusionReductionOperation::Mean,
        input_class,
        accumulator_class: NumericClass::Double,
        output_class,
        input_shape: shape,
        output_shape,
        axes,
        identity,
        nan_mode,
        count: if nan_mode == ReductionNaN::Omit {
            FusionReductionCount::NonNan
        } else {
            FusionReductionCount::SliceLength
        },
        empty: FusionReductionEmpty::NaN,
        ties: FusionReductionTies::NotApplicable,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_fact::value_fact;

    fn receipt(value: &Value, controls: &[Value]) -> Option<FusionReductionContract> {
        let fact = value_fact(value);
        futures::executor::block_on(DECLARATION.resolve(FusionReductionContext {
            input: &fact,
            controls,
            requested_outputs: 1,
        }))
        .unwrap()
    }
    fn stages(value: &Value, controls: &[Value]) -> Option<Vec<FusionReductionContract>> {
        let fact = value_fact(value);
        futures::executor::block_on(DECLARATION.resolve_stages(FusionReductionContext {
            input: &fact,
            controls,
            requested_outputs: 1,
        }))
        .unwrap()
    }
    fn single(values: Vec<f32>, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::from_f32(values, shape).unwrap())
    }

    #[test]
    fn mean_uses_double_sum_and_actual_valid_count_before_single_restoration() {
        let input = single(vec![16_777_216.0, 1.0, 1.0], vec![3, 1]);
        let contract = receipt(&input, &[]).unwrap();
        assert_eq!(contract.accumulator_class, NumericClass::Double);
        assert_eq!(contract.output_class, NumericClass::Single);
        assert_eq!(contract.count, FusionReductionCount::SliceLength);
        let actual = futures::executor::block_on(mean_builtin(input.clone(), vec![])).unwrap();
        let Value::Tensor(actual) = actual else {
            panic!("Single mean must retain typed carrier");
        };
        assert_eq!(actual.numeric_dtype(), NumericDType::F32);
        assert_eq!(actual.materialize_f64(), vec![5_592_406.0]);
        let omitted = receipt(&input, &[Value::String("omitnan".into())]).unwrap();
        assert_eq!(omitted.count, FusionReductionCount::NonNan);
        assert_eq!(omitted.empty, FusionReductionEmpty::NaN);
        let double = receipt(&input, &[Value::String("double".into())]).unwrap();
        assert_eq!(double.output_class, NumericClass::Double);
        let scalar = single(vec![-0.0], vec![1, 1]);
        assert!(receipt(&scalar, &[]).unwrap().identity);
        let Value::Tensor(result) =
            futures::executor::block_on(mean_builtin(scalar, vec![])).unwrap()
        else {
            panic!("Single mean");
        };
        assert!(result.materialize_f64()[0].is_sign_negative());
        let dimensions = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        assert!(receipt(&input, &[dimensions]).is_none());
    }

    #[test]
    fn vecdim_receipt_keeps_double_scratch_until_single_publication() {
        let input = single(vec![1.0; 2 * 3 * 4], vec![2, 3, 4]);
        let axes = Value::Tensor(Tensor::new(vec![3.0, 2.0], vec![1, 2]).unwrap());
        let controls = [axes.clone(), Value::String("native".into())];
        let sequence = stages(&input, &controls).expect("ordered mean stages");
        assert_eq!(sequence.len(), 2);
        assert_eq!(sequence[0].axes, vec![1]);
        assert_eq!(sequence[0].input_class, NumericClass::Single);
        assert_eq!(sequence[0].accumulator_class, NumericClass::Double);
        assert_eq!(sequence[0].output_class, NumericClass::Double);
        assert_eq!(sequence[0].output_shape, sequence[1].input_shape);
        assert_eq!(sequence[1].axes, vec![2]);
        assert_eq!(sequence[1].input_class, NumericClass::Double);
        assert_eq!(sequence[1].output_class, NumericClass::Single);
        assert!(receipt(&input, &controls).is_none());

        let double = stages(&input, &[axes, Value::String("double".into())]).unwrap();
        assert_eq!(double[0].output_class, NumericClass::Double);
        assert_eq!(double[1].output_class, NumericClass::Double);
        let nested_first = stages(&input, &[Value::Num(2.0)]).unwrap();
        assert_eq!(nested_first.len(), 1);
        assert_eq!(nested_first[0].output_class, NumericClass::Single);
    }

    #[test]
    fn vecdim_out_of_rank_identity_preserves_intermediate_carrier() {
        let input = single(vec![1.0; 6], vec![2, 3]);
        let axes =
            |first, second| Value::Tensor(Tensor::new(vec![first, second], vec![1, 2]).unwrap());
        let identities = stages(&input, &[axes(4.0, 5.0)]).unwrap();
        assert_eq!(identities.len(), 2);
        assert!(identities.iter().all(|stage| stage.identity));
        assert_eq!(identities[0].output_class, NumericClass::Single);
        assert_eq!(identities[1].input_class, NumericClass::Single);
        assert_eq!(identities[1].output_class, NumericClass::Single);

        let mixed = stages(&input, &[axes(1.0, 4.0)]).unwrap();
        assert!(!mixed[0].identity);
        assert_eq!(mixed[0].output_class, NumericClass::Double);
        assert!(mixed[1].identity);
        assert_eq!(mixed[1].input_class, NumericClass::Double);
        assert_eq!(mixed[1].output_class, NumericClass::Single);
    }

    #[test]
    fn vecdim_singleton_shape_tracks_tensor_until_publication() {
        let input = single(vec![3.0, 5.0], vec![1, 1, 2]);
        let axes = |dims: Vec<f64>| Value::Tensor(Tensor::new(dims, vec![1, 2]).unwrap());
        let one = stages(&input, &[Value::Num(3.0)]).unwrap();
        assert_eq!(one[0].output_shape, vec![1, 1]);
        assert_eq!(one[0].output_class, NumericClass::Single);
        let Value::Tensor(one_actual) =
            futures::executor::block_on(mean_builtin(input.clone(), vec![Value::Num(3.0)]))
                .unwrap()
        else {
            panic!("Single one-element mean is republished as a tensor");
        };
        assert_eq!(one_actual.shape, one[0].output_shape);
        let pair = stages(&input, &[axes(vec![3.0, 4.0])]).unwrap();
        assert_eq!(pair[0].output_shape, vec![1, 1, 1]);
        assert_eq!(pair[0].output_class, NumericClass::Double);
        assert_eq!(pair[1].input_shape, vec![1, 1, 1]);
        assert_eq!(pair[1].output_shape, vec![1, 1]);
        assert_eq!(pair[1].output_class, NumericClass::Single);
        let Value::Tensor(pair_actual) =
            futures::executor::block_on(mean_builtin(input.clone(), vec![axes(vec![3.0, 4.0])]))
                .unwrap()
        else {
            panic!("Single vecdim publication remains a tensor");
        };
        assert_eq!(pair_actual.shape, pair[1].output_shape);
        let double = stages(
            &input,
            &[axes(vec![3.0, 4.0]), Value::String("double".into())],
        )
        .unwrap();
        assert_eq!(double[1].output_shape, vec![1, 1]);
        let double_actual = futures::executor::block_on(mean_builtin(
            input,
            vec![axes(vec![3.0, 4.0]), Value::String("double".into())],
        ))
        .unwrap();
        assert!(matches!(double_actual, Value::Num(4.0)));
    }
}
