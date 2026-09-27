use super::*;
use crate::builtins::common::spec::{
    checked_reduction_elements, FusionReductionContext, FusionReductionContract,
    FusionReductionControls, FusionReductionCount, FusionReductionDeclaration,
    FusionReductionEmpty, FusionReductionOperation, FusionReductionPrepareFuture,
    FusionReductionTies, PreparedFusionReduction,
};
use runmat_types::NumericClass;

pub(super) const DECLARATION: FusionReductionDeclaration = FusionReductionDeclaration {
    revision: 2,
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
        let Some((input_class, shape)) = (FusionReductionContext {
            input,
            controls: &[],
            requested_outputs: 1,
        })
        .real_floating_input() else {
            return Ok(None);
        };
        let parsed = &self.0;
        let mut dimensions = match &parsed.axes {
            MeanAxes::Default => vec![default_dimension_from_shape(&shape)],
            MeanAxes::Dim(dim) => vec![*dim],
            MeanAxes::Vec(dims) => dims.clone(),
            MeanAxes::All => Vec::new(),
        };
        dimensions.sort_unstable();
        dimensions.dedup();
        // Multiple mean dimensions are sequential host reductions, not one
        // global sum/count (especially with omitnan). No false combined receipt.
        if matches!(parsed.axes, MeanAxes::Vec(_)) && dimensions.len() != 1 {
            return Ok(None);
        }
        let scalar = is_scalar_shape(&shape);
        let all = matches!(parsed.axes, MeanAxes::All);
        let (axes, mut output_shape, identity) = if all && !scalar {
            ((0..shape.len()).collect(), vec![1, 1], false)
        } else {
            let dim = if all { 1 } else { dimensions[0] };
            if scalar {
                (vec![0], vec![1, 1], true)
            } else if let Some(output) = reduction_shape(&shape, dim) {
                (vec![dim - 1], output, false)
            } else {
                (vec![], shape.clone(), true)
            }
        };
        let output_class = match parsed.output {
            OutputTemplate::Double => NumericClass::Double,
            OutputTemplate::Default | OutputTemplate::Native => input_class,
        };
        let Some(count) = checked_reduction_elements(&output_shape) else {
            return Ok(None);
        };
        let raw_double = !identity || scalar || input_class == NumericClass::Double;
        if count == 1 && (raw_double || output_class == NumericClass::Double) {
            output_shape = vec![1, 1];
        }
        Ok(Some(FusionReductionContract {
            operation: FusionReductionOperation::Mean,
            input_class,
            accumulator_class: NumericClass::Double,
            output_class,
            input_shape: shape,
            output_shape,
            axes,
            identity,
            nan_mode: parsed.nan_mode,
            count: if parsed.nan_mode == ReductionNaN::Omit {
                FusionReductionCount::NonNan
            } else {
                FusionReductionCount::SliceLength
            },
            empty: FusionReductionEmpty::NaN,
            ties: FusionReductionTies::NotApplicable,
        }))
    }
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
}
