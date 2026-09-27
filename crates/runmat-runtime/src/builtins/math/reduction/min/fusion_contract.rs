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
        let ParsedCall::Reduction(parsed) = parse_call(controls.values).await? else {
            return Ok(None);
        };
        if parsed.comparison == ComparisonMethod::Abs || parsed.linear_index {
            return Ok(None);
        }
        Ok(Some(
            Box::new(Prepared(parsed)) as Box<dyn PreparedFusionReduction>
        ))
    })
}

struct Prepared(ReductionArgs);
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
        // Empty extrema have a separate host shape path; it is not an identity
        // value suitable for the initial nonempty fused reduction contract.
        if checked_reduction_elements(&shape) == Some(0) {
            return Ok(None);
        }
        let resolved = resolve_reduction_dims(&shape, &parsed.selection)?;
        let mut output_shape = resolved.output_shape;
        let Some(count) = checked_reduction_elements(&output_shape) else {
            return Ok(None);
        };
        if count == 1 && input_class == NumericClass::Double {
            output_shape = vec![1, 1];
        }
        Ok(Some(FusionReductionContract {
            operation: FusionReductionOperation::Minimum,
            input_class,
            accumulator_class: input_class,
            output_class: input_class,
            input_shape: shape,
            output_shape,
            axes: resolved.reduced_dims,
            identity: false,
            nan_mode: parsed.nan_mode,
            count: FusionReductionCount::None,
            empty: FusionReductionEmpty::NaN,
            ties: FusionReductionTies::FirstWithNegativeZero,
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
    fn minimum_preserves_class_negative_zero_and_nan_policy() {
        let input = single(vec![0.0, -0.0], vec![2, 1]);
        let contract = receipt(&input, &[]).unwrap();
        assert_eq!(contract.output_class, NumericClass::Single);
        assert_eq!(contract.ties, FusionReductionTies::FirstWithNegativeZero);
        let Value::Tensor(result) =
            futures::executor::block_on(min_builtin(input.clone(), vec![])).unwrap()
        else {
            panic!("Single minimum");
        };
        assert!(result.materialize_f64()[0].is_sign_negative());
        let empty = Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
        let options = [empty.clone(), Value::String("omitnan".into())];
        let nan = single(vec![f32::NAN, f32::NAN], vec![2, 1]);
        assert_eq!(
            receipt(&nan, &options).unwrap().empty,
            FusionReductionEmpty::NaN
        );
        let Value::Tensor(result) =
            futures::executor::block_on(min_builtin(nan, options.to_vec())).unwrap()
        else {
            panic!("Single minimum");
        };
        assert!(result.materialize_f64()[0].is_nan());
        assert!(receipt(&input, &[Value::Num(2.0)]).is_none()); // pairwise, not axis
        assert!(receipt(&empty, &[]).is_none());
    }
}
