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
        let resolved = resolve_dims(&shape, &parsed.selection)?;
        let identity = resolved.dims_in_bounds.is_empty();
        let raw_class = if identity {
            input_class
        } else {
            NumericClass::Double
        };
        let output_class = match parsed.output {
            OutputTemplate::Double => raw_class, // Existing host no-op Single remains Single.
            OutputTemplate::Native => input_class,
        };
        let mut output_shape = if identity {
            shape.clone()
        } else {
            normalize_scalar_shape(&shape)
        };
        if !identity {
            for &axis in &resolved.dims_in_bounds {
                output_shape[axis] = 1;
            }
        }
        let Some(count) = checked_reduction_elements(&output_shape) else {
            return Ok(None);
        };
        // Host raw Double scalar collapses before the final native conversion.
        if count == 1 && raw_class == NumericClass::Double {
            output_shape = vec![1, 1];
        }
        Ok(Some(FusionReductionContract {
            operation: FusionReductionOperation::Sum,
            input_class,
            accumulator_class: NumericClass::Double,
            output_class,
            input_shape: shape,
            output_shape,
            axes: resolved.dims_in_bounds,
            identity,
            nan_mode: parsed.nan_mode,
            count: FusionReductionCount::None,
            empty: FusionReductionEmpty::PositiveZero,
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
    fn host_classes_and_actual_owner_parser_define_sum_receipt() {
        let input = single(vec![16_777_216.0, 1.0], vec![2, 1]);
        let base = receipt(&input, &[]).unwrap();
        assert_eq!(base.accumulator_class, NumericClass::Double);
        assert_eq!(base.output_class, NumericClass::Double);
        assert_eq!(base.axes, vec![0]);
        assert_eq!(base.output_shape, vec![1, 1]);
        assert_eq!(
            futures::executor::block_on(sum_builtin(input.clone(), vec![])).unwrap(),
            Value::Num(16_777_217.0)
        );
        let native = receipt(&input, &[Value::String("native".into())]).unwrap();
        assert_eq!(native.output_class, NumericClass::Single);
        let noop = receipt(&input, &[Value::Num(3.0)]).unwrap();
        assert!(noop.identity);
        assert_eq!(noop.output_class, NumericClass::Single);
        assert_eq!(noop.output_shape, vec![2, 1]);
        let nan = single(vec![f32::NAN], vec![1, 1]);
        let omitted = receipt(&nan, &[Value::String("omitnan".into())]).unwrap();
        assert_eq!(omitted.empty, FusionReductionEmpty::PositiveZero);
        assert_eq!(omitted.nan_mode, ReductionNaN::Omit);
        assert_eq!(
            futures::executor::block_on(sum_builtin(nan, vec![Value::String("omitnan".into())]))
                .unwrap(),
            Value::Num(0.0)
        );
        let fact = value_fact(&input);
        assert!(
            futures::executor::block_on(DECLARATION.resolve(FusionReductionContext {
                input: &fact,
                controls: &[Value::Num(0.5)],
                requested_outputs: 1
            }))
            .is_err()
        );
    }
}
