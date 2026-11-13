use super::*;
use crate::builtins::common::spec::{
    checked_reduction_elements, FusionReductionContext, FusionReductionContract,
    FusionReductionControls, FusionReductionCount, FusionReductionDeclaration,
    FusionReductionEmpty, FusionReductionOperation, FusionReductionPrepareFuture,
    FusionReductionTies, PreparedFusionReduction,
};
use crate::builtins::common::spec::{
    reduction_control, reduction_layout, reduction_sum_bytes, FusionReductionPreparationQuery,
    FusionReductionPreparationQuote, ReductionAxisPlan, ReductionDimensions,
};
use runmat_types::NumericClass;

pub(super) const DECLARATION: FusionReductionDeclaration = FusionReductionDeclaration {
    revision: 3,
    operation: FusionReductionOperation::Sum,
    prepare,
    resources,
};
fn prepare(controls: FusionReductionControls<'_>) -> FusionReductionPrepareFuture<'_> {
    Box::pin(prepare_impl(controls))
}
async fn prepare_impl(
    controls: FusionReductionControls<'_>,
) -> BuiltinResult<Option<Box<dyn PreparedFusionReduction>>> {
    if let Some(plan) = lexical_plan(&controls) {
        return Ok(Some(Box::new(Prepared(plan.materialize()?))));
    }
    if !controls.supported() {
        return Ok(None);
    }
    let parsed = parse_arguments(crate::call::FunctionArgumentView::from_slice(
        controls.values,
    ))
    .await?;
    Ok(Some(
        Box::new(Prepared(parsed)) as Box<dyn PreparedFusionReduction>
    ))
}

struct Plan<'a> {
    axes: ReductionAxisPlan<'a>,
    nan_mode: ReductionNaN,
    output: OutputTemplate,
}
fn lexical_plan<'a>(controls: &FusionReductionControls<'a>) -> Option<Plan<'a>> {
    if !controls.supported() {
        return None;
    }
    let mut plan = Plan {
        axes: ReductionAxisPlan::Auto,
        nan_mode: ReductionNaN::Include,
        output: OutputTemplate::Default,
    };
    let mut selection_set = false;
    let mut output_set = false;
    for value in controls.values {
        if let Some(control) = reduction_control(value, SumControl::from_public_option) {
            match control {
                SumControl::Missing(mode) => plan.nan_mode = mode,
                SumControl::All => {
                    if selection_set && !plan.axes.is_auto() {
                        return None;
                    }
                    plan.axes = ReductionAxisPlan::All;
                    selection_set = true;
                }
                SumControl::Output(output) => {
                    if output_set {
                        return None;
                    }
                    plan.output = output;
                    output_set = true;
                }
            }
        } else {
            if selection_set && !plan.axes.is_auto() {
                return None;
            }
            let dimensions = ReductionDimensions::parse(value, false)?;
            if let Value::Tensor(tensor) = value {
                if !is_vector_shape(tensor.shape()) {
                    return None;
                }
            }
            plan.axes = dimensions.axis_plan();
            selection_set = true;
        }
    }
    Some(plan)
}
impl Plan<'_> {
    fn materialize(self) -> BuiltinResult<ParsedArguments> {
        let selection = match self.axes {
            ReductionAxisPlan::Auto => DimSelection::Auto,
            ReductionAxisPlan::Dim(dimension) => DimSelection::Dim(dimension),
            ReductionAxisPlan::All => DimSelection::All,
            ReductionAxisPlan::Dimensions(dimensions) => match dimensions.len {
                0 => DimSelection::Auto,
                _ => DimSelection::Vec(dimensions.materialize()?),
            },
        };
        Ok(ParsedArguments {
            invocation: runmat_types::SumInvocation::Sum,
            selection,
            nan_mode: self.nan_mode,
            output: self.output,
        })
    }
}
fn resources(
    query: FusionReductionPreparationQuery<'_>,
) -> Option<FusionReductionPreparationQuote> {
    let rank = query.input_rank_upper.max(2);
    let (retained_axes, axes) = match lexical_plan(&query.controls) {
        Some(plan) => (plan.axes.retained_len(), plan.axes.resolved_capacity(rank)),
        None if query.controls.supported()
            && crate::builtins::common::spec::reduction_rejected_scalar_axis(
                query.controls.values,
            ) =>
        {
            (0, 1)
        }
        None => return None,
    };
    Some(FusionReductionPreparationQuote {
        prepare_scratch_bytes: u64::try_from(std::mem::size_of_val(&prepare_impl(query.controls)))
            .ok()?,
        prepared_retained_bytes: reduction_sum_bytes(&[
            reduction_layout::<Prepared>(1)?,
            reduction_layout::<usize>(retained_axes)?,
        ])?,
        // One-based axis worklist and the old output shape overlapping scalar
        // publication. Final input/output/axis backing is retained below.
        resolve_scratch_bytes: reduction_sum_bytes(&[
            reduction_layout::<usize>(axes)?,
            reduction_layout::<usize>(rank)?,
        ])?,
        contract_retained_bytes: reduction_sum_bytes(&[
            reduction_layout::<FusionReductionContract>(1)?,
            reduction_layout::<usize>(rank)?,
            reduction_layout::<usize>(rank)?,
            reduction_layout::<usize>(axes)?,
        ])?,
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
        .real_floating_input()?
        else {
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
        let output_class = parsed.output.resolve_numeric_class(input_class);
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

    async fn sum_builtin(
        observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
        value: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        super::super::sum_builtin(
            None,
            observation,
            value,
            crate::builtins::common::test_support::builtin_arguments(rest),
        )
        .await
        .map(|owned| owned.value().clone())
    }

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
        // Differential against the original operation parser, not a second
        // expected-options table. No extra numerical execution is required.
        let fact = value_fact(&single(vec![1.0; 6], vec![2, 3]));
        for controls in [
            vec![],
            vec![Value::String("AlL".into())],
            vec![Value::CharArray(runmat_value::CharArray::new_row("ALL"))],
            vec![
                Value::Tensor(Tensor::new(vec![], vec![1, 0]).unwrap()),
                Value::Num(2.0),
            ],
            vec![
                Value::Tensor(Tensor::new(vec![2.0, 1.0, 2.0], vec![1, 3]).unwrap()),
                Value::String("NaTiVe".into()),
            ],
            vec![
                Value::Num(1.0 + 5.0e-7),
                Value::String("OMITMISSING".into()),
            ],
        ] {
            let request = FusionReductionControls {
                values: &controls,
                requested_outputs: 1,
            };
            let plan = lexical_plan(&request).expect("existing valid host controls are quoted");
            assert!(resources(FusionReductionPreparationQuery {
                controls: request,
                input_rank_upper: 2
            })
            .is_some());
            let original = futures::executor::block_on(parse_arguments(
                crate::call::FunctionArgumentView::from_slice(&controls),
            ))
            .unwrap();
            assert_eq!(
                Prepared(plan.materialize().unwrap())
                    .resolve(&fact)
                    .unwrap(),
                Prepared(original).resolve(&fact).unwrap()
            );
        }
        let nonvector = [Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap())];
        assert!(lexical_plan(&FusionReductionControls {
            values: &nonvector,
            requested_outputs: 1,
        })
        .is_none());
        assert!(futures::executor::block_on(parse_arguments(
            crate::call::FunctionArgumentView::from_slice(&nonvector),
        ))
        .is_err());
        let duplicate = [
            Value::String("default".into()),
            Value::String("DEFAULT".into()),
        ];
        assert!(resources(FusionReductionPreparationQuery {
            controls: FusionReductionControls {
                values: &duplicate,
                requested_outputs: 1
            },
            input_rank_upper: 2
        })
        .is_none());
        let original = futures::executor::block_on(parse_arguments(
            crate::call::FunctionArgumentView::from_slice(&duplicate),
        ))
        .err()
        .unwrap();
        let fallback = futures::executor::block_on(prepare(FusionReductionControls {
            values: &duplicate,
            requested_outputs: 1,
        }))
        .err()
        .unwrap();
        assert_eq!(original.identifier, fallback.identifier);
        assert_eq!(original.to_string(), fallback.to_string());
        let input = single(vec![16_777_216.0, 1.0], vec![2, 1]);
        let base = receipt(&input, &[]).unwrap();
        assert_eq!(base.accumulator_class, NumericClass::Double);
        assert_eq!(base.output_class, NumericClass::Single);
        assert_eq!(base.axes, vec![0]);
        assert_eq!(base.output_shape, vec![1, 1]);
        assert_eq!(
            futures::executor::block_on(sum_builtin(None, input.clone(), vec![])).unwrap(),
            single(vec![16_777_216.0], vec![1, 1])
        );
        let double = receipt(&input, &[Value::String("double".into())]).unwrap();
        assert_eq!(double.output_class, NumericClass::Double);
        assert_eq!(
            futures::executor::block_on(sum_builtin(
                None,
                input.clone(),
                vec![Value::String("double".into())]
            ))
            .unwrap(),
            Value::Num(16_777_217.0)
        );
        let native = receipt(&input, &[Value::String("native".into())]).unwrap();
        assert_eq!(native.output_class, NumericClass::Single);
        let noop = receipt(&input, &[Value::Num(3.0)]).unwrap();
        assert!(noop.identity);
        assert_eq!(noop.output_class, NumericClass::Single);
        assert_eq!(noop.output_shape, vec![2, 1]);
        let double_noop =
            receipt(&input, &[Value::Num(3.0), Value::String("double".into())]).unwrap();
        assert!(double_noop.identity);
        assert_eq!(double_noop.output_class, NumericClass::Double);
        let nan = single(vec![f32::NAN], vec![1, 1]);
        let omitted = receipt(&nan, &[Value::String("omitnan".into())]).unwrap();
        assert_eq!(omitted.empty, FusionReductionEmpty::PositiveZero);
        assert_eq!(omitted.nan_mode, ReductionNaN::Omit);
        assert_eq!(
            futures::executor::block_on(sum_builtin(
                None,
                nan,
                vec![Value::String("omitnan".into())]
            ))
            .unwrap(),
            single(vec![0.0], vec![1, 1])
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
