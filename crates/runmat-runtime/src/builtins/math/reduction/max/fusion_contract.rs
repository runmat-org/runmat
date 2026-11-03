use super::*;
use crate::builtins::common::shape::{is_scalar_shape, normalize_scalar_shape};
use crate::builtins::common::spec::{
    checked_reduction_elements, FusionReductionContext, FusionReductionContract,
    FusionReductionControls, FusionReductionCount, FusionReductionDeclaration,
    FusionReductionEmpty, FusionReductionOperation, FusionReductionPrepareFuture,
    FusionReductionTies, PreparedFusionReduction,
};
use crate::builtins::common::spec::{
    reduction_control, reduction_layout, reduction_sum_bytes, reduction_vec,
    FusionReductionPreparationQuery, FusionReductionPreparationQuote, ReductionAxisPlan,
    ReductionDimensions,
};
use runmat_types::NumericClass;

pub(super) const DECLARATION: FusionReductionDeclaration = FusionReductionDeclaration {
    revision: 2,
    operation: FusionReductionOperation::Maximum,
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
    let ParsedCall::Reduction(parsed) = parse_call(crate::call::FunctionArgumentView::from_slice(
        controls.values,
    ))
    .await?
    else {
        return Ok(None);
    };
    if parsed.comparison == ComparisonMethod::Abs || parsed.linear_index {
        return Ok(None);
    }
    Ok(Some(
        Box::new(Prepared(parsed)) as Box<dyn PreparedFusionReduction>
    ))
}

struct Plan<'a> {
    axes: ReductionAxisPlan<'a>,
    nan_mode: ReductionNaN,
    comparison: ComparisonMethod,
}
fn lexical_plan<'a>(controls: &FusionReductionControls<'a>) -> Option<Plan<'a>> {
    if !controls.supported() {
        return None;
    }
    let mut plan = Plan {
        axes: ReductionAxisPlan::Auto,
        nan_mode: ReductionNaN::Include,
        comparison: ComparisonMethod::Auto,
    };
    let mut index = 0;
    if let Some(first) = controls.values.first() {
        if !is_empty_placeholder(first) {
            return None;
        }
        index = 1;
    }
    let mut selection_set = false;
    while index < controls.values.len() {
        let value = &controls.values[index];
        if let Some(control) = reduction_control(value, MaxControl::from_public_option) {
            match control {
                MaxControl::Missing(mode) => plan.nan_mode = mode,
                MaxControl::All => {
                    if selection_set {
                        return None;
                    }
                    plan.axes = ReductionAxisPlan::All;
                    selection_set = true;
                }
                MaxControl::Linear => return None,
                MaxControl::ComparisonMethod => {
                    index += 1;
                    plan.comparison = reduction_control(
                        controls.values.get(index)?,
                        ComparisonMethod::from_public_option,
                    )?;
                }
            }
        } else {
            if selection_set {
                return None;
            }
            let dimensions = ReductionDimensions::parse(value, false)?;
            plan.axes = dimensions.axis_plan();
            selection_set = true;
        }
        index += 1;
    }
    if plan.comparison == ComparisonMethod::Abs {
        return None;
    }
    Some(plan)
}
impl Plan<'_> {
    fn materialize(self) -> BuiltinResult<ReductionArgs> {
        let selection = match self.axes {
            ReductionAxisPlan::Auto => DimSelection::Auto,
            ReductionAxisPlan::Dim(dimension) => DimSelection::Dim(dimension),
            ReductionAxisPlan::All => DimSelection::All,
            ReductionAxisPlan::Dimensions(dimensions) => match dimensions.len {
                0 => DimSelection::Auto,
                _ => DimSelection::Vec(dimensions.materialize()?),
            },
        };
        Ok(ReductionArgs {
            selection,
            nan_mode: self.nan_mode,
            comparison: self.comparison,
            linear_index: false,
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
            && query
                .controls
                .values
                .first()
                .is_some_and(is_empty_placeholder)
            && crate::builtins::common::spec::reduction_rejected_scalar_axis(
                &query.controls.values[1..],
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
        // Old output shape during scalar collapse; no host-kernel masks or
        // subspace strides are built by this semantic receipt.
        resolve_scratch_bytes: reduction_sum_bytes(&[reduction_layout::<usize>(rank)?])?,
        contract_retained_bytes: reduction_sum_bytes(&[
            reduction_layout::<FusionReductionContract>(1)?,
            reduction_layout::<usize>(rank)?,
            reduction_layout::<usize>(rank)?,
            reduction_layout::<usize>(axes)?,
        ])?,
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
        .real_floating_input()?
        else {
            return Ok(None);
        };
        let parsed = &self.0;
        // Empty extrema have a separate host shape path; it is not an identity
        // value suitable for the initial nonempty fused reduction contract.
        if checked_reduction_elements(&shape) == Some(0) {
            return Ok(None);
        }
        // Same public geometry as the ordinary owner, without allocating its
        // numerical-kernel-only masks and subspace strides.
        let scalar = is_scalar_shape(&shape);
        let axis_capacity = if scalar {
            0
        } else {
            match &parsed.selection {
                DimSelection::Auto | DimSelection::Dim(_) => 1,
                DimSelection::Vec(dims) => dims.len(),
                DimSelection::All => shape.len(),
            }
        };
        let mut axes = reduction_vec(axis_capacity)?;
        if !scalar {
            match &parsed.selection {
                DimSelection::Auto => {
                    axes.push(shape.iter().position(|&extent| extent > 1).unwrap_or(0))
                }
                DimSelection::Dim(dim) => {
                    if *dim > 0 && *dim <= shape.len() {
                        axes.push(dim - 1);
                    }
                }
                DimSelection::Vec(dims) => {
                    for &dim in dims {
                        if dim > 0 && dim <= shape.len() {
                            axes.push(dim - 1);
                        }
                    }
                }
                DimSelection::All => axes.extend(0..shape.len()),
            }
        }
        axes.sort_unstable();
        axes.dedup();
        let mut output_shape = if scalar {
            normalize_scalar_shape(&shape)
        } else {
            shape.clone()
        };
        for &axis in &axes {
            output_shape[axis] = 1;
        }
        let Some(count) = checked_reduction_elements(&output_shape) else {
            return Ok(None);
        };
        if count == 1 && input_class == NumericClass::Double {
            output_shape = vec![1, 1];
        }
        Ok(Some(FusionReductionContract {
            operation: FusionReductionOperation::Maximum,
            input_class,
            accumulator_class: input_class,
            output_class: input_class,
            input_shape: shape,
            output_shape,
            axes,
            identity: false,
            nan_mode: parsed.nan_mode,
            count: FusionReductionCount::None,
            empty: FusionReductionEmpty::NaN,
            ties: FusionReductionTies::FirstWithPositiveZero,
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
    fn maximum_preserves_class_positive_zero_and_nan_policy() {
        let fact = value_fact(&single(vec![1.0; 6], vec![2, 3]));
        let empty = || Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
        for controls in [
            vec![],
            vec![empty(), Value::String("AlL".into())],
            vec![
                empty(),
                empty(),
                Value::String("OMITNAN".into()),
                Value::String("includenan".into()),
            ],
            vec![
                empty(),
                Value::Tensor(Tensor::new(vec![2.0, 1.0, 2.0], vec![1, 3]).unwrap()),
            ],
            vec![empty(), Value::Num(1.0 + 5.0e-7)],
            vec![
                empty(),
                Value::CharArray(runmat_value::CharArray::new_row("ComparisonMethod")),
                Value::String("AbS".into()),
                Value::String("ComparisonMethod".into()),
                Value::String("ReAl".into()),
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
            let ParsedCall::Reduction(original) = futures::executor::block_on(parse_call(
                crate::call::FunctionArgumentView::from_slice(&controls),
            ))
            .unwrap() else {
                panic!("reduction controls");
            };
            assert_eq!(
                Prepared(plan.materialize().unwrap())
                    .resolve(&fact)
                    .unwrap(),
                Prepared(original).resolve(&fact).unwrap()
            );
        }
        for controls in [
            vec![
                empty(),
                Value::String("ComparisonMethod".into()),
                Value::String("abs".into()),
            ],
            vec![empty(), Value::String("linear".into())],
            vec![Value::Num(2.0)],
        ] {
            assert!(resources(FusionReductionPreparationQuery {
                controls: FusionReductionControls {
                    values: &controls,
                    requested_outputs: 1
                },
                input_rank_upper: 2
            })
            .is_none());
            assert!(
                futures::executor::block_on(prepare(FusionReductionControls {
                    values: &controls,
                    requested_outputs: 1
                }))
                .unwrap()
                .is_none()
            );
        }
        let duplicate = [
            empty(),
            Value::String("all".into()),
            Value::String("ALL".into()),
        ];
        let original = futures::executor::block_on(parse_call(
            crate::call::FunctionArgumentView::from_slice(&duplicate),
        ))
        .err()
        .unwrap();
        assert!(resources(FusionReductionPreparationQuery {
            controls: FusionReductionControls {
                values: &duplicate,
                requested_outputs: 1
            },
            input_rank_upper: 2
        })
        .is_none());
        let fallback = futures::executor::block_on(prepare(FusionReductionControls {
            values: &duplicate,
            requested_outputs: 1,
        }))
        .err()
        .unwrap();
        assert_eq!(original.identifier, fallback.identifier);
        assert_eq!(original.to_string(), fallback.to_string());
        let input = single(vec![-0.0, 0.0], vec![2, 1]);
        let contract = receipt(&input, &[]).unwrap();
        assert_eq!(contract.output_class, NumericClass::Single);
        assert_eq!(contract.ties, FusionReductionTies::FirstWithPositiveZero);
        let Value::Tensor(result) = futures::executor::block_on(max_builtin(
            None,
            input.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
            panic!("Single maximum");
        };
        assert!(result.materialize_f64()[0].is_sign_positive());
        let empty = Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
        let options = [empty.clone(), Value::String("omitnan".into())];
        let nan = single(vec![f32::NAN, f32::NAN], vec![2, 1]);
        assert_eq!(
            receipt(&nan, &options).unwrap().empty,
            FusionReductionEmpty::NaN
        );
        let Value::Tensor(result) = futures::executor::block_on(max_builtin(
            None,
            nan,
            crate::builtins::common::test_support::builtin_arguments(options.to_vec()),
        ))
        .unwrap() else {
            panic!("Single maximum");
        };
        assert!(result.materialize_f64()[0].is_nan());
        assert!(receipt(&input, &[Value::Num(2.0)]).is_none());
        assert!(receipt(&empty, &[]).is_none());
    }
}
