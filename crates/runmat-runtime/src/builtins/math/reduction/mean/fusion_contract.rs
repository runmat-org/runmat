use super::*;
use crate::builtins::common::spec::{
    checked_reduction_elements, FusionReductionContext, FusionReductionContract,
    FusionReductionControls, FusionReductionCount, FusionReductionDeclaration,
    FusionReductionEmpty, FusionReductionOperation, FusionReductionPrepareFuture,
    FusionReductionTies, PreparedFusionReduction,
};
use crate::builtins::common::spec::{
    reduction_layout, reduction_sum_bytes, FusionReductionPreparationQuery,
    FusionReductionPreparationQuote, ReductionAxisPlan, ReductionDimensions,
};
use runmat_types::NumericClass;

pub(super) const DECLARATION: FusionReductionDeclaration = FusionReductionDeclaration {
    revision: 6,
    operation: FusionReductionOperation::Mean,
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
    if parsed.weights.is_some() {
        return Ok(None);
    }
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
    let mut axes_set = false;
    let mut output_set = false;
    for value in controls.values {
        let control = match value {
            Value::String(runmat_value::StringElement::Text(text)) => {
                MeanControl::from_public_option(text)
            }
            Value::StringArray(array)
                if array.data().len() == 1 && !array.data()[0].is_missing() =>
            {
                MeanControl::from_public_option(
                    array.data()[0]
                        .text_ref()
                        .expect("checked nonmissing scalar string"),
                )
            }
            Value::CharArray(array) if array.rows == 1 => {
                MeanControl::from_public_code_units(array.data().iter().map(|unit| unit.raw()))
            }
            _ => None,
        };
        if let Some(control) = control {
            match control {
                MeanControl::Weights => return None,
                MeanControl::Missing(mode) => plan.nan_mode = mode,
                MeanControl::All => {
                    if axes_set && !plan.axes.is_auto() {
                        return None;
                    }
                    plan.axes = ReductionAxisPlan::All;
                    axes_set = true;
                }
                MeanControl::Output(output) => {
                    if output_set {
                        return None;
                    }
                    plan.output = output;
                    output_set = true;
                }
            }
        } else {
            if axes_set && !plan.axes.is_auto() {
                return None;
            }
            plan.axes = ReductionAxisPlan::Dimensions(ReductionDimensions::parse(value, true)?);
            axes_set = true;
        }
    }
    Some(plan)
}
impl Plan<'_> {
    fn materialize(self) -> BuiltinResult<ParsedArguments> {
        let axes = match self.axes {
            ReductionAxisPlan::Auto => MeanAxes::Default,
            ReductionAxisPlan::Dim(dimension) => MeanAxes::Dim(dimension),
            ReductionAxisPlan::All => MeanAxes::All,
            ReductionAxisPlan::Dimensions(dimensions) => {
                let mut values = dimensions.materialize()?;
                match values.len() {
                    0 => MeanAxes::Default,
                    1 => MeanAxes::Dim(values.pop().expect("one axis")),
                    _ => MeanAxes::Vec(values),
                }
            }
        };
        Ok(ParsedArguments {
            axes,
            nan_mode: self.nan_mode,
            output: self.output,
            weights: None,
        })
    }
}
fn resources(
    query: FusionReductionPreparationQuery<'_>,
) -> Option<FusionReductionPreparationQuote> {
    let rank = query.input_rank_upper.max(2);
    let (dimensions, retained_axes, temporary_scalar_axis, all_axes) =
        match lexical_plan(&query.controls) {
            Some(plan) => (
                plan.axes.retained_len().max(1),
                if plan.axes.retained_len() > 1 {
                    plan.axes.retained_len()
                } else {
                    0
                },
                usize::from(plan.axes.retained_len() == 1),
                matches!(plan.axes, ReductionAxisPlan::All),
            ),
            // The ordinary scalar-axis parser may allocate its one-element
            // dimension worklist before returning the original typed diagnostic.
            None if query.controls.supported()
                && crate::builtins::common::spec::reduction_rejected_scalar_axis(
                    query.controls.values,
                ) =>
            {
                (1, 0, 1, false)
            }
            None => return None,
        };
    let stages = dimensions;
    let mut retained = reduction_layout::<FusionReductionContract>(stages)?;
    // Each actual stage owns input shape, output shape, and axes. All-axes
    // uses rank axes; other stages use one. Rank is the captured shape bound.
    let axis_count = if all_axes { rank } else { 1 };
    let stage_backing = reduction_sum_bytes(&[
        reduction_layout::<usize>(rank)?,
        reduction_layout::<usize>(rank)?,
        reduction_layout::<usize>(axis_count)?,
    ])?;
    retained = retained.checked_add(stage_backing.checked_mul(u64::try_from(stages).ok()?)?)?;
    Some(FusionReductionPreparationQuote {
        prepare_scratch_bytes: reduction_sum_bytes(&[
            u64::try_from(std::mem::size_of_val(&prepare_impl(query.controls))).ok()?,
            reduction_layout::<usize>(temporary_scalar_axis)?,
        ])?,
        prepared_retained_bytes: reduction_sum_bytes(&[
            reduction_layout::<Prepared>(1)?,
            reduction_layout::<usize>(retained_axes)?,
        ])?,
        // Vecdim sort copy and mapped Option worklist; current-shape carry
        // and pre-publication output overlap.
        resolve_scratch_bytes: reduction_sum_bytes(&[
            reduction_layout::<usize>(retained_axes)?,
            reduction_layout::<Option<usize>>(dimensions)?,
            reduction_layout::<usize>(rank)?,
            reduction_layout::<usize>(rank)?,
        ])?,
        contract_retained_bytes: retained,
    })
}

struct Prepared(ParsedArguments);
impl PreparedFusionReduction for Prepared {
    fn publication_storage(
        &self,
        input: &runmat_types::ValueFact,
        output: &runmat_types::ValueFact,
        stages: &[FusionReductionContract],
    ) -> Option<runmat_types::StorageFact> {
        // These checks use represented ordinary geometry/classes and the
        // public carrier only; they do not assume nonnegative input payloads.
        if !matches!(
            input.storage,
            runmat_types::StorageFact::Dense | runmat_types::StorageFact::Scalar
        ) {
            return None;
        }
        super::fusion_sign::ordinary_output_storage(input, output, stages)
    }

    fn nonnegative_output_storage(
        &self,
        input: &runmat_types::ValueFact,
        output: &runmat_types::ValueFact,
        stages: &[FusionReductionContract],
    ) -> Option<runmat_types::StorageFact> {
        super::fusion_sign::ordinary_output_storage(input, output, stages)
    }

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
        .real_floating_input()?
        else {
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
            let final_output_class =
                (index + 1 == dimensions.len()).then_some(match parsed.output {
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
    // Only Double scratch/publication collapses a singleton into Num. An
    // out-of-rank Single identity returns its original full-rank tensor;
    // a native/default Single template preserves that carrier and shape.
    // Intermediate stages retain their geometry until final publication.
    if final_output_class.is_some() && (raw_double || output_class == NumericClass::Double) {
        output_shape = mean_public_shape(&output_shape, count);
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

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn publication_carrier_does_not_assume_positive_payload_or_certify_opaque_input() {
        let value = single(vec![-3.0, -5.0, -7.0, -9.0], vec![2, 2, 1]);
        let input = value_fact(&value);
        let controls = [
            Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()),
            Value::String("native".into()),
        ];
        let actual = futures::executor::block_on(mean_builtin(
            value,
            crate::builtins::common::test_support::builtin_arguments(controls.to_vec()),
        ))
        .unwrap();
        let mut output = value_fact(&actual);
        output.storage = runmat_types::StorageFact::Unknown;
        let certify = |input: &runmat_types::ValueFact| {
            futures::executor::block_on(DECLARATION.certify_publication_storage(
                FusionReductionContext {
                    input,
                    controls: &controls,
                    requested_outputs: 1,
                },
                &output,
            ))
            .unwrap()
        };
        assert_eq!(certify(&input), Some(runmat_types::StorageFact::Dense));
        for storage in [
            runmat_types::StorageFact::Unknown,
            runmat_types::StorageFact::Sparse,
            runmat_types::StorageFact::Opaque,
        ] {
            let mut wrong = input.clone();
            wrong.storage = storage;
            assert_eq!(certify(&wrong), None);
        }
    }

    #[test]
    fn mean_uses_double_sum_and_actual_valid_count_before_single_restoration() {
        let fact = value_fact(&single(vec![1.0; 6], vec![2, 3]));
        for controls in [
            vec![],
            vec![Value::String(" AlL ".into())],
            vec![Value::CharArray(runmat_value::CharArray::new_row(
                "\u{2003}ALL\u{2003}",
            ))],
            vec![
                Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
                Value::Num(2.0),
            ],
            vec![
                Value::Tensor(Tensor::new(vec![0.0, 2.0, 2.0], vec![1, 3]).unwrap()),
                Value::String("NaTiVe".into()),
            ],
            vec![Value::Num(1.0), Value::String("OMITMISSING".into())],
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
                    .resolve_stages(&fact)
                    .unwrap(),
                Prepared(original).resolve_stages(&fact).unwrap()
            );
        }
        let weights = [Value::String("Weights".into()), Value::Num(1.0)];
        assert!(resources(FusionReductionPreparationQuery {
            controls: FusionReductionControls {
                values: &weights,
                requested_outputs: 1
            },
            input_rank_upper: 2
        })
        .is_none());
        assert!(futures::executor::block_on(parse_arguments(
            crate::call::FunctionArgumentView::from_slice(&weights)
        ))
        .unwrap()
        .weights
        .is_some());
        assert!(
            futures::executor::block_on(prepare(FusionReductionControls {
                values: &weights,
                requested_outputs: 1
            }))
            .unwrap()
            .is_none()
        );
        let duplicate = [
            Value::String("default".into()),
            Value::String("DEFAULT".into()),
        ];
        let original = futures::executor::block_on(parse_arguments(
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
        let input = single(vec![16_777_216.0, 1.0, 1.0], vec![3, 1]);
        let contract = receipt(&input, &[]).unwrap();
        assert_eq!(contract.accumulator_class, NumericClass::Double);
        assert_eq!(contract.output_class, NumericClass::Single);
        assert_eq!(contract.count, FusionReductionCount::SliceLength);
        let actual = futures::executor::block_on(mean_builtin(
            input.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
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
        let Value::Tensor(result) = futures::executor::block_on(mean_builtin(
            scalar,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
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
        for input in [
            single(vec![-0.0], vec![1, 1, 1]),
            Value::Tensor(Tensor::new(vec![-0.0], vec![1, 1, 1]).unwrap()),
        ] {
            for template in [None, Some("native"), Some("double")] {
                let mut controls = vec![Value::Num(4.0)];
                if let Some(template) = template {
                    controls.push(Value::String(template.into()));
                }
                let contract = receipt(&input, &controls).unwrap();
                assert!(contract.identity);
                assert!(contract.axes.is_empty());
                assert_eq!(contract.accumulator_class, NumericClass::Double);
                let expected_shape = if contract.output_class == NumericClass::Single {
                    vec![1, 1, 1]
                } else {
                    vec![1, 1]
                };
                assert_eq!(contract.output_shape, expected_shape);
                let actual = futures::executor::block_on(mean_builtin(
                    input.clone(),
                    crate::builtins::common::test_support::builtin_arguments(controls),
                ))
                .unwrap();
                let fact = value_fact(&actual);
                assert_eq!(fact.numeric().unwrap().class, contract.output_class);
                let shape: Option<Vec<_>> = fact.shape.known_dims().unwrap().into_iter().collect();
                assert_eq!(shape.unwrap(), contract.output_shape);
                let bits = match actual {
                    Value::Num(value) => value.to_bits(),
                    Value::Tensor(tensor) => tensor.materialize_f64()[0].to_bits(),
                    other => panic!("unexpected mean identity carrier {other:?}"),
                };
                assert_eq!(bits, (-0.0f64).to_bits());
            }
        }
        let input = single(vec![3.0, 5.0], vec![1, 1, 2]);
        let axes = |dims: Vec<f64>| Value::Tensor(Tensor::new(dims, vec![1, 2]).unwrap());
        let one = stages(&input, &[Value::Num(3.0)]).unwrap();
        assert_eq!(one[0].output_shape, vec![1, 1]);
        assert_eq!(one[0].output_class, NumericClass::Single);
        let Value::Tensor(one_actual) = futures::executor::block_on(mean_builtin(
            input.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(3.0)]),
        ))
        .unwrap() else {
            panic!("Single one-element mean is republished as a tensor");
        };
        assert_eq!(one_actual.shape(), one[0].output_shape);
        let pair = stages(&input, &[axes(vec![3.0, 4.0])]).unwrap();
        assert_eq!(pair[0].output_shape, vec![1, 1, 1]);
        assert_eq!(pair[0].output_class, NumericClass::Double);
        assert_eq!(pair[1].input_shape, vec![1, 1, 1]);
        assert_eq!(pair[1].output_shape, vec![1, 1]);
        assert_eq!(pair[1].output_class, NumericClass::Single);
        let Value::Tensor(pair_actual) = futures::executor::block_on(mean_builtin(
            input.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![axes(vec![3.0, 4.0])]),
        ))
        .unwrap() else {
            panic!("Single vecdim publication remains a tensor");
        };
        assert_eq!(pair_actual.shape(), pair[1].output_shape);
        let double = stages(
            &input,
            &[axes(vec![3.0, 4.0]), Value::String("double".into())],
        )
        .unwrap();
        assert_eq!(double[1].output_shape, vec![1, 1]);
        let double_actual = futures::executor::block_on(mean_builtin(
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![
                axes(vec![3.0, 4.0]),
                Value::String("double".into()),
            ]),
        ))
        .unwrap();
        assert!(matches!(double_actual, Value::Num(4.0)));
    }
}
