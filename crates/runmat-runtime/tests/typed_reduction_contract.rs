//! Single-library canonical production declaration joins, independent of legacy lib-test fixtures.
use futures::executor::block_on;
use runmat_runtime::{
    builtins::common::spec::{
        selected_fusion_binding, FusionReductionContext, FusionReductionContract,
        FusionReductionCount, FusionReductionDeclaration, FusionReductionEmpty,
        FusionReductionOperation, FusionReductionTies, ReductionNaN, SelectedFusionIdentity,
    },
    call_builtin_async_with_outputs,
    context::{RuntimeBuiltinService, RuntimeContext, RuntimeServicePorts},
    value_fact::value_fact,
    RuntimeError,
};
use runmat_types::NumericClass;
use runmat_value::{IntValue, Tensor, Value};
use std::rc::Rc;

fn declaration(name: &str) -> &'static FusionReductionDeclaration {
    let selected = selected_fusion_binding(name, None, None)
        .unwrap_or_else(|| panic!("{name}: production binding/provenance join missing"));
    let entry = runmat_builtins::builtin_catalog_entry_by_name(name).expect("canonical reduction");
    let [binding] = entry.bindings else {
        panic!("one canonical reduction binding")
    };
    let expected = entry.binding_identity(binding);
    let SelectedFusionIdentity::Canonical(identity) = selected.identity else {
        panic!("{name}: production reduction must select canonical authority");
    };
    assert_eq!(identity, expected);
    assert_eq!(
        selected.canonical.expect("canonical declaration").binding,
        expected
    );
    let declaration = selected.reduction.expect("joined typed declaration");
    let revision = match name {
        "sum" => 3,
        "mean" => 6,
        "min" | "max" => 2,
        _ => panic!("unregistered reduction witness"),
    };
    assert_eq!(declaration.revision, revision);
    declaration
}
fn receipt(
    name: &str,
    input: &Value,
    controls: &[Value],
    outputs: usize,
) -> Result<Option<FusionReductionContract>, Box<RuntimeError>> {
    let fact = value_fact(input);
    block_on(declaration(name).resolve(FusionReductionContext {
        input: &fact,
        controls,
        requested_outputs: outputs,
    }))
    .map_err(Box::new)
}
fn host(
    name: &str,
    input: &Value,
    controls: &[Value],
) -> Result<runmat_runtime::execution::RootedValue, Box<RuntimeError>> {
    let mut args = vec![input.clone()];
    args.extend_from_slice(controls);
    let result = block_on(call_builtin_async_with_outputs(name, &args, 1)).map_err(Box::new)?;
    result.try_map_owned(
        |value| -> Result<Value, Box<RuntimeError>> {
            Ok(match value {
                Value::OutputList(outputs) => {
                    assert_eq!(outputs.len(), 1, "{name}: exactly one requested result");
                    runmat_runtime::sequence::into_first_output(outputs)
                        .map_err(Box::new)?
                        .expect("one requested result")
                }
                value => value,
            })
        },
        |error| Box::new(error.into_runtime_error()),
    )
}
fn array(single: bool, data: &[f64], shape: &[usize]) -> Value {
    if single {
        Value::Tensor(
            Tensor::from_f32(data.iter().map(|&x| x as f32).collect(), shape.to_vec()).unwrap(),
        )
    } else {
        Value::Tensor(Tensor::new(data.to_vec(), shape.to_vec()).unwrap())
    }
}
fn empty() -> Value {
    array(false, &[], &[0, 0])
}
fn text(value: &str) -> Value {
    Value::String(value.into())
}
fn values(value: &Value) -> Vec<f64> {
    match value {
        Value::Num(value) => vec![*value],
        Value::Tensor(value) => value.materialize_f64(),
        other => panic!("unexpected host carrier {other:?}"),
    }
}
fn compare(
    name: &str,
    input: &Value,
    controls: &[Value],
    expected: &[f64],
) -> FusionReductionContract {
    let contract = receipt(name, input, controls, 1)
        .unwrap()
        .expect("admitted host contract");
    let actual = host(name, input, controls).unwrap();
    let fact = value_fact(&actual);
    assert_eq!(
        fact.numeric()
            .unwrap_or_else(|| panic!("{name}: nonnumeric result {actual:?}"))
            .class,
        contract.output_class,
        "{name} {controls:?}"
    );
    let shape = fact
        .shape
        .known_dims()
        .unwrap()
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .unwrap();
    assert_eq!(shape, contract.output_shape, "{name} {controls:?}");
    let actual = values(&actual);
    assert_eq!(actual.len(), expected.len(), "{name} {controls:?}");
    for (&actual, &expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "{name}: {actual}");
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{name} {controls:?}");
        }
    }
    contract
}

struct EmptyService;
impl RuntimeBuiltinService for EmptyService {
    fn bindings_by_name(&self, _: &str) -> Vec<runmat_runtime::builtin::RuntimeBuiltinBinding> {
        Vec::new()
    }
}

#[test]
fn all_four_join_exact_compiled_canonical_provenance_and_reject_other_authority() {
    runmat_gc::gc_test_context(|| {
        let wrong = runmat_builtins::builtin_catalog_entry_by_name("sin").unwrap();
        let [binding] = wrong.bindings else {
            panic!("one canonical sin binding");
        };
        let wrong = wrong.binding_identity(binding);
        let context = RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ))
        .with_service_ports(RuntimeServicePorts::default().with_builtin(Rc::new(EmptyService)));
        for name in ["sum", "mean", "min", "max"] {
            assert_eq!(
                runmat_builtins::with_builtin_functions_by_exact_name(name, |entries| entries
                    .len()),
                0,
                "{name}: canonical reduction must not retain a legacy dispatcher"
            );
            let entry = runmat_builtins::builtin_catalog_entry_by_name(name).unwrap();
            let [binding] = entry.bindings else {
                panic!("one canonical reduction binding")
            };
            let expected = entry.binding_identity(binding);
            let compiled =
                runmat_runtime::builtin::with_runtime_builtin_bindings_by_name(name, |entries| {
                    let [binding] = entries else {
                        panic!("one compiled reduction binding")
                    };
                    *binding
                });
            assert_eq!(compiled.identity, expected);
            let owner =
                runmat_builtins::with_builtin_implementation_provenance_by_name(name, |entries| {
                    let canonical = entries
                        .iter()
                        .filter(|entry| {
                            entry.authority
                                == runmat_builtins::BuiltinImplementationAuthority::CanonicalBinding
                                && entry.binding_variant == Some(expected.variant)
                        })
                        .collect::<Vec<_>>();
                    let [owner] = canonical.as_slice() else {
                        panic!("one exact canonical provenance owner")
                    };
                    assert!(!entries.iter().any(|entry| entry.authority
                        == runmat_builtins::BuiltinImplementationAuthority::LegacyFunction));
                    owner.builtin_path
                });
            assert_eq!(
                compiled
                    .provenance
                    .expect("compiled provenance")
                    .builtin_path,
                owner
            );
            let selected = selected_fusion_binding(name, Some(expected), None).unwrap();
            let SelectedFusionIdentity::Canonical(identity) = selected.identity else {
                panic!("expected canonical join");
            };
            assert_eq!(identity, expected);
            assert_eq!(selected.canonical.unwrap().binding, expected);
            assert!(selected.reduction.is_some());
            assert!(selected_fusion_binding(name, Some(wrong), None).is_none());
            assert!(selected_fusion_binding(name, None, Some(&context)).is_none());
            assert!(
                selected_fusion_binding(name, None, None).is_some(),
                "scoped denial must not poison global join"
            );
        }
    });
}

#[test]
fn output_classes_and_nd_axes_match_actual_host_dispatch() {
    runmat_gc::gc_test_context(|| {
        for single in [false, true] {
            let input = array(single, &[1., 2., 3., 4., 5., 6., 7., 8.], &[2, 2, 2]);
            let class = if single {
                NumericClass::Single
            } else {
                NumericClass::Double
            };
            let sum = compare("sum", &input, &[], &[3., 7., 11., 15.]);
            assert_eq!(sum.axes, vec![0]);
            assert_eq!(sum.output_shape, vec![1, 2, 2]);
            assert_eq!(sum.output_class, class);
            assert_eq!(sum.accumulator_class, NumericClass::Double);
            assert_eq!(
                compare("sum", &input, &[text("native")], &[3., 7., 11., 15.]).output_class,
                class
            );
            assert_eq!(
                compare("mean", &input, &[], &[1.5, 3.5, 5.5, 7.5]).output_class,
                class
            );
            assert_eq!(
                compare("mean", &input, &[text("double")], &[1.5, 3.5, 5.5, 7.5]).output_class,
                NumericClass::Double
            );
            compare("sum", &input, &[Value::Num(2.0)], &[4., 6., 12., 14.]);
            compare("mean", &input, &[Value::Num(2.0)], &[2., 3., 6., 7.]);
            compare(
                "min",
                &input,
                &[empty(), Value::Num(2.0)],
                &[1., 2., 5., 6.],
            );
            compare(
                "max",
                &input,
                &[empty(), Value::Num(2.0)],
                &[3., 4., 7., 8.],
            );
            compare("sum", &input, &[text("all")], &[36.]);
            compare("mean", &input, &[text("all")], &[4.5]);
            compare("min", &input, &[empty(), text("all")], &[1.]);
            compare("max", &input, &[empty(), text("all")], &[8.]);
            for name in ["sum", "mean", "min", "max"] {
                let controls = if matches!(name, "min" | "max") {
                    vec![empty(), Value::Num(4.0)]
                } else {
                    vec![Value::Num(4.0)]
                };
                let contract = compare(name, &input, &controls, &[1., 2., 3., 4., 5., 6., 7., 8.]);
                assert!(contract.axes.is_empty(), "{name}");
                assert_eq!(contract.output_class, class, "{name} no-op class");
                assert_eq!(contract.input_shape, vec![2, 2, 2]);
            }
        }
    });
}

#[test]
fn rounding_and_signed_zero_are_owner_semantics_not_provider_precision() {
    runmat_gc::gc_test_context(|| {
        let input = array(true, &[16_777_216., 1., 1.], &[3, 1]);
        assert_eq!(
            compare("sum", &input, &[], &[16_777_218.]).accumulator_class,
            NumericClass::Double
        );
        compare("mean", &input, &[], &[5_592_406.]);
        for single in [false, true] {
            let scalar = array(single, &[-0.0], &[1, 1]);
            let sum = compare("sum", &scalar, &[], &[0.0]);
            assert!(!sum.identity);
            assert_eq!(sum.operation, FusionReductionOperation::Sum);
            let mean = compare("mean", &scalar, &[], &[-0.0]);
            assert!(mean.identity);
            let zeros = array(single, &[0.0, -0.0], &[2, 1]);
            assert_eq!(
                compare("min", &zeros, &[], &[-0.0]).ties,
                FusionReductionTies::FirstWithNegativeZero
            );
            assert_eq!(
                compare("max", &zeros, &[], &[0.0]).ties,
                FusionReductionTies::FirstWithPositiveZero
            );
            // A rank-three singleton is not the owner's scalar fast path.
            // Its default arithmetic and out-of-rank identity path differ.
            let nd_singleton = array(single, &[-0.0], &[1, 1, 1]);
            compare("sum", &nd_singleton, &[], &[0.0]);
            compare("mean", &nd_singleton, &[], &[0.0]);
            for name in ["sum", "mean", "min", "max"] {
                let controls = if matches!(name, "min" | "max") {
                    vec![empty(), Value::Num(4.0)]
                } else {
                    vec![Value::Num(4.0)]
                };
                compare(name, &nd_singleton, &controls, &[-0.0]);
            }
        }
    });
}

#[test]
fn nan_counts_and_empty_shapes_match_host_results() {
    runmat_gc::gc_test_context(|| {
        for single in [false, true] {
            for name in ["sum", "mean", "min", "max"] {
                let input = array(single, &[f64::NAN, 2.], &[2, 1]);
                assert_eq!(
                    compare(name, &input, &[], &[f64::NAN]).nan_mode,
                    ReductionNaN::Include
                );
                let controls = if matches!(name, "min" | "max") {
                    vec![empty(), text("omitnan")]
                } else {
                    vec![text("omitnan")]
                };
                let contract = compare(name, &input, &controls, &[2.]);
                assert_eq!(contract.nan_mode, ReductionNaN::Omit);
                if name == "mean" {
                    assert_eq!(contract.count, FusionReductionCount::NonNan);
                }
                let all_nan = array(single, &[f64::NAN, f64::NAN], &[2, 1]);
                let expected = if name == "sum" { 0.0 } else { f64::NAN };
                let contract = compare(name, &all_nan, &controls, &[expected]);
                assert_eq!(
                    contract.empty,
                    if name == "sum" {
                        FusionReductionEmpty::PositiveZero
                    } else {
                        FusionReductionEmpty::NaN
                    }
                );
            }
            for shape in [&[0, 3][..], &[3, 0][..], &[2, 0, 2][..]] {
                let input = array(single, &[], shape);
                for name in ["sum", "mean"] {
                    let value = if name == "sum" { 0.0 } else { f64::NAN };
                    let expected = if shape[0] == 0 {
                        vec![value; 3]
                    } else {
                        vec![]
                    };
                    compare(name, &input, &[], &expected);
                    compare(name, &input, &[text("all")], &[value]);
                }
                for name in ["min", "max"] {
                    assert!(receipt(name, &input, &[], 1).unwrap().is_none());
                }
            }
        }
    });
}

#[test]
fn invalid_controls_keep_the_owning_error_and_unsupported_forms_stay_explicit() {
    runmat_gc::gc_test_context(|| {
        let input = array(false, &[1., 2.], &[2, 1]);
        for name in ["sum", "mean", "min", "max"] {
            let controls = if matches!(name, "min" | "max") {
                vec![empty(), Value::Num(0.5)]
            } else {
                vec![Value::Num(0.5)]
            };
            let contract_error = receipt(name, &input, &controls, 1).unwrap_err();
            let host_error = host(name, &input, &controls).unwrap_err();
            assert_eq!(
                contract_error.identifier(),
                host_error.identifier(),
                "{name}"
            );
            assert_eq!(contract_error.message(), host_error.message(), "{name}");
            assert!(receipt(name, &input, &[], 2).unwrap().is_none());
            for unsupported in [
                Value::Complex(1., 2.),
                Value::Bool(true),
                Value::Int(IntValue::I64(9)),
            ] {
                assert!(receipt(name, &unsupported, &[], 1).unwrap().is_none());
            }
        }
        assert!(
            receipt("mean", &input, &[text("Weights"), Value::Num(1.0)], 1)
                .unwrap()
                .is_none()
        );
        let dims = array(false, &[1., 2.], &[1, 2]);
        assert!(receipt("mean", &input, &[dims], 1).unwrap().is_none());
        for name in ["min", "max"] {
            assert!(receipt(name, &input, &[Value::Num(1.0)], 1)
                .unwrap()
                .is_none());
        }
    });
}
