//! Optimization option construction and resident-payload materialization.

use crate::{build_runtime_error, BuiltinResult, RuntimeError, RuntimeErrorBuilder};
use runmat_builtins::{
    BuiltinErrorDescriptor, OPTIMSET_ERROR_INVALID_ARGUMENT, OPTIMSET_ERROR_INVALID_INPUT,
    OPTIMSET_INTEGER_OPTION_EXTENSION as INTEGER_OPTION_EXTENSION,
    OPTIMSET_RESIDENT_OPTION_EXTENSION as RESIDENT_OPTION_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

mod admission;

const NAME: &str = "optimset";

fn optimset_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    optimset_error_builder(error, detail).build()
}

fn optimset_error_builder(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeErrorBuilder {
    let detail = detail.as_ref();
    let message = if detail.starts_with("optimset:") {
        detail.to_string()
    } else {
        format!("{}: {detail}", error.message)
    };
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder
}

#[runtime_builtin(
    name = "optimset",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::optimset"
)]
async fn optimset_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    crate::context::host_allocation::require_unbounded_workspace(
        runtime.as_ref(),
        NAME,
        "finite host-memory execution is unsupported for option metadata construction",
    )?;
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OPTIMSET_DESCRIPTOR,
        &OPTIMSET_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    let _sources = rest.fork();
    // All names and pair grammar are admitted before any resident gather.
    let defaults = if rest.is_empty() {
        Some(false)
    } else if rest.len() == 1 {
        let name = match &rest[0] {
            Value::FunctionHandle(_) => {
                let _entry = runtime.as_ref().map(crate::context::RuntimeContext::enter);
                match crate::canonicalize_callback_handle_for_semantic_resolution(rest[0].clone())? {
                    Value::FunctionHandle(handle)=>Some(handle.name().to_owned()),
                    _=>return Err(optimset_error_with_detail(&OPTIMSET_ERROR_INVALID_ARGUMENT,"solver defaults require a builtin solver handle, not a selected user callback")),
                }
            }
            value => crate::builtins::common::validation::value_text_scalar_string(value),
        };
        match name {
            Some(name) if name.eq_ignore_ascii_case("fminsearch") => Some(true),
            Some(_) => {
                return Err(optimset_error_with_detail(
                    &OPTIMSET_ERROR_INVALID_ARGUMENT,
                    "solver defaults are currently available for fminsearch only",
                ))
            }
            None => None,
        }
    } else {
        None
    };
    if let Some(solver_defaults) = defaults {
        if rest.is_empty() && crate::output_count::current_output_count() == Some(0) {
            let _entry = runtime.as_ref().map(crate::context::RuntimeContext::enter);
            for field in runmat_builtins::FminsearchOptionName::ALL {
                use runmat_builtins::FminsearchOptionName::*;
                let choices = match field {
                    Display => "off | none | notify | final | iter",
                    FunValCheck => "off | on",
                    OutputFcn => "function handle | cell of handles | []",
                    PlotFcns => "function handle/name | cell of handles/names | []",
                    TolX | TolFun => "positive floating scalar",
                    MaxIter | MaxFunEvals => {
                        "positive integer-valued floating scalar | 200*numberOfVariables"
                    }
                };
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stdout,
                    format!("{}: {choices}", field.name()),
                )?;
            }
        }
        return default_options(solver_defaults, runtime.as_ref());
    }
    let steps = admission::admit(rest)?;
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    for step in steps {
        match step {
            admission::Step::Copy(existing, nonempty_only) => {
                for (name, value) in existing.into_fields_unbounded() {
                    if nonempty_only && value_is_empty(&value) {
                        continue;
                    }
                    let name = admission::canonical_name(name);
                    let value = prepare_optimset_payload(&name, value, runtime.as_ref()).await?;
                    fields.insert(name, value);
                }
            }
            admission::Step::Pair(name, value) => {
                let name = name.into_public_name();
                let value = prepare_optimset_payload(&name, value, runtime.as_ref()).await?;
                fields.insert(name, value);
            }
        }
    }
    Ok(Value::Struct(fields.finish()))
}

fn callback_field(name: &str) -> bool {
    matches!(runmat_builtins::OptimParameterName::resolve(name),runmat_builtins::OptimParameterLookup::Matched(field)if field.is_callback())
}
fn value_is_empty(value: &Value) -> bool {
    crate::value_fact::value_fact(value).shape.element_count() == Some(0)
}
fn default_options(
    solver: bool,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    let empty = super::common::output_callback::empty(runtime)?;
    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
    for field in runmat_builtins::FminsearchOptionName::ALL {
        use runmat_builtins::FminsearchOptionName::*;
        let value = if !solver {
            empty.clone()
        } else {
            match field {
                Display => Value::from("notify"),
                FunValCheck => Value::from("off"),
                MaxIter | MaxFunEvals => Value::from(runmat_builtins::FMINSEARCH_DEFAULT_BUDGET),
                TolX | TolFun => Value::Num(1e-4),
                OutputFcn | PlotFcns => empty.clone(),
            }
        };
        fields.insert(field.name(), value);
    }
    Ok(Value::Struct(fields.finish()))
}
fn ensure_optimset_extensions(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<()> {
    let mut integer = false;
    let mut explicit = false;
    let mut inspect = |name: &str, value: &Value| {
        if callback_field(name) {
            return;
        }
        integer |= crate::builtins::common::validation::value_contains_native_integer_class(value);
        explicit |= crate::builtins::common::validation::value_contains_explicit_gpu(value);
    };
    let mut start = 0;
    if let Some(value) = args.first() {
        if let Value::Struct(value) = value {
            for (name, value) in &value.fields {
                inspect(name, value);
            }
            start = 1;
        } else if matches!(value,Value::Tensor(value)if value.is_empty()) {
            start = 1;
        }
    }
    if args.len() == 2 && start == 1 {
        if let Value::Struct(value) = &args[1] {
            for (name, value) in &value.fields {
                inspect(name, value);
            }
        }
    } else {
        let mut index = start;
        while index + 1 < args.len() {
            let name = super::common::field_name(&args[index]).unwrap_or_default();
            inspect(&name, &args[index + 1]);
            index += 2;
        }
    }
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_OPTION_EXTENSION, NAME)?;
    }
    if explicit {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_OPTION_EXTENSION, NAME)?;
    }
    Ok(())
}
async fn prepare_optimset_payload(
    name: &str,
    value: Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<Value> {
    // Callback payloads and captures retain their original execution authority.
    if callback_field(name)
        || crate::builtins::common::validation::value_contains_explicit_gpu(&value)
    {
        return Ok(value);
    }
    let prepared = crate::dispatcher::gather_if_needed_in_context_async(&value, runtime)
        .await
        .map_err(|mut error| {
            if crate::runtime_error::error_has_control_cancellation_origin(&error) {
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return error;
            }
            let origin = error.failure_origin();
            optimset_error_builder(
                &OPTIMSET_ERROR_INVALID_INPUT,
                format!("failed to gather option value: {error}"),
            )
            .with_failure_origin(origin)
            .with_source(error)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
        })?;
    if let Some(runtime) = runtime {
        let mut budget = runtime.cpu_cooperation();
        budget
            .checkpoint(0)
            .await
            .map_err(|source| source.into_runtime_error_in_context(Some(runtime)))?;
    }
    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::OPTIMSET_DESCRIPTOR;
    use runmat_value::{IntegerStorage, Tensor};

    #[test]
    fn optimset_builds_struct_from_pairs() {
        let value = block_on(optimset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("TolX"),
                Value::Num(1.0e-8),
                Value::from("Display"),
                Value::from("off"),
            ]),
        ))
        .unwrap();
        match value {
            Value::Struct(options) => {
                assert!(matches!(options.fields.get("TolX"), Some(Value::Num(_))));
                assert!(matches!(
                    options.fields.get("Display"),
                    Some(Value::String(_))
                ));
            }
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn optimset_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = OPTIMSET_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                "options = optimset('fminsearch')",
                "options = optimset(oldopts, newopts)",
                "options = optimset()",
                "options = optimset(name, value, ...)",
                "options = optimset(oldopts, name, value, ...)",
            ]
        );

        let codes: Vec<&str> = OPTIMSET_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert_eq!(
            codes,
            vec!["RM.OPTIMSET.INVALID_ARGUMENT", "RM.OPTIMSET.INVALID_INPUT"]
        );
    }

    #[test]
    fn optimset_odd_name_value_pairs_use_stable_identifier() {
        let err = block_on(optimset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::from("TolX")]),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:optimset:InvalidArgument"));
    }

    #[test]
    fn optimset_rejects_all_names_and_grammar_before_resident_materialization() {
        let _guard = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(
            crate::context::physical_test_support::OwnedPhysicalProvider::new(),
        ));
        let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        // This existing fixture owns a live allocation but cannot download.
        // A premature gather would replace the intended argument error.
        let allocation = provider.allocate_f64(&[1, 1]);
        let handle = allocation
            .handle
            .clone()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap();
        for bad_tail in [
            vec![Value::from("Display")],
            vec![Value::Num(1.0), Value::Bool(true)],
        ] {
            let odd = bad_tail.len() == 1;
            let mut arguments = vec![Value::from("TolX"), Value::GpuTensor(handle.clone())];
            arguments.extend(bad_tail);
            let error = block_on(optimset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(arguments),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some(if odd {
                    "RunMat:optimset:InvalidArgument"
                } else {
                    "RunMat:optimset:InvalidInput"
                })
            );
            assert!(
                error.source.is_none(),
                "no provider failure may precede admission"
            );
            assert!(!error.message().contains("failed to gather"));
            assert!(handle.allocation_lease().validate_live().is_ok());
        }
    }

    #[test]
    fn optimset_strict_mode_rejects_integer_option_payload() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let value = Tensor::new_integer(IntegerStorage::I64(vec![17]), vec![1, 1]).unwrap();

        let error = block_on(optimset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("MaxIter"),
                Value::Tensor(value),
            ]),
        ))
        .expect_err("integer payload is a RunMat-only extension");

        assert_eq!(
            error.identifier(),
            INTEGER_OPTION_EXTENSION.error_identifier
        );
    }

    #[test]
    fn optimset_runmat_mode_preserves_wide_integer_payload_exactly() {
        let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
        let expected = u64::MAX;
        let value = Tensor::new_integer(IntegerStorage::U64(vec![expected]), vec![1, 1]).unwrap();

        let result = block_on(optimset_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::from("MaxIter"),
                Value::Tensor(value),
            ]),
        ))
        .expect("RunMat integer payload");

        let Value::Struct(options) = result else {
            panic!("expected options struct")
        };
        let Some(Value::Tensor(stored)) = options.fields.get("MaxIter") else {
            panic!("expected exact MaxIter tensor")
        };
        assert!(matches!(
            stored.numeric_value_at(0),
            Some(runmat_value::NumericScalar::U64(value)) if value == expected
        ));
    }

    #[test]
    fn optimset_automatic_resident_payload_gathers_but_explicit_payload_is_gated() {
        test_support::with_test_provider(|provider| {
            let values = [3.5];
            let shape = [1, 1];
            let automatic = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("automatic upload");
            let automatic = automatic
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("admit provenance view");

            let automatic_result = block_on(optimset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("TolX"),
                    Value::GpuTensor(automatic),
                ]),
            ))
            .expect("automatic residency gathers");
            assert!(matches!(
                automatic_result,
                Value::Struct(ref options)
                    if matches!(options.fields.get("TolX"), Some(Value::Tensor(tensor)) if tensor.materialize_f64() == [3.5])
            ));

            let explicit_input = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("explicit upload");
            let explicit = explicit_input
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
                .expect("admit provenance view");
            drop(explicit_input);
            let physical = explicit.allocation_lease().downgrade();
            let strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(optimset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("TolX"),
                    Value::GpuTensor(explicit.clone()),
                ]),
            ))
            .expect_err("explicit payload must be gated");
            assert_eq!(
                error.identifier(),
                RESIDENT_OPTION_EXTENSION.error_identifier
            );
            drop(strict);

            let _extensions = crate::compatibility::push_runmat_extensions_enabled(true);
            let mixed_automatic = provider
                .upload(&HostTensorView {
                    data: &values,
                    shape: &shape,
                })
                .expect("mixed automatic upload");
            let mixed_automatic = mixed_automatic
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("admit provenance view");
            let explicit_result = block_on(optimset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("TolX"),
                    Value::GpuTensor(explicit),
                    Value::from("MaxFunEvals"),
                    Value::GpuTensor(mixed_automatic),
                ]),
            ))
            .expect("RunMat preserves explicit and gathers automatic payloads independently");
            assert!(matches!(
                explicit_result,
                Value::Struct(ref options)
                    if matches!(options.fields.get("TolX"), Some(Value::GpuTensor(_)))
                        && matches!(options.fields.get("MaxFunEvals"), Some(Value::Tensor(tensor)) if tensor.materialize_f64() == [3.5])
            ));
            assert!(
                physical.acquire().is_some(),
                "returned option owns explicit payload"
            );
            let retained = explicit_result.clone();
            drop(explicit_result);
            assert!(
                physical.acquire().is_some(),
                "cloned option retains physical ownership"
            );
            let Value::Struct(options) = &retained else {
                unreachable!()
            };
            let Some(Value::GpuTensor(handle)) = options.fields.get("TolX") else {
                unreachable!()
            };
            assert!(runmat_accelerate_api::handle_is_explicit(handle));
            let payload = test_support::gather(Value::GpuTensor(handle.clone())).unwrap();
            assert_eq!(payload.materialize_f64(), vec![3.5]);
            drop(retained);
            assert!(
                physical.acquire().is_none(),
                "final returned option releases its allocation"
            );

            // These are actual owned ObjectArray elements, nested through both
            // a struct and cell. Compatibility and retention must traverse the
            // same owned graph as gather, without following managed referents.
            fn nested_objects(payload: Value) -> Value {
                let mut object = runmat_value::ObjectInstance::new("OptimsetPayload");
                object.replace_property_unbounded("payload".into(), payload);
                let array = runmat_value::ObjectArray::from_objects(
                    "OptimsetPayload",
                    vec![object.clone(), object],
                    vec![1, 2],
                )
                .expect("homogeneous value objects");
                let mut structure = runmat_value::UnboundedStructValueBuilder::new();
                structure.insert("items", Value::ObjectArray(array));
                Value::Cell(
                    runmat_value::CellArray::new(vec![Value::Struct(structure.finish())], 1, 1)
                        .unwrap(),
                )
            }
            fn objects(payload: &Value) -> &runmat_value::ObjectArray {
                let Value::Cell(cell) = payload else {
                    panic!("outer cell")
                };
                let Value::Struct(structure) = &cell.data()[0] else {
                    panic!("nested struct")
                };
                let Value::ObjectArray(array) = &structure.fields["items"] else {
                    panic!("object array")
                };
                assert_eq!(array.class_name().display_name(), "OptimsetPayload");
                assert_eq!(array.shape(), &[1, 2]);
                array
            }
            let upload = |intent| {
                provider
                    .upload(&HostTensorView {
                        data: &values,
                        shape: &shape,
                    })
                    .unwrap()
                    .with_provenance(intent)
                    .unwrap()
            };
            let explicit = upload(runmat_accelerate_api::GpuHandleProvenance::Explicit);
            let automatic = upload(runmat_accelerate_api::GpuHandleProvenance::Automatic);
            let explicit_lease = explicit.allocation_lease().downgrade();
            let automatic_lease = automatic.allocation_lease().downgrade();
            let mixed = nested_objects(Value::Cell(
                runmat_value::CellArray::new(
                    vec![
                        Value::GpuTensor(explicit.clone()),
                        Value::GpuTensor(automatic.clone()),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ));
            let strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(optimset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("FutureOption"),
                    mixed.clone(),
                ]),
            ))
            .expect_err("nested explicit ObjectArray value must be gated before gather");
            assert_eq!(
                error.identifier(),
                RESIDENT_OPTION_EXTENSION.error_identifier
            );
            drop(strict);
            let Value::Struct(result) = block_on(optimset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("FutureOption"),
                    mixed.clone(),
                ]),
            ))
            .unwrap() else {
                panic!("options struct")
            };
            let stored = &result.fields["FutureOption"];
            assert_eq!(stored, &mixed, "entire mixed payload is retained");
            assert_eq!(objects(stored).len(), 2);
            let mut handles = Vec::new();
            crate::dispatcher::try_visit_gpu_handles(stored, &mut |handle| {
                handles.push(handle.clone())
            })
            .unwrap();
            assert_eq!(handles.len(), 4);
            for pair in handles.chunks_exact(2) {
                assert!(pair[0].shares_allocation_with(&explicit));
                assert!(runmat_accelerate_api::handle_is_explicit(&pair[0]));
                assert!(pair[1].shares_allocation_with(&automatic));
                assert!(!runmat_accelerate_api::handle_is_explicit(&pair[1]));
            }
            drop(handles);
            drop(mixed);
            drop(explicit);
            drop(automatic);
            assert!(explicit_lease.acquire().is_some());
            assert!(
                automatic_lease.acquire().is_some(),
                "retained automatic sibling owns its lease"
            );
            drop(result);
            assert!(explicit_lease.acquire().is_none());
            assert!(automatic_lease.acquire().is_none());

            let automatic = nested_objects(Value::GpuTensor(upload(
                runmat_accelerate_api::GpuHandleProvenance::Automatic,
            )));
            let strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let Value::Struct(result) = block_on(optimset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("FutureOption"),
                    automatic,
                ]),
            ))
            .unwrap() else {
                panic!("options struct")
            };
            for value in objects(&result.fields["FutureOption"]).data() {
                let Value::Object(object) = value else {
                    panic!("value object")
                };
                assert!(
                    matches!(&object.properties()["payload"], Value::Tensor(tensor) if tensor.materialize_f64() == values)
                );
            }
            drop(strict);

            let integer = nested_objects(Value::Tensor(
                Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
            ));
            let strict = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(optimset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("FutureOption"),
                    integer.clone(),
                ]),
            ))
            .expect_err("nested native ObjectArray property must be gated");
            assert_eq!(
                error.identifier(),
                INTEGER_OPTION_EXTENSION.error_identifier
            );
            drop(strict);
            let Value::Struct(result) = block_on(optimset_builtin(
                None,
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::from("FutureOption"),
                    integer,
                ]),
            ))
            .unwrap() else {
                panic!("options struct")
            };
            for value in objects(&result.fields["FutureOption"]).data() {
                let Value::Object(object) = value else {
                    panic!("value object")
                };
                let Value::Tensor(tensor) = &object.properties()["payload"] else {
                    panic!("integer tensor")
                };
                assert!(matches!(
                    tensor.numeric_value_at(0),
                    Some(runmat_value::NumericScalar::U64(u64::MAX))
                ));
            }
        });
    }

    #[test]
    fn legacy_defaults_merge_reset_and_prefix_forms_are_real() {
        let args = crate::builtins::common::test_support::builtin_arguments;
        let Value::Struct(defaults) = block_on(optimset_builtin(None, args(vec![]))).unwrap()
        else {
            panic!("defaults")
        };
        assert_eq!(
            defaults.fields.len(),
            runmat_builtins::FminsearchOptionName::ALL.len()
        );
        assert!(defaults
            .fields
            .values()
            .all(|value| matches!(value,Value::Tensor(value)if value.is_empty())));
        let Value::Struct(solver) = block_on(optimset_builtin(
            None,
            args(vec![Value::from("fminsearch")]),
        ))
        .unwrap() else {
            panic!("solver defaults")
        };
        assert!(matches!(solver.fields["TolX"], Value::Num(1e-4)));
        assert_eq!(
            crate::builtins::common::validation::value_text_scalar_string(
                &solver.fields["MaxIter"]
            )
            .as_deref(),
            Some(runmat_builtins::FMINSEARCH_DEFAULT_BUDGET)
        );
        let Value::Struct(old) = block_on(optimset_builtin(
            None,
            args(vec![
                Value::from("tolx"),
                Value::Num(0.01),
                Value::from("MaxIt"),
                Value::Num(5.0),
            ]),
        ))
        .unwrap() else {
            panic!("old")
        };
        let mut newer = runmat_value::UnboundedStructValueBuilder::new();
        newer.insert(
            "TolX",
            super::super::common::output_callback::empty(None).unwrap(),
        );
        newer.insert("maxiter", Value::Num(9.0));
        let Value::Struct(merged) = block_on(optimset_builtin(
            None,
            args(vec![Value::Struct(old), Value::Struct(newer.finish())]),
        ))
        .unwrap() else {
            panic!("merge")
        };
        assert!(matches!(merged.fields["TolX"], Value::Num(0.01)));
        assert!(matches!(merged.fields["MaxIter"], Value::Num(9.0)));
        let Value::Struct(reset) = block_on(optimset_builtin(
            None,
            args(vec![
                Value::Struct(merged),
                Value::from("TolX"),
                super::super::common::output_callback::empty(None).unwrap(),
            ]),
        ))
        .unwrap() else {
            panic!("reset")
        };
        assert!(matches!(&reset.fields["TolX"],Value::Tensor(value)if value.is_empty()));
        let Value::Struct(from_empty) = block_on(optimset_builtin(
            None,
            args(vec![
                super::super::common::output_callback::empty(None).unwrap(),
                Value::from("Dis"),
                Value::from("off"),
            ]),
        ))
        .unwrap() else {
            panic!("empty oldopts")
        };
        assert!(from_empty.fields.contains_key("Display"));
    }
    #[test]
    fn callback_capture_identity_is_not_an_option_numeric_materialization() {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let _lock = test_support::accel_test_lock();
        let provider = Box::leak(Box::new(
            crate::context::physical_test_support::OwnedPhysicalProvider::new(),
        ));
        let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let allocation = provider.allocate_f64(&[1, 1]);
        let handle = allocation
            .handle
            .clone()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .unwrap();
        let callback = Value::Closure(runmat_value::Closure::new(
            "callback_capture".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            None,
            vec![
                Value::GpuTensor(handle.clone()),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1]).unwrap(),
                ),
            ],
            None,
        ));
        let instance = callback.callable_instance_key();
        let args = crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("OutputFcn"),
            callback,
        ]);
        let Value::Struct(result) = block_on(optimset_builtin(None, args)).unwrap() else {
            panic!("options")
        };
        assert_eq!(result.fields["OutputFcn"].callable_instance_key(), instance);
        let Value::Closure(stored) = &result.fields["OutputFcn"] else {
            panic!("closure")
        };
        assert!(
            matches!(&stored.captures[0],Value::GpuTensor(value)if value.view_identity()==handle.view_identity())
        );
        assert!(handle.allocation_lease().validate_live().is_ok());
    }
}
