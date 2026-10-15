//! Optimization option construction and resident-payload materialization.

use crate::{build_runtime_error, BuiltinResult, RuntimeError, RuntimeErrorBuilder};
use runmat_builtins::{
    BuiltinErrorDescriptor, OPTIMSET_ERROR_INVALID_ARGUMENT, OPTIMSET_ERROR_INVALID_INPUT,
    OPTIMSET_INTEGER_OPTION_EXTENSION as INTEGER_OPTION_EXTENSION,
    OPTIMSET_RESIDENT_OPTION_EXTENSION as RESIDENT_OPTION_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{StructValue, Value};

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
    builtin_path = "crate::builtins::math::optim::optimset"
)]
async fn optimset_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::OPTIMSET_DESCRIPTOR,
        &OPTIMSET_ERROR_INVALID_ARGUMENT,
        NAME,
    )?;
    // All names and pair grammar are admitted before any resident gather.
    let steps = admission::admit(rest)?;
    let mut fields = StructValue::new();
    for step in steps {
        match step {
            admission::Step::Copy(existing) => {
                for (name, value) in existing.fields {
                    fields.insert(name, prepare_optimset_payload(value).await?);
                }
            }
            admission::Step::Pair(name, value) => {
                fields.insert(
                    name.into_public_name(),
                    prepare_optimset_payload(value).await?,
                );
            }
        }
    }
    Ok(Value::Struct(fields))
}

fn ensure_optimset_extensions(args: &[Value]) -> BuiltinResult<()> {
    let mut payloads = Vec::new();
    let mut pair_start = 0usize;
    if let Some(Value::Struct(existing)) = args.first() {
        payloads.extend(existing.fields.values());
        pair_start = 1;
    }
    let mut index = pair_start + 1;
    while index < args.len() {
        payloads.push(&args[index]);
        index += 2;
    }
    let integer = payloads.iter().any(|value| {
        crate::builtins::common::validation::value_contains_native_integer_class(value)
    });
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(&INTEGER_OPTION_EXTENSION, NAME)?;
    }
    let explicit = payloads
        .iter()
        .any(|value| crate::builtins::common::validation::value_contains_explicit_gpu(value));
    if explicit {
        crate::compatibility::ensure_builtin_extension_enabled(&RESIDENT_OPTION_EXTENSION, NAME)?;
    }
    Ok(())
}

async fn prepare_optimset_payload(value: Value) -> BuiltinResult<Value> {
    if crate::builtins::common::validation::value_contains_explicit_gpu(&value) {
        return Ok(value);
    }
    crate::dispatcher::gather_if_needed_async(&value)
        .await
        .map_err(|error| {
            optimset_error_builder(
                &OPTIMSET_ERROR_INVALID_INPUT,
                format!("failed to gather option value: {error}"),
            )
            .with_source(error)
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
        })
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
        let value = block_on(optimset_builtin(vec![
            Value::from("TolX"),
            Value::Num(1.0e-8),
            Value::from("Display"),
            Value::from("off"),
        ]))
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
        let err = block_on(optimset_builtin(vec![Value::from("TolX")])).unwrap_err();
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
            let error = block_on(optimset_builtin(arguments)).unwrap_err();
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

        let error = block_on(optimset_builtin(vec![
            Value::from("MaxIter"),
            Value::Tensor(value),
        ]))
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

        let result = block_on(optimset_builtin(vec![
            Value::from("MaxIter"),
            Value::Tensor(value),
        ]))
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

            let automatic_result = block_on(optimset_builtin(vec![
                Value::from("TolX"),
                Value::GpuTensor(automatic),
            ]))
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
            let error = block_on(optimset_builtin(vec![
                Value::from("TolX"),
                Value::GpuTensor(explicit.clone()),
            ]))
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
            let explicit_result = block_on(optimset_builtin(vec![
                Value::from("TolX"),
                Value::GpuTensor(explicit),
                Value::from("MaxFunEvals"),
                Value::GpuTensor(mixed_automatic),
            ]))
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
        });
    }
}
