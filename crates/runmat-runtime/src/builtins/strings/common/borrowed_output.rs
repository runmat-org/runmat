//! Exact retained output for text transforms that select a borrowed substring.

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{
    exact_capacity_vec, ExactArrayInitializationError, ExactArrayStorage, SharedShape, StringArray,
    StringElement, Utf8Text,
};

const CHECKPOINT_CHUNK: usize = 4096;

pub(crate) fn transform_string_array_borrowed(
    runtime: Option<&RuntimeContext>,
    input: StringArray,
    select: impl for<'a> Fn(&'a str) -> &'a str,
    admission_error: impl Fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
    invalid: impl Fn(String) -> RuntimeError,
) -> BuiltinResult<StringArray> {
    transform_string_array_borrowed_indexed(
        runtime,
        input,
        |_, text| select(text),
        admission_error,
        invalid,
    )
}

pub(crate) fn transform_string_array_borrowed_indexed(
    runtime: Option<&RuntimeContext>,
    input: StringArray,
    select: impl for<'a> Fn(usize, &'a str) -> &'a str,
    admission_error: impl Fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
    invalid: impl Fn(String) -> RuntimeError,
) -> BuiltinResult<StringArray> {
    let ordinary = || -> BuiltinResult<StringArray> {
        let values = input
            .data()
            .iter()
            .enumerate()
            .map(|(index, text)| match text.as_str() {
                Some(text) => StringElement::text(select(index, text).to_owned()),
                None => StringElement::Missing,
            })
            .collect();
        StringArray::from_elements_unbounded(values, input.shape().to_vec()).map_err(&invalid)
    };
    let Some(runtime) = runtime else {
        return ordinary();
    };

    let reuse_shape = input.shape.is_admitted();
    let mut physical = ExactArrayStorage::<StringElement>::requested_bytes(input.data().len())
        .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?;
    if !reuse_shape {
        physical = physical
            .checked_add(
                SharedShape::requested_bytes_for_rank(input.shape.len()).map_err(|cause| {
                    admission_error(runtime, HostAllocationError::Numeric(cause))
                })?,
            )
            .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
    }
    let mut claims = usize::from(!input.data().is_empty()) + usize::from(!reuse_shape);
    for (index, text) in input.data().iter().enumerate() {
        if index % CHECKPOINT_CHUNK == 0 && runtime.is_cancelled() {
            return Err(admission_error(runtime, HostAllocationError::Cancelled));
        }
        let Some(source) = text.as_str() else {
            continue;
        };
        let selected = select(index, source);
        if selected == source && text.has_retained_claim() {
            continue;
        }
        physical = physical
            .checked_add(
                Utf8Text::requested_bytes_for_capacity(selected.len()).map_err(|cause| {
                    admission_error(runtime, HostAllocationError::Numeric(cause))
                })?,
            )
            .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
        claims = claims
            .checked_add(usize::from(!selected.is_empty()))
            .ok_or_else(|| admission_error(runtime, HostAllocationError::Overflow))?;
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical)
            .map_err(|_| admission_error(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|cause| admission_error(runtime, cause))?;
    let plan =
        HostAllocationPlan::new(0, retained).map_err(|cause| admission_error(runtime, cause))?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|cause| admission_error(runtime, cause))?
    else {
        return ordinary();
    };

    let shape = if reuse_shape {
        input.shape.clone()
    } else {
        receipt
            .with_output_allocation(runtime, |permit| {
                SharedShape::new_admitted_from_slice(input.shape(), permit)
            })
            .map_err(|cause| admission_error(runtime, cause))?
    };
    let data = receipt
        .with_output_construction(runtime, |permit| {
            ExactArrayStorage::try_initialize(input.data().len(), permit, |index| {
                if index % CHECKPOINT_CHUNK == 0 {
                    receipt.checkpoint(runtime)?;
                }
                let text = &input.data()[index];
                let Some(source) = text.as_str() else {
                    return Ok(StringElement::Missing);
                };
                let selected = select(index, source);
                if selected == source && text.has_retained_claim() {
                    return Ok(text.clone());
                }
                let mut bytes = exact_capacity_vec::<u8>(selected.len())
                    .map_err(HostAllocationError::Numeric)?;
                bytes.extend_from_slice(selected.as_bytes());
                // SAFETY: selected is a borrowed UTF-8 substring and its bytes
                // move unchanged into this exact-capacity backing.
                let fresh = Utf8Text::new(unsafe { String::from_utf8_unchecked(bytes) });
                fresh
                    .try_admit_exact(permit)
                    .map(StringElement::Text)
                    .map_err(|failure| HostAllocationError::Numeric(failure.source()))
            })
        })
        .map_err(|cause| admission_error(runtime, cause))?
        .map_err(|cause| match cause {
            ExactArrayInitializationError::Allocation(cause) => {
                admission_error(runtime, HostAllocationError::Numeric(cause))
            }
            ExactArrayInitializationError::Element(cause) => admission_error(runtime, cause),
        })?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| admission_error(runtime, cause))?;
    let output = StringArray::from_admitted(data, shape).map_err(invalid)?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| admission_error(runtime, cause))?;
    Ok(output)
}

pub(crate) fn transform_scalar_borrowed(
    runtime: Option<&RuntimeContext>,
    input: StringElement,
    select: impl for<'a> Fn(&'a str) -> &'a str,
    admission_error: impl Fn(&RuntimeContext, HostAllocationError) -> RuntimeError,
) -> BuiltinResult<StringElement> {
    if let Some(runtime) = runtime {
        if runtime.is_cancelled() {
            return Err(admission_error(runtime, HostAllocationError::Cancelled));
        }
    }
    let Some(source) = input.as_str() else {
        return Ok(StringElement::Missing);
    };
    let selected = select(source);
    if selected == source && input.has_retained_claim() {
        return Ok(input);
    }
    let ordinary = || Ok(StringElement::text(select(source).to_owned()));
    let Some(runtime) = runtime else {
        return ordinary();
    };
    let physical = Utf8Text::requested_bytes_for_capacity(selected.len())
        .map_err(|cause| admission_error(runtime, HostAllocationError::Numeric(cause)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical)
            .map_err(|_| admission_error(runtime, HostAllocationError::Overflow))?,
        usize::from(!selected.is_empty()),
    )
    .map_err(|cause| admission_error(runtime, cause))?;
    let plan =
        HostAllocationPlan::new(0, retained).map_err(|cause| admission_error(runtime, cause))?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|cause| admission_error(runtime, cause))?
    else {
        return ordinary();
    };
    let output = receipt
        .with_output_construction(runtime, |permit| {
            let mut bytes =
                exact_capacity_vec::<u8>(selected.len()).map_err(HostAllocationError::Numeric)?;
            bytes.extend_from_slice(selected.as_bytes());
            // SAFETY: selected is valid UTF-8 borrowed from the source.
            Utf8Text::new(unsafe { String::from_utf8_unchecked(bytes) })
                .try_admit_exact(permit)
                .map_err(|failure| HostAllocationError::Numeric(failure.source()))
        })
        .map_err(|cause| admission_error(runtime, cause))?
        .map_err(|cause| admission_error(runtime, cause))?;
    receipt
        .checkpoint(runtime)
        .map_err(|cause| admission_error(runtime, cause))?;
    Ok(StringElement::Text(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use std::rc::Rc;

    struct ManagedParallel(ManagedHostEffectLedger);

    impl RuntimeParallelService for ManagedParallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }

        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }

        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn borrowed_substrings_hold_exact_claims_until_last_alias_and_refuse_before_output() {
        let source = StringArray::from_elements_unbounded(
            vec![
                StringElement::text("  text  "),
                StringElement::text("<missing>"),
                StringElement::text("<Missing>"),
                StringElement::text(""),
                StringElement::Missing,
            ],
            vec![1, 5],
        )
        .unwrap();
        let map_error = |_: &RuntimeContext, cause: HostAllocationError| {
            crate::build_runtime_error(format!("borrowed text: {cause}"))
                .with_source(cause)
                .build()
        };
        let invalid = |detail: String| crate::build_runtime_error(detail).build();
        let context = |ledger: &ManagedHostEffectLedger| {
            RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
                .with_service_ports(
                    RuntimeServicePorts::default()
                        .with_parallel(Rc::new(ManagedParallel(ledger.clone()))),
                )
        };
        let denied = ManagedHostEffectLedger::new(1);
        assert!(transform_string_array_borrowed(
            Some(&context(&denied)),
            source.clone(),
            |text| text.trim(),
            map_error,
            invalid,
        )
        .is_err());
        assert_eq!(denied.snapshot().live_reserved_bytes, 0);

        let ledger = ManagedHostEffectLedger::new(1 << 20);
        let output = transform_string_array_borrowed(
            Some(&context(&ledger)),
            source,
            |text| text.trim(),
            map_error,
            invalid,
        )
        .unwrap();
        assert_eq!(output.data()[0].as_str(), Some("text"));
        assert_eq!(output.data()[1].as_str(), Some("<missing>"));
        assert_eq!(output.data()[2].as_str(), Some("<Missing>"));
        assert_eq!(output.data()[3].as_str(), Some(""));
        assert!(output.data()[4].is_missing());
        assert!(!output.data()[4].has_retained_claim());
        let retained = ledger.snapshot().live_reserved_bytes;
        assert!(retained > 0);
        let alias = output.clone();
        drop(output);
        assert_eq!(ledger.snapshot().live_reserved_bytes, retained);
        drop(alias);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);

        let rank_zero = transform_string_array_borrowed(
            Some(&context(&ledger)),
            StringArray::new(vec!["  one  ".to_owned()], vec![]).unwrap(),
            |text| text.trim(),
            map_error,
            invalid,
        )
        .unwrap();
        assert!(rank_zero.shape().is_empty());
        assert_eq!(rank_zero.data()[0].as_str(), Some("one"));
        drop(rank_zero);
        assert_eq!(ledger.snapshot().live_reserved_bytes, 0);
    }
}
