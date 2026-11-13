//! Fence legacy calendar scratch until it has a complete finite owner.
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
    HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;

#[path = "allocation/datenum.rs"]
pub(super) mod datenum;
#[path = "allocation/dateshift.rs"]
pub(super) mod dateshift;
#[path = "allocation/numeric_view.rs"]
pub(super) mod numeric_view;

/// Publish the six clock components directly into the final owned payload.
/// The fixed-size calendar calculation stays on the stack; only the returned
/// tensor and its shape retain a managed allocation claim.
pub(super) fn clock_components_output(
    runtime: Option<&RuntimeContext>,
    components: [f64; 6],
) -> BuiltinResult<runmat_value::Value> {
    let failure = |source| {
        let mut error = host_allocation_runtime_error(
            runtime,
            "RunMat:clock:AllocationFailed",
            "clock: component output allocation failed",
            source,
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    };
    let shape = [1, 6];
    let bytes = HostAllocationPlan::retained_tensor_bytes(runmat_value::NumericDType::F64, &shape)
        .map_err(failure)?;
    let plan = HostAllocationPlan::new(0, bytes).map_err(failure)?;
    let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(failure)?;
    let tensor = if let Some(receipt) = receipt {
        let runtime = runtime.ok_or_else(|| failure(HostAllocationError::StaleAuthority))?;
        let mut output = receipt
            .exact_output_tensor(runtime, runmat_value::NumericDType::F64, &shape)
            .map_err(failure)?;
        output
            .as_f64_slice_mut()
            .expect("clock output allocation is double")
            .copy_from_slice(&components);
        receipt.checkpoint(runtime).map_err(failure)?;
        let tensor = output
            .into_tensor()
            .map_err(|source| failure(source.into()))?;
        receipt.checkpoint(runtime).map_err(failure)?;
        tensor
    } else {
        // Admission observed unrestricted allocation and no suspension occurs
        // between that decision and this bounded, synchronous publication.
        runmat_value::Tensor::new(components.to_vec(), shape.to_vec())
            .map_err(|source| super::datetime_error(format!("clock: {source}")))?
    };
    Ok(runmat_value::Value::Tensor(tensor))
}

/// This retains the invocation context, not an allocation grant. Every policy
/// decision comes from the canonical host allocation owner. An initially
/// unbounded call must still be unbounded after each suspension and at publish.
pub(super) struct UnboundedCalendarWork<'a> {
    runtime: Option<&'a RuntimeContext>,
    builtin: &'static str,
}

impl<'a> UnboundedCalendarWork<'a> {
    pub(super) fn admit(
        runtime: Option<&'a RuntimeContext>,
        builtin: &'static str,
    ) -> BuiltinResult<Self> {
        let work = Self { runtime, builtin };
        work.check()?;
        Ok(work)
    }

    pub(super) fn check(&self) -> BuiltinResult<()> {
        let source = match HostAllocationReceipt::policy(self.runtime) {
            Ok(HostAllocationPolicy::Unbounded) => return Ok(()),
            Ok(HostAllocationPolicy::Managed) => HostAllocationError::MissingManagedOwner,
            Err(source) => source,
        };
        let mut error = host_allocation_runtime_error(
            self.runtime,
            "RunMat:datetime:CalendarWorkspace",
            format!(
                "{}: calendar workspace requires unrestricted allocation; {source}",
                self.builtin
            ),
            source,
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        Err(error)
    }

    /// The producer is evaluated before this call, so even a synchronous
    /// construction observes the current authority again before publication.
    pub(super) fn finish<T>(&self, result: BuiltinResult<T>) -> BuiltinResult<T> {
        let output = result?;
        self.check()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use crate::execution::{ExecutionServiceError, RuntimeExecutionService};
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::cell::Cell;
    use std::error::Error;
    use std::rc::Rc;

    struct CalendarPolicy {
        finite: Cell<bool>,
        ledger: ManagedHostEffectLedger,
    }
    impl RuntimeParallelService for CalendarPolicy {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let snapshot = self.ledger.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: self.finite.get().then_some(snapshot.free_bytes()),
                epoch: snapshot.epoch,
                ..RuntimeParallelResources::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.ledger.clone())
        }
    }

    #[test]
    fn legacy_unbounded_work_refuses_after_policy_upgrade_without_publishing() {
        let ledger = ManagedHostEffectLedger::new(65_536);
        let policy = Rc::new(CalendarPolicy {
            finite: Cell::new(false),
            ledger: ledger.clone(),
        });
        let source = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            source.token(),
        )
        .with_service_ports(RuntimeServicePorts::default().with_parallel(policy.clone()));
        let work = UnboundedCalendarWork::admit(Some(&runtime), "dateshift").unwrap();
        assert_eq!(work.finish(Ok(42_u8)).unwrap(), 42);
        let before = ledger.snapshot().free_bytes();
        policy.finite.set(true);
        let error = work.check().unwrap_err();
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(error
            .message()
            .contains("dateshift: calendar workspace requires unrestricted allocation;"));
        let publication_error = work.finish(Ok(42_u8)).unwrap_err();
        assert_eq!(
            publication_error.gpu_gather_retry,
            crate::GpuGatherRetry::Never
        );
        assert!(matches!(
            error
                .source()
                .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
            Some(ExecutionServiceError::Infrastructure(_))
        ));
        // Selected component and dateshift operations now have their own
        // finite receipts; this test exercises only an already-issued legacy
        // work token when its policy is upgraded.
        assert_eq!(ledger.snapshot().free_bytes(), before);
        policy.finite.set(false);
        source.cancel(CancellationReason::User);
        let cancelled = work.finish(Ok(42_u8)).unwrap_err();
        assert_eq!(cancelled.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(matches!(
            cancelled
                .source()
                .and_then(|cause| cause.downcast_ref::<ExecutionServiceError>()),
            Some(ExecutionServiceError::Cancelled)
        ));
    }

    #[test]
    fn stored_second_managed_scalar_empty_native_geometry_and_output_lifetime() {
        let ledger = ManagedHostEffectLedger::new(1_048_576);
        let policy = Rc::new(CalendarPolicy {
            finite: Cell::new(true),
            ledger: ledger.clone(),
        });
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_service_ports(RuntimeServicePorts::default().with_parallel(policy));
        let initial = ledger.snapshot().free_bytes();
        let clock =
            clock_components_output(Some(&runtime), [2026.0, 10.0, 21.0, 1.0, 2.0, 3.0]).unwrap();
        let runmat_value::Value::Tensor(ref tensor) = clock else {
            panic!("clock component tensor")
        };
        assert_eq!(tensor.shape(), &[1, 6]);
        assert_eq!(
            tensor.as_f64_slice().unwrap(),
            &[2026.0, 10.0, 21.0, 1.0, 2.0, 3.0]
        );
        let quote =
            HostAllocationPlan::retained_tensor_bytes(runmat_value::NumericDType::F64, &[1, 6])
                .unwrap();
        assert_eq!(initial - ledger.snapshot().free_bytes(), quote);
        let retained = clock.clone();
        drop(clock);
        assert_eq!(initial - ledger.snapshot().free_bytes(), quote);
        drop(retained);
        assert_eq!(initial, ledger.snapshot().free_bytes());
        let scalar_fields = [(
            runmat_builtins::DATETIME_SERIAL_PROPERTY,
            runmat_value::Value::Num(719_529.0),
        )];
        let scalar = crate::execution::object::construct_object_from_borrowed_fields(
            None,
            runmat_types::standard::DATETIME.display_name(),
            scalar_fields.iter().map(|(key, value)| (*key, value)),
        )
        .unwrap()
        .into_value_and_lease()
        .0;
        let before = ledger.snapshot().free_bytes();
        let result = futures::executor::block_on(super::super::second_builtin::second_builtin(
            Some(runtime.clone()),
            scalar,
        ))
        .unwrap();
        assert_eq!(result, runmat_value::Value::Num(0.0));
        assert_eq!(ledger.snapshot().free_bytes(), before);
        for shape in [vec![0, 2, 1], vec![2, 1, 3]] {
            let len: usize = shape.iter().product();
            let source =
                runmat_value::Tensor::from_f32(vec![719_529.0; len], shape.clone()).unwrap();
            let input =
                super::super::datetime_object_from_serial_tensor(None, source, "yyyy-MM-dd")
                    .unwrap();
            let before = ledger.snapshot().free_bytes();
            let quote = crate::context::host_allocation::HostAllocationPlan::retained_tensor_bytes(
                runmat_value::NumericDType::F64,
                &shape,
            )
            .unwrap();
            let result = futures::executor::block_on(super::super::second_builtin::second_builtin(
                Some(runtime.clone()),
                input,
            ))
            .unwrap();
            let runmat_value::Value::Tensor(tensor) = &result else {
                panic!("nonscalar component tensor")
            };
            assert_eq!(tensor.shape(), shape);
            assert_eq!(tensor.as_f64_slice().unwrap(), vec![0.0; len]);
            assert_eq!(before - ledger.snapshot().free_bytes(), quote);
            let shared = result.clone();
            drop(result);
            assert_eq!(before - ledger.snapshot().free_bytes(), quote);
            drop(shared);
            assert_eq!(ledger.snapshot().free_bytes(), before);
        }
        for serial in [f64::NAN, f64::INFINITY] {
            let input = super::super::datetime_object_from_serials(
                None,
                vec![serial],
                vec![1, 1],
                "yyyy-MM-dd",
            )
            .unwrap();
            assert!(
                futures::executor::block_on(super::super::second_builtin::second_builtin(
                    Some(runtime.clone()),
                    input
                ))
                .is_err()
            );
            assert_eq!(ledger.snapshot().free_bytes(), before);
        }
    }
}
