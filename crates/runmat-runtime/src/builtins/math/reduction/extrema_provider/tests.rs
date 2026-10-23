use super::*;
use runmat_accelerate_api::{
    test_support::TestAllocationOwner, AccelDownloadFuture, GpuTensorStorage, HostTensorOwned,
    HostTensorView, NumericElementType, ReduceDimResult, ThreadProviderGuard,
};
use std::sync::atomic::{AtomicUsize, Ordering};

struct SelectedExtremaPlacement {
    selections: AtomicUsize,
    feedback: std::cell::RefCell<Vec<runmat_execution::PlacementFeedback>>,
}

impl crate::context::RuntimePlacementService for SelectedExtremaPlacement {
    fn whole_work_revision(
        &self,
        program: Option<runmat_execution::ProgramRevision>,
        provider: runmat_execution::Digest,
    ) -> Option<runmat_execution::PlacementRevision> {
        let revision = runmat_execution::Digest::sha256(b"selected-extrema-submission");
        Some(runmat_execution::PlacementRevision {
            program,
            catalog: revision,
            compiler: revision,
            authority: runmat_execution::PlacementExecutionAuthorityRevision::Provider(provider),
            policy: revision,
        })
    }

    fn select_whole_work(
        &self,
        request: runmat_execution::WholeWorkPlacementRequest,
    ) -> Result<crate::context::RuntimeWholeWorkSelection, crate::RuntimeError> {
        request.validate().expect("typed extrema placement request");
        let candidate = request.candidates.first().expect("one provider candidate");
        self.selections.fetch_add(1, Ordering::SeqCst);
        Ok(crate::context::RuntimeWholeWorkSelection {
            decision: runmat_execution::WholeWorkDecision {
                signature: request.signature.clone(),
                selected: candidate.selected(&request.signature.operation),
                kind: runmat_execution::WholeWorkDecisionKind::Unmeasured,
                predicted_total_ns: None,
            },
            probe: None,
            observation_permit: None,
        })
    }

    fn plan(
        &self,
        _: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, crate::RuntimeError> {
        panic!("selected extrema uses whole-work placement")
    }

    fn observe(
        &self,
        feedback: runmat_execution::PlacementFeedback,
    ) -> Result<(), crate::RuntimeError> {
        self.feedback.borrow_mut().push(feedback);
        Ok(())
    }

    fn invalidate(&self, _: runmat_execution::PlacementInvalidation) {}
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn scalar_extrema_use_native_single_storage_selection() {
    use runmat_value::{ComplexStorage, ComplexTensor, NumericStorage};

    for (name, expected) in [("min", 1.25), ("max", 3.5)] {
        for shape in [vec![1, 1], vec![1, 1, 1]] {
            let left = Value::Tensor(Tensor::from_f32(vec![3.5], shape.clone()).unwrap());
            let right = Value::Tensor(Tensor::from_f32(vec![1.25], shape.clone()).unwrap());
            let result = futures::executor::block_on(crate::call_builtin_async(
                name,
                &[left.clone(), right],
            ))
            .unwrap();
            let Value::Tensor(result) = result.value() else {
                panic!("{name}: Single scalar must retain its native carrier");
            };
            assert_eq!(result.shape(), shape);
            assert_eq!(
                result.clone().into_numeric_storage().unwrap(),
                NumericStorage::F32(vec![expected])
            );

            // Preserve the existing mixed-class policy, in both operand orders.
            for args in [[left.clone(), Value::Num(1.25)], [Value::Num(1.25), left]] {
                assert_eq!(
                    futures::executor::block_on(crate::call_builtin_async(name, &args)).unwrap(),
                    Value::Num(f64::from(expected))
                );
            }

            // Zero imaginary parts still belong to the complex Single carrier.
            let args = [
                Value::ComplexTensor(
                    ComplexTensor::from_f32(vec![(3.5, 0.0)], shape.clone()).unwrap(),
                ),
                Value::ComplexTensor(
                    ComplexTensor::from_f32(vec![(1.25, 0.0)], shape.clone()).unwrap(),
                ),
            ];
            let result =
                futures::executor::block_on(crate::call_builtin_async(name, &args)).unwrap();
            let Value::ComplexTensor(result) = result.value() else {
                panic!("{name}: complex Single scalar must retain its native carrier");
            };
            assert_eq!(result.shape(), shape);
            assert_eq!(
                result.clone().into_complex_storage(),
                ComplexStorage::F32(vec![(expected, 0.0)].into())
            );
        }
    }
}

#[derive(Clone, Copy)]
enum Fault {
    Execution,
    Unsupported,
    UnsupportedFillAfterAttempt,
    UnsupportedAfterSubmission,
    Download,
}

struct Provider {
    owner: TestAllocationOwner,
    fault: Fault,
    resources: runmat_execution::ProviderResourceSnapshot,
    reductions: AtomicUsize,
    downloads: AtomicUsize,
}

impl Provider {
    fn selected_extrema_estimate(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> Option<runmat_accelerate_api::ProviderResourceEstimate> {
        use runmat_accelerate_api::{
            ProviderElementType, ProviderExtremaDemand, ProviderExtremaKind, ProviderLayout,
            ProviderOperationFamily, ProviderResidency, ProviderStorage,
        };
        if !matches!(self.fault, Fault::UnsupportedAfterSubmission) {
            return None;
        }
        let route = query.workload.extrema.as_ref()?;
        let [input] = query.inputs.as_slice() else {
            return None;
        };
        let [output] = query.outputs.as_slice() else {
            return None;
        };
        let identity = match route.kind {
            ProviderExtremaKind::Minimum => runmat_builtins::MIN_CATALOG_ENTRY.identity.name,
            ProviderExtremaKind::Maximum => runmat_builtins::MAX_CATALOG_ENTRY.identity.name,
        };
        if query.operation.0 != identity
            || query.family != ProviderOperationFamily::Reduction
            || query.workload.requested_outputs != Some(1)
            || route.demand != ProviderExtremaDemand::ValuesOnly
            || route.axis != 0
            || input.shape != [2, 3]
            || input.element_type != ProviderElementType::F64
            || input.layout != ProviderLayout::ColumnMajorContiguous
            || input.storage != ProviderStorage::DenseReal
            || input.residency != ProviderResidency::Device
        {
            return None;
        }
        let backing = route
            .input_backing
            .filter(|backing| backing.matches_representation(input))?;
        let backing_bytes = backing.live_requested_bytes()?;
        if input.checked_byte_len()? != backing_bytes {
            return None;
        }
        let mut expected_output = input.clone();
        expected_output.shape[0] = 1;
        if output != &expected_output {
            return None;
        }
        let values = output.checked_byte_len()?;
        // The dimensional hook keeps its borrowed input, both F64 outputs
        // (including discarded indices), and its 16-byte uniform live.
        Some(runmat_accelerate_api::ProviderResourceEstimate {
            transient_bytes: Some(
                backing_bytes
                    .checked_add(values.checked_mul(2)?)?
                    .checked_add(16)?,
            ),
            output_bytes: Some(values),
            dispatches: Some(1),
        })
    }

    fn reject_execution<T>(&self) -> runmat_accelerate_api::ProviderExecutionResult<T> {
        self.reject().map_err(|error| {
            let error = runmat_accelerate_api::ProviderExecutionError::from(error);
            if matches!(self.fault, Fault::UnsupportedAfterSubmission) {
                error.after_effect()
            } else {
                error
            }
        })
    }

    fn reject<T>(&self) -> anyhow::Result<T> {
        self.reductions.fetch_add(1, Ordering::SeqCst);
        match self.fault {
            Fault::Execution => Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "unsupported transport state is not a capability rejection",
            )
            .into()),
            Fault::UnsupportedAfterSubmission => {
                runmat_accelerate_api::observation::current_provider_observation()
                    .expect("selected extrema observation")
                    .submission()
                    .expect("selected provider submission")
                    .complete();
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "dimensional reduction failed after a native submission",
                ))
            }
            Fault::Unsupported | Fault::UnsupportedFillAfterAttempt | Fault::Download => {
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "dimensional reduction unavailable",
                ))
            }
        }
    }
}

impl AccelProvider for Provider {
    fn placement_resources(&self) -> runmat_execution::ProviderResourceSnapshot {
        runmat_execution::ProviderResourceSnapshot {
            device_id: self.device_id(),
            ..self.resources.clone()
        }
    }

    fn query_feasibility(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::ProviderFeasibility {
        if let Some(estimate) = self.selected_extrema_estimate(query) {
            return runmat_accelerate_api::ProviderFeasibility::supported(estimate);
        }
        runmat_accelerate_api::ProviderFeasibility::rejected(
            runmat_accelerate_api::ProviderRejectionCode::UnsupportedOperation,
            "fixture.unselected_route",
        )
    }

    fn operation_observation_capability(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::observation::ProviderObservationCapability {
        use runmat_accelerate_api::observation::{
            ProviderObservationCapability, ProviderObservationOverhead,
        };
        if self.selected_extrema_estimate(query).is_some() {
            ProviderObservationCapability::Supported(ProviderObservationOverhead {
                submissions: 1,
                requested_peak_buffer_bytes: 0,
                clear_commands: 0,
                map_callbacks: 0,
                wgpu_compute_query: None,
            })
        } else {
            ProviderObservationCapability::Unsupported
        }
    }

    fn begin_operation_observation(
        &self,
        query: &runmat_accelerate_api::ProviderFeasibilityQuery,
    ) -> runmat_accelerate_api::observation::ProviderObservationSupport {
        use runmat_accelerate_api::observation::{
            ProviderObservationSupport, ProviderOperationReceipt,
        };
        if self.selected_extrema_estimate(query).is_some() {
            ProviderObservationSupport::Observed(ProviderOperationReceipt::new())
        } else {
            ProviderObservationSupport::Unsupported
        }
    }

    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "extrema execution authority fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("unexpected restoration")
    }
    fn download<'a>(&'a self, input: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        Box::pin(async move {
            self.downloads.fetch_add(1, Ordering::SeqCst);
            if matches!(self.fault, Fault::Download) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "download failed",
                )
                .into());
            }
            assert_eq!(input.device_id(), self.device_id());
            input.allocation_lease().validate_live()?;
            Ok(HostTensorOwned {
                data: vec![0.0; input.shape().iter().product()],
                shape: input.shape().to_vec(),
                storage: input.physical().storage(),
            })
        })
    }
    fn reduce_min_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.reject_execution() })
    }
    fn reduce_max_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, ReduceDimResult> {
        Box::pin(async move { self.reject_execution() })
    }

    fn elem_min<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reject_execution() })
    }
    fn elem_max<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: &'a GpuTensorHandle,
    ) -> runmat_accelerate_api::ProviderExecutionFuture<'a, GpuTensorHandle> {
        Box::pin(async move { self.reject_execution() })
    }
    fn fill_like(
        &self,
        _: &GpuTensorHandle,
        _: f64,
    ) -> runmat_accelerate_api::ProviderExecutionResult<Option<GpuTensorHandle>> {
        // A pure optional-constructor decline is not an attempted failure.
        // Keep the separate attempted Unsupported error below terminal.
        if matches!(self.fault, Fault::Unsupported) {
            return Ok(None);
        }
        let result: anyhow::Result<GpuTensorHandle> = self.reject();
        result.map(Some).map_err(|source| {
            runmat_accelerate_api::ProviderExecutionError::from(source).with_unproven_effects()
        })
    }
    fn scalar_min(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.reject()
    }
    fn scalar_max(&self, _: &GpuTensorHandle, _: f64) -> anyhow::Result<GpuTensorHandle> {
        self.reject()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dimensional_extrema_only_gather_after_typed_unsupported() {
    for name in ["min", "max"] {
        for fault in [Fault::Execution, Fault::Unsupported, Fault::Download] {
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                fault,
                resources: Default::default(),
                reductions: AtomicUsize::new(0),
                downloads: AtomicUsize::new(0),
            }));
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let result = futures::executor::block_on(call_extrema_owner(
                name,
                &[Value::GpuTensor(input.handle.clone())],
            ));
            assert_eq!(provider.reductions.load(Ordering::SeqCst), 1);
            match fault {
                Fault::Unsupported => assert_eq!(result.unwrap(), Value::Num(0.0)),
                Fault::Execution
                | Fault::Download
                | Fault::UnsupportedAfterSubmission
                | Fault::UnsupportedFillAfterAttempt => {
                    let error = result.unwrap_err();
                    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                    let mut source = std::error::Error::source(&error);
                    let mut retained = None;
                    while let Some(error) = source {
                        if let Some(error) = error.downcast_ref::<std::io::Error>() {
                            retained = Some(error.kind());
                        }
                        source = error.source();
                    }
                    assert_eq!(
                        retained,
                        Some(if matches!(fault, Fault::Execution) {
                            std::io::ErrorKind::BrokenPipe
                        } else {
                            std::io::ErrorKind::ConnectionReset
                        }),
                        "original {name} provider failure must survive"
                    );
                }
            }
            assert_eq!(
                provider.downloads.load(Ordering::SeqCst),
                usize::from(!matches!(fault, Fault::Execution))
            );
            assert_eq!(input.probe.retirements(), 0);
            input.handle.allocation_lease().validate_live().unwrap();
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }

        // An unsupported feasibility quote is not an unsupported hook result:
        // the ordinary caller would still execute that hook after no receipt.
        // Missing bounds cannot authorize it under a finite or invalid snapshot.
        for resources in [
            runmat_execution::ProviderResourceSnapshot {
                capacity_bytes: Some(1024),
                ..Default::default()
            },
            runmat_execution::ProviderResourceSnapshot {
                queue_depth: Some(0),
                queue_limit: Some(2),
                ..Default::default()
            },
            runmat_execution::ProviderResourceSnapshot {
                lost: true,
                ..Default::default()
            },
            runmat_execution::ProviderResourceSnapshot {
                queue_depth: Some(0),
                ..Default::default()
            },
        ] {
            let expected = if resources.lost {
                "RunMat:Placement:NoLegalCandidate"
            } else if resources.queue_depth.is_some() && resources.queue_limit.is_none() {
                "RunMat:Placement:InvalidRequest"
            } else {
                "RunMat:Placement:InsufficientResourceEvidence"
            };
            let provider = Box::leak(Box::new(Provider {
                owner: TestAllocationOwner::new(),
                fault: Fault::Unsupported,
                resources,
                reductions: AtomicUsize::new(0),
                downloads: AtomicUsize::new(0),
            }));
            let _selected = ThreadProviderGuard::set(Some(provider));
            let input = provider
                .owner
                .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap();
            let result = futures::executor::block_on(if name == "min" {
                ExtremaKind::Min.reduce_dim(None, &input.handle, 0, ExtremaDemand::ValuesOnly)
            } else {
                ExtremaKind::Max.reduce_dim(None, &input.handle, 0, ExtremaDemand::ValuesOnly)
            });
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("unbounded hook cannot bypass physical admission"),
            };
            assert_eq!(error.identifier(), Some(expected));
            assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
            assert_eq!(provider.reductions.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            // A normal ND source has no direct two-dimensional backing plan.
            // Its known owner must not receive an unadmitted hook either.
            let nd = provider
                .owner
                .allocate_zeroed(
                    vec![2, 1, 1],
                    NumericElementType::F64,
                    GpuTensorStorage::Real,
                )
                .unwrap();
            let error = match futures::executor::block_on(ExtremaKind::Min.reduce_dim(
                None,
                &nd.handle,
                0,
                ExtremaDemand::ValuesOnly,
            )) {
                Err(error) => error,
                Ok(_) => panic!("missing backing is not resource admission"),
            };
            assert_eq!(error.identifier(), Some(expected));
            assert_eq!(provider.reductions.load(Ordering::SeqCst), 0);
            assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
            nd.handle.allocation_lease().validate_live().unwrap();
            drop(nd.handle);
            assert_eq!(nd.probe.retirements(), 1);
            input.handle.allocation_lease().validate_live().unwrap();
            assert_eq!(input.probe.retirements(), 0);
            drop(input.handle);
            assert_eq!(input.probe.retirements(), 1);
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn selected_extrema_unsupported_after_submission_retains_cause_without_retry() {
    use std::rc::Rc;

    let provider = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        fault: Fault::UnsupportedAfterSubmission,
        resources: Default::default(),
        reductions: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let placement = Rc::new(SelectedExtremaPlacement {
        selections: AtomicUsize::new(0),
        feedback: std::cell::RefCell::new(Vec::new()),
    });
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ))
    .with_service_ports(
        crate::context::RuntimeServicePorts::default().with_placement(placement.clone()),
    );
    let input = provider
        .owner
        .allocate_zeroed(vec![2, 3], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let error = match futures::executor::block_on(ExtremaKind::Min.reduce_dim(
        Some(&runtime),
        &input.handle,
        0,
        ExtremaDemand::ValuesOnly,
    )) {
        Err(error) => error,
        Ok(_) => panic!("submission cannot be retried as an unsupported capability"),
    };
    assert_eq!(placement.selections.load(Ordering::SeqCst), 1);
    let feedback = placement.feedback.borrow();
    assert_eq!(feedback.len(), 1);
    assert!(!feedback[0].succeeded);
    assert_eq!(provider.reductions.load(Ordering::SeqCst), 1);
    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
    assert_eq!(
        error.identifier(),
        runmat_builtins::MIN_ERROR_INTERNAL.identifier
    );
    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
    let retained = std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<runmat_accelerate_api::ProviderExecutionError>())
        .expect("post-submission error retains typed provider execution owner");
    assert!(!retained.is_unsupported());
    let mut source = std::error::Error::source(retained);
    let mut original = false;
    while let Some(cause) = source {
        original |= cause.is::<runmat_accelerate_api::UnsupportedProviderOperation>();
        source = cause.source();
    }
    assert!(
        original,
        "the original typed unsupported marker remains nested"
    );
    input.handle.allocation_lease().validate_live().unwrap();
    assert_eq!(input.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dimensional_extrema_never_dispatch_unregistered_input_to_selected_provider() {
    let selected = Box::leak(Box::new(Provider {
        owner: TestAllocationOwner::new(),
        fault: Fault::Execution,
        resources: Default::default(),
        reductions: AtomicUsize::new(0),
        downloads: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(selected));
    let input = TestAllocationOwner::new()
        .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    for name in ["min", "max"] {
        let error = futures::executor::block_on(crate::call_builtin_async(
            name,
            &[Value::GpuTensor(input.handle.clone())],
        ))
        .unwrap_err();
        assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
        assert!(error.message().contains("exact owning provider"));
    }
    assert_eq!(selected.reductions.load(Ordering::SeqCst), 0);
    assert_eq!(selected.downloads.load(Ordering::SeqCst), 0);
    input.handle.allocation_lease().validate_live().unwrap();
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn elementwise_extrema_propagate_execution_failure_and_keep_capability_fallback() {
    for name in ["min", "max"] {
        for form in 0..3 {
            for fault in [
                Fault::Execution,
                Fault::Unsupported,
                Fault::UnsupportedFillAfterAttempt,
            ] {
                let provider = Box::leak(Box::new(Provider {
                    owner: TestAllocationOwner::new(),
                    fault,
                    resources: Default::default(),
                    reductions: AtomicUsize::new(0),
                    downloads: AtomicUsize::new(0),
                }));
                let _selected = ThreadProviderGuard::set(Some(provider));
                let left = provider
                    .owner
                    .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                    .unwrap();
                let right = provider
                    .owner
                    .allocate_zeroed(vec![2, 1], NumericElementType::F64, GpuTensorStorage::Real)
                    .unwrap();
                let args = match form {
                    0 => vec![
                        Value::GpuTensor(left.handle.clone()),
                        Value::GpuTensor(right.handle.clone()),
                    ],
                    1 => vec![Value::GpuTensor(left.handle.clone()), Value::Num(2.0)],
                    _ => vec![Value::Num(2.0), Value::GpuTensor(right.handle.clone())],
                };
                let result = futures::executor::block_on(call_extrema_owner(name, &args));
                let attempted_fill = matches!(fault, Fault::UnsupportedFillAfterAttempt)
                    && name == "max"
                    && form != 0;
                if matches!(fault, Fault::Execution) || attempted_fill {
                    let error = result.unwrap_err();
                    assert_eq!(error.gpu_gather_retry, GpuGatherRetry::Never);
                    let source = error.source.as_ref().unwrap();
                    let source = source
                        .downcast_ref::<runmat_accelerate_api::ProviderExecutionError>()
                        .unwrap();
                    if attempted_fill {
                        assert!(
                            !source.is_unsupported(),
                            "an attempted fill is not a capability decline"
                        );
                        assert!(std::error::Error::source(source)
                            .unwrap()
                            .is::<runmat_accelerate_api::UnsupportedProviderOperation>());
                    } else {
                        assert_eq!(
                            std::error::Error::source(source)
                                .unwrap()
                                .downcast_ref::<std::io::Error>()
                                .unwrap()
                                .kind(),
                            std::io::ErrorKind::BrokenPipe,
                        );
                    }
                    assert_eq!(provider.reductions.load(Ordering::SeqCst), 1);
                    assert_eq!(provider.downloads.load(Ordering::SeqCst), 0);
                } else {
                    let Value::Tensor(output) = result.unwrap() else {
                        panic!("host capability fallback");
                    };
                    assert_eq!(output.shape(), &[2, 1]);
                    let expected = if name == "max" && form != 0 { 2.0 } else { 0.0 };
                    assert_eq!(output.materialize_f64(), vec![expected, expected]);
                    assert_eq!(
                        provider.downloads.load(Ordering::SeqCst),
                        if form == 0 { 2 } else { 1 }
                    );
                    assert_eq!(provider.reductions.load(Ordering::SeqCst), 1);
                }
                drop(args);
                for input in [left, right] {
                    assert_eq!(input.probe.retirements(), 0);
                    input.handle.allocation_lease().validate_live().unwrap();
                    drop(input.handle);
                    assert_eq!(input.probe.retirements(), 1);
                }
            }
        }
    }
}

// Runtime unit tests also link the normal runtime through the acceleration
// dev-dependency, so legacy inventory contains both copies. Test these domain
// owners directly; provider_execution_sources separately checks public dispatch
// in a standalone executable with a single runtime inventory.
async fn call_extrema_owner(name: &str, args: &[Value]) -> crate::BuiltinResult<Value> {
    let (value, rest) = args.split_first().expect("extrema data operand");
    match name {
        "min" => {
            crate::builtins::math::reduction::min::min_builtin(
                None,
                value.clone(),
                crate::builtins::common::test_support::builtin_arguments(rest.to_vec()),
            )
            .await
        }
        "max" => {
            crate::builtins::math::reduction::max::max_builtin(
                None,
                value.clone(),
                crate::builtins::common::test_support::builtin_arguments(rest.to_vec()),
            )
            .await
        }
        _ => unreachable!("fixture covers min and max"),
    }
}
