use super::*;
use runmat_builtins::{
    BuiltinOperandPreparation, BuiltinPreparationOperation, BuiltinPreparationProviderRequirement,
};

/// Invocation-owned completion and borrowed host/provider preparation evidence.
#[derive(Default)]
pub(super) struct BuiltinPreparationContext<'a> {
    pub(super) runtime: Option<&'a RuntimeContext>,
    pub(super) completion: Option<runmat_runtime::context::preparation::PreparedCompletionRight>,
    pub(super) host: Option<(
        &'a crate::fusion_exec::host_readback::CpuHostResources,
        runmat_execution::Digest,
    )>,
    pub(super) compound: Option<&'a runmat_runtime::context::preparation::AdmittedBuiltinCompound>,
}

pub(super) enum ArgumentOperation {
    Reduction,
    AdmittedSum {
        requested_outputs: Option<u32>,
        route: runmat_accelerate_api::ProviderSumWorkload,
    },
    Binary(BinaryOp),
    ElementwiseNary {
        data_arity: usize,
        requested_outputs: Option<u32>,
    },
    Unary(UnaryOp),
    FloatingCast(ProviderPrecision),
    RealUnary(runmat_accelerate_api::NumericalDomainEvidence),
    RoundDigits(runmat_accelerate_api::NumericalDomainEvidence),
    FloatingRemainder(runmat_accelerate_api::NumericalDomainEvidence),
    RealPower(runmat_accelerate_api::NumericalDomainEvidence),
}

impl ArgumentOperation {
    fn preparation_program(
        &self,
    ) -> Option<runmat_accelerate_api::ProviderPreparationProgram<'static>> {
        match self {
            Self::FloatingCast(target) => Some(
                runmat_accelerate_api::ProviderPreparationProgram::FloatingCast { target: *target },
            ),
            Self::AdmittedSum { .. } => {
                Some(runmat_accelerate_api::ProviderPreparationProgram::Sum)
            }
            _ => None,
        }
    }
    fn for_call(name: &str, args: &[Value]) -> Option<Self> {
        if runmat_builtins::builtin_catalog_primary_entry_by_name(name).is_some_and(|entry| {
            matches!(entry.placement.preparation, runmat_builtins::BuiltinOperandPreparation::Forms(forms)
                if forms.iter().any(|form| form.operation == runmat_builtins::BuiltinPreparationOperation::Reduction(runmat_builtins::BuiltinPreparationReduction::Sum)))
        }) {
            // The selected sum binding prepares after its own option admission.
            // Legacy pre-call promotion cannot inspect those controls safely.
            return None;
        }
        let policy = builtin_policy(name)?;
        let has = |tag| policy.accel_tags.contains(&tag);
        if args.is_empty() {
            return None;
        }
        if has(AccelTag::Reduction) && !should_skip_reduction_promotion(name, args) {
            Some(Self::Reduction)
        } else if args.len() >= 2 && has(AccelTag::MatMul) {
            Some(Self::Binary(BinaryOp::MatMul))
        } else if args.len() >= 2 && has(AccelTag::Elementwise) {
            Some(Self::Binary(BinaryOp::Elementwise))
        } else if has(AccelTag::Transpose) {
            Some(Self::Unary(UnaryOp::Transpose))
        } else if has(AccelTag::Unary) {
            Some(Self::Unary(UnaryOp::Generic))
        } else {
            None
        }
    }

    fn count(&self) -> usize {
        match self {
            Self::Binary(_) | Self::RealPower(_) | Self::FloatingRemainder(_) => 2,
            Self::ElementwiseNary { data_arity, .. } => *data_arity,
            _ => 1,
        }
    }

    fn evaluate(
        &self,
        auto: &NativeAutoOffload,
        args: &[Value],
    ) -> Option<(
        DecisionEvaluation,
        ProviderOperationFamily,
        ProviderWorkload,
    )> {
        let elements = value_len(&args[0]).unwrap_or(0);
        let mut workload = ProviderWorkload {
            elements: u64::try_from(elements).ok(),
            ..Default::default()
        };
        let (mut evaluation, family) = match self {
            Self::ElementwiseNary {
                data_arity,
                requested_outputs,
            } => {
                let args = args.get(..*data_arity)?;
                let elements = super::nary::broadcast_elements(args)?;
                workload.elements = u64::try_from(elements).ok();
                workload.requested_outputs = *requested_outputs;
                let values = args.iter().collect::<Vec<_>>();
                (
                    auto.evaluate_elementwise(elements, &values),
                    ProviderOperationFamily::Elementwise,
                )
            }
            Self::Reduction | Self::AdmittedSum { .. } => (
                auto.evaluate_reduction(elements),
                ProviderOperationFamily::Reduction,
            ),
            Self::Unary(op) => (
                auto.evaluate_unary(elements, *op, &args[0]),
                ProviderOperationFamily::Elementwise,
            ),
            Self::FloatingCast(_) => (
                auto.evaluate_unary(elements, UnaryOp::Generic, &args[0]),
                ProviderOperationFamily::Elementwise,
            ),
            Self::RealUnary(evidence) | Self::RoundDigits(evidence) => {
                workload.numerical_domain = *evidence;
                (
                    auto.evaluate_unary(elements, UnaryOp::Generic, &args[0]),
                    ProviderOperationFamily::Elementwise,
                )
            }
            Self::Binary(BinaryOp::Elementwise)
            | Self::RealPower(_)
            | Self::FloatingRemainder(_) => {
                if let Self::RealPower(evidence) | Self::FloatingRemainder(evidence) = self {
                    workload.numerical_domain = *evidence;
                }
                let elements = element_count_pair(&args[0], &args[1]).unwrap_or(0);
                workload.elements = u64::try_from(elements).ok();
                (
                    auto.evaluate_elementwise(elements, &[&args[0], &args[1]]),
                    ProviderOperationFamily::Elementwise,
                )
            }
            Self::Binary(BinaryOp::MatMul) => {
                let (rows, inner) = tensor_rows_cols(&args[0])?;
                let (other_inner, cols) = tensor_rows_cols(&args[1])?;
                if inner != other_inner {
                    return None;
                }
                let flops = rows.checked_mul(inner)?.checked_mul(cols)?;
                workload.elements = None;
                workload.flops = u64::try_from(flops).ok();
                (
                    {
                        let mut evaluation = auto.evaluate_matmul(flops);
                        evaluation.cpu_secs = original_work::matrix(auto, rows, inner, cols);
                        evaluation
                    },
                    ProviderOperationFamily::MatrixMultiply,
                )
            }
        };
        if let Self::AdmittedSum {
            requested_outputs,
            route,
        } = self
        {
            evaluation.cpu_secs = original_work::sum(auto, &args[0], route);
            workload.requested_outputs = *requested_outputs;
            workload.sum = Some(route.clone());
        }
        Some((evaluation, family, workload))
    }
}

impl NativeAutoOffload {
    pub(super) fn prepare_call(
        &self,
        name: &str,
        args: &[Value],
        runtime: Option<&RuntimeContext>,
    ) -> ProviderExecutionResult<PreparedBuiltinCall> {
        let requires_double_provider = runmat_builtins::builtin_catalog_primary_entry_by_name(name)
            .is_some_and(|entry| {
                matches!(entry.placement.preparation, BuiltinOperandPreparation::Forms(forms)
                    if forms.iter().any(|form| {
                        form.operation == BuiltinPreparationOperation::FloatingCast(runmat_types::NumericClass::Double)
                            && form.provider_requirement == BuiltinPreparationProviderRequirement::F64
                    }))
            });
        if !self.enabled
            || (requires_double_provider && self.provider.precision() != ProviderPrecision::F64)
        {
            return Ok(PreparedBuiltinCall::unobserved(args.to_vec()));
        }
        let Some(operation) = ArgumentOperation::for_call(name, args) else {
            return Ok(PreparedBuiltinCall::unobserved(args.to_vec()));
        };
        self.prepare_operation(name, args, runtime, operation)
    }

    pub(super) fn prepare_operation(
        &self,
        name: &str,
        args: &[Value],
        runtime: Option<&RuntimeContext>,
        operation: ArgumentOperation,
    ) -> ProviderExecutionResult<PreparedBuiltinCall> {
        self.prepare_operation_with_completion(name, args, runtime, operation, None)
    }

    pub(super) fn prepare_operation_with_completion(
        &self,
        name: &str,
        args: &[Value],
        runtime: Option<&RuntimeContext>,
        operation: ArgumentOperation,
        completion: Option<runmat_runtime::context::preparation::PreparedCompletionRight>,
    ) -> ProviderExecutionResult<PreparedBuiltinCall> {
        self.prepare_operation_with_context(
            name,
            args,
            operation,
            BuiltinPreparationContext {
                runtime,
                completion,
                ..Default::default()
            },
        )
    }

    pub(super) fn prepare_operation_with_context(
        &self,
        name: &str,
        args: &[Value],
        operation: ArgumentOperation,
        context: BuiltinPreparationContext<'_>,
    ) -> ProviderExecutionResult<PreparedBuiltinCall> {
        let BuiltinPreparationContext {
            runtime,
            completion,
            host,
            compound,
        } = context;
        let Some((evaluation, family, workload)) = operation.evaluate(self, args) else {
            return Ok(PreparedBuiltinCall::unobserved(args.to_vec()));
        };
        let inputs = args.iter().take(operation.count()).collect::<Vec<_>>();
        let identity = runmat_builtins::builtin_id_by_name(name)
            .map(|id| id.0)
            .unwrap_or_else(|| name.to_owned());
        let (evaluation, rejection, plan) = self.plan_operation(
            evaluation,
            &identity,
            family,
            &inputs,
            workload.clone(),
            super::observation::BuiltinPlanningContext {
                compound,
                runtime,
                source_decline: if matches!(&operation, ArgumentOperation::FloatingCast(_)) {
                    crate::placement::LocalSourceDecline::PlainFloatingCast
                } else {
                    crate::placement::LocalSourceDecline::None
                },
                preparation_program: compound
                    .map(|plan| plan.preparation_program())
                    .or_else(|| operation.preparation_program()),
                prepared_authority: completion.as_ref().map(|right| right.session_authority()),
                host,
            },
        )?;
        let trace = record_decision(decision_entry(
            &identity,
            workload.elements.and_then(|n| usize::try_from(n).ok()),
            workload.flops.and_then(|n| usize::try_from(n).ok()),
            &evaluation,
        ));
        record_provider_rejection(&trace, rejection);
        let mut observation =
            BuiltinCallObservation::new(plan.feedback, trace, evaluation.recommend_gpu, args)
                .with_completion(completion);
        if plan.source_declined {
            observation.preparation_fallback();
        }
        let query = plan
            .provider_query
            .as_deref()
            .filter(|_| plan.provider_observation);
        if let Some(compound) = compound {
            observation.prepare_compound(self.provider, query, compound);
        } else {
            observation.prepare_provider(self.provider, query);
        }
        let recorder = observation.recorder();
        runmat_accelerate_api::observation::with_provider_observation(recorder, || {
            let prepared: ProviderExecutionResult<_> = (|| {
                let mut processed = args.to_vec();
                let mut uploads = None;
                if evaluation.recommend_gpu {
                    for (offset, value) in processed.iter_mut().take(operation.count()).enumerate()
                    {
                        if !matches!(value, Value::Tensor(tensor) if !tensor.is_empty())
                            || compound
                                .is_some_and(|plan| plan.preserves_host_scalar_operand(offset))
                        {
                            continue;
                        }
                        if uploads.is_none() {
                            uploads = Some(
                                super::ownership::OwnedUploads::new(args)
                                    .map_err(|error| ProviderExecutionError::from_source(*error))?,
                            );
                        }
                        let uploads = uploads.as_mut().expect("initialized before upload");
                        *value = self.promote_tensor_protecting(
                            value,
                            1,
                            observation.trace.as_ref(),
                            &uploads.protected(),
                        )?;
                        if let Value::GpuTensor(handle) = value {
                            uploads.track(handle);
                        }
                    }
                    if !processed
                        .iter()
                        .take(operation.count())
                        .any(runmat_runtime::dispatcher::value_contains_gpu)
                    {
                        observation.preparation_fallback();
                    }
                }
                Ok((processed, uploads))
            })();
            match prepared {
                Ok((processed, uploads)) => Ok(PreparedBuiltinCall {
                    compound: None,
                    args: processed,
                    observation: Some(observation),
                    uploads,
                    request_input_root: None,
                    readbacks: None,
                    host_numerical_output: None,
                }),
                Err(error) => {
                    // A completed preparation error is an operational failure;
                    // abandoning a future later is not.
                    observation.preparation_failed();
                    Err(error)
                }
            }
        })
    }
}
