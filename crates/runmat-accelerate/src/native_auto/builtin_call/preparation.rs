use super::*;

pub(super) enum ArgumentOperation {
    Reduction,
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
            _ => None,
        }
    }
    fn for_call(name: &str, args: &[Value]) -> Option<Self> {
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
        let (evaluation, family) = match self {
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
            Self::Reduction => (
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
                let flops = rows.saturating_mul(inner).saturating_mul(cols);
                workload.elements = None;
                workload.flops = u64::try_from(flops).ok();
                (
                    auto.evaluate_matmul(flops),
                    ProviderOperationFamily::MatrixMultiply,
                )
            }
        };
        Some((evaluation, family, workload))
    }
}

impl NativeAutoOffload {
    pub(super) fn prepare_call(
        &self,
        name: &str,
        args: &[Value],
        runtime: Option<&RuntimeContext>,
    ) -> Result<PreparedBuiltinCall> {
        if !self.enabled
            || (name.eq_ignore_ascii_case("double")
                && self.provider.precision() != ProviderPrecision::F64)
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
    ) -> Result<PreparedBuiltinCall> {
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
                runtime,
                preparation_program: operation.preparation_program(),
            },
        );
        let trace = record_decision(decision_entry(
            &identity,
            workload.elements.and_then(|n| usize::try_from(n).ok()),
            workload.flops.and_then(|n| usize::try_from(n).ok()),
            &evaluation,
        ));
        record_provider_rejection(&trace, rejection);
        let mut observation =
            BuiltinCallObservation::new(plan.feedback, trace, evaluation.recommend_gpu, args);
        observation.prepare_provider(
            self.provider,
            plan.provider_query
                .as_deref()
                .filter(|_| plan.provider_observation),
        );
        let recorder = observation.recorder();
        runmat_accelerate_api::observation::with_provider_observation(recorder, || {
            let mut processed = args.to_vec();
            let mut uploads = None;
            if evaluation.recommend_gpu {
                for value in processed.iter_mut().take(operation.count()) {
                    if !matches!(value, Value::Tensor(tensor) if !tensor.is_empty()) {
                        continue;
                    }
                    if uploads.is_none() {
                        uploads = Some(super::ownership::OwnedUploads::new(self.provider, args)?);
                    }
                    let uploads = uploads.as_mut().expect("initialized before upload");
                    *value = self.promote_tensor_protecting(
                        value,
                        1,
                        observation.trace.as_ref(),
                        &uploads.protected(),
                    )?;
                    uploads.track(value);
                }
                if !processed
                    .iter()
                    .take(operation.count())
                    .any(runmat_runtime::dispatcher::value_contains_gpu)
                {
                    observation.preparation_fallback();
                }
            }
            observation.execution_started();
            Ok(PreparedBuiltinCall {
                args: processed,
                observation: Some(observation),
                uploads,
            })
        })
    }
}
