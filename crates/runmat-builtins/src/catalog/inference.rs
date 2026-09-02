use super::{
    AccelerationInferenceRule, AggregateInferenceRule, ArrayInferenceRule, BuiltinCatalogEntry,
    BuiltinContractMaturity, BuiltinInferenceRule, IntrospectionInferenceRule, MathInferenceRule,
    NumericComponentRule, NumericLimitRule, ParallelInferenceRule, StatsInferenceRule,
    StatsRandomInferenceRule,
};
use runmat_types::{
    codistributor_fact, infer_call, infer_numeric_conversion, AliasFact, CallContract,
    CallInference, CallRequest, CodistributorClass, ContiguityFact, DistributedFact, DynamicReason,
    ExecutionFact, FutureStateFact, InferenceDiagnostic, LayoutFact, LiteralContext, LiteralValue,
    MutationFact, NumericClass, NumericDomain, NumericFact, OutputListFact, ResidencyFact,
    ShapeFact, StorageFact, StructFact, ValueFact, ValueKindFact, ViewFact,
};
use std::collections::BTreeMap;

mod math_binary;
mod math_special;
mod math_unary;
mod stats_random;

pub fn infer_catalog_call(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    let distributed = request.arguments.iter().find_map(|argument| {
        let ValueKindFact::Distributed(distributed) = &argument.kind else {
            return None;
        };
        Some(distributed.clone())
    });
    if let Some(distributed) = distributed {
        match entry.placement.distributed {
            crate::BuiltinDistributedPolicy::MapUnary => {
                return infer_distributed_map(entry, request, distributed);
            }
            crate::BuiltinDistributedPolicy::ScalarLikePrototype => {
                return infer_distributed_scalar_like(entry, request, distributed);
            }
            crate::BuiltinDistributedPolicy::MaterializeArguments => {
                return infer_partition_local_call(entry, request);
            }
            crate::BuiltinDistributedPolicy::Unsupported
            | crate::BuiltinDistributedPolicy::InspectHandles => {}
        }
    }
    infer_catalog_call_local(entry, request)
}

fn infer_catalog_call_local(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    match entry.contract.inference_rule {
        BuiltinInferenceRule::Array(ArrayInferenceRule::Full) => infer_full(request, entry),
        BuiltinInferenceRule::Array(ArrayInferenceRule::Zeros) => infer_zeros(request, entry),
        BuiltinInferenceRule::Math(MathInferenceRule::Abs) => infer_abs(request, entry),
        BuiltinInferenceRule::Math(MathInferenceRule::Atan2) => {
            math_binary::infer_atan2(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Hypot) => {
            math_binary::infer_hypot(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Gamma) => {
            math_special::infer_gamma(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::GammaLn) => {
            math_special::infer_gammaln(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::PhaseAngle) => {
            math_unary::infer_phase_angle(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Exp) => math_unary::infer_exp(request, entry),
        BuiltinInferenceRule::Math(MathInferenceRule::Expm1) => {
            math_unary::infer_expm1(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Log1p) => {
            math_unary::infer_log1p(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Log2) => {
            math_unary::infer_log2(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Logarithm(base)) => {
            math_unary::infer_logarithm(request, entry, base)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Root(kind)) => {
            math_unary::infer_root(request, entry, kind)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::NumericLimit(rule)) => {
            infer_numeric_limit(request, entry, rule)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::NumericConversion(target)) => {
            infer_numeric_conversion_call(request, entry, target)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::NumericConversionWithLike(target)) => {
            infer_numeric_conversion_with_like_call(request, entry, target)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::NumericComponent(rule)) => {
            infer_numeric_component_call(request, entry, rule)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Rounding(function)) => {
            math_unary::infer_rounding(request, entry, function)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Round) => {
            math_unary::infer_round(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Remainder(function)) => {
            math_binary::infer_remainder(request, entry, function)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Signum) => {
            math_unary::infer_signum(request, entry)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Trigonometric(function)) => {
            math_unary::infer_trigonometric(request, entry, function)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::Hyperbolic(function)) => {
            math_unary::infer_hyperbolic(request, entry, function)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::PiScaledTrigonometric(function)) => {
            math_unary::infer_pi_scaled_trigonometric(request, entry, function)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::DegreeTrigonometric(function)) => {
            math_unary::infer_degree_trigonometric(request, entry, function)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::InverseTrigonometric(function)) => {
            math_unary::infer_inverse_trigonometric(request, entry, function)
        }
        BuiltinInferenceRule::Math(MathInferenceRule::InverseHyperbolic(function)) => {
            math_unary::infer_inverse_hyperbolic(request, entry, function)
        }
        BuiltinInferenceRule::Stats(StatsInferenceRule::Random(function)) => match function {
            StatsRandomInferenceRule::Binomial => stats_random::infer_binornd(request, entry),
            StatsRandomInferenceRule::Gamma => stats_random::infer_gamrnd(request, entry),
        },
        BuiltinInferenceRule::Acceleration(AccelerationInferenceRule::Gather) => {
            infer_gather(request, entry)
        }
        BuiltinInferenceRule::Acceleration(AccelerationInferenceRule::GpuArray) => {
            infer_gpu_array(request, entry)
        }
        BuiltinInferenceRule::Aggregate(AggregateInferenceRule::Struct) => {
            infer_struct_builtin(request, entry)
        }
        BuiltinInferenceRule::Introspection(IntrospectionInferenceRule::Feval) => {
            infer_feval(request, entry)
        }
        BuiltinInferenceRule::Parallel(ParallelInferenceRule::Parpool) => {
            infer_parallel_pool(request, entry, false)
        }
        BuiltinInferenceRule::Parallel(ParallelInferenceRule::Gcp) => {
            infer_parallel_pool(request, entry, true)
        }
        BuiltinInferenceRule::Parallel(
            ParallelInferenceRule::Parfeval | ParallelInferenceRule::ParfevalOnAll,
        ) => infer_parallel_future(request, entry),
        BuiltinInferenceRule::Parallel(ParallelInferenceRule::FetchOutputs) => {
            infer_parallel_fetch(request, entry, false)
        }
        BuiltinInferenceRule::Parallel(ParallelInferenceRule::FetchNext) => {
            infer_parallel_fetch(request, entry, true)
        }
        BuiltinInferenceRule::Parallel(
            ParallelInferenceRule::GetCurrentJob
            | ParallelInferenceRule::GetCurrentTask
            | ParallelInferenceRule::GetCurrentWorker,
        ) => unavailable_rule(entry, request),
        BuiltinInferenceRule::Parallel(_) => infer_parallel_data(request, entry),
    }
}

fn infer_numeric_conversion_call(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    target: NumericClass,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-CONVERSION-ARITY",
            format!("{} requires exactly one input", entry.identity.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-CONVERSION-ARITY",
            format!("{} accepts exactly one input", entry.identity.name),
            1,
        ));
    }
    let mut inference = infer_numeric_conversion(input, target);
    diagnostics.append(&mut inference.diagnostics);
    finish_fixed(entry, request, inference.fact, diagnostics)
}

fn infer_numeric_conversion_with_like_call(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    target: NumericClass,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-CONVERSION-ARITY",
            format!("{} requires an input value", entry.identity.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let mut inference = infer_numeric_conversion(input, target);
    diagnostics.append(&mut inference.diagnostics);
    match request.arguments.as_slice() {
        [_] => {
            if matches!(inference.fact.residency, ResidencyFact::Device { .. }) {
                inference.fact.residency = ResidencyFact::Unknown;
            }
        }
        [_, keyword, prototype] => {
            let keyword_literal = request.literals.literal_args.get(1).and_then(literal_text);
            match keyword_literal.as_deref() {
                Some(value) if value.eq_ignore_ascii_case("like") => {
                    apply_like_prototype_residency(
                        &mut inference.fact,
                        prototype,
                        &mut diagnostics,
                    );
                }
                Some(_) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-CONVERSION-LIKE",
                        format!("{} accepts only the \"like\" option", entry.identity.name),
                        1,
                    ));
                    inference.fact.residency = ResidencyFact::Unknown;
                }
                None if matches!(
                    keyword.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                ) =>
                {
                    inference.fact.residency = ResidencyFact::Unknown;
                }
                None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-CONVERSION-LIKE",
                        format!(
                            "{} requires \"like\" as its second input",
                            entry.identity.name
                        ),
                        1,
                    ));
                    inference.fact.residency = ResidencyFact::Unknown;
                }
            }
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-CONVERSION-ARITY",
                format!(
                    "{} accepts either one input or an input followed by \"like\" and a prototype",
                    entry.identity.name
                ),
                request.arguments.len().saturating_sub(1),
            ));
            inference.fact.residency = ResidencyFact::Unknown;
        }
    }

    finish_fixed(entry, request, inference.fact, diagnostics)
}

fn apply_like_prototype_residency(
    output: &mut ValueFact,
    prototype: &ValueFact,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    match prototype.kind {
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        })
        | ValueKindFact::Logical => {
            output.residency = match &prototype.residency {
                ResidencyFact::Host => ResidencyFact::Host,
                ResidencyFact::Device { provider } => ResidencyFact::Device {
                    provider: provider.clone(),
                },
                ResidencyFact::Unknown => ResidencyFact::Unknown,
                ResidencyFact::Remote { .. } => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
                        "a remote value cannot be used as a numeric conversion prototype",
                        2,
                    ));
                    ResidencyFact::Unknown
                }
            };
        }
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Complex,
            ..
        }) => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
                "complex numeric conversion prototypes are not supported",
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
        ValueKindFact::Unknown => output.residency = ResidencyFact::Unknown,
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE",
                "numeric conversion prototypes must be real numeric or logical values",
                2,
            ));
            output.residency = ResidencyFact::Unknown;
        }
    }
}

fn infer_numeric_component_call(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    rule: NumericComponentRule,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(input) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-COMPONENT-ARITY",
            format!("{} requires exactly one input", entry.identity.name),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-COMPONENT-ARITY",
            format!("{} accepts exactly one input", entry.identity.name),
            1,
        ));
    }

    if matches!(input.storage, StorageFact::Sparse) {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-COMPONENT-SPARSE",
            format!(
                "{} does not currently accept sparse input",
                entry.identity.name
            ),
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
            diagnostics,
        );
    }

    let mut output = input.clone();
    let mut materializes = false;
    let mut changes_class = false;
    match (&mut output.kind, rule) {
        (ValueKindFact::Numeric(numeric), NumericComponentRule::Conjugate) => {
            materializes = matches!(numeric.domain, NumericDomain::Complex);
        }
        (ValueKindFact::Numeric(numeric), NumericComponentRule::RealPart) => {
            materializes = matches!(numeric.domain, NumericDomain::Complex);
            numeric.domain = NumericDomain::Real;
        }
        (ValueKindFact::Numeric(numeric), NumericComponentRule::ImaginaryPart) => {
            materializes = true;
            numeric.domain = NumericDomain::Real;
        }
        (ValueKindFact::Logical, NumericComponentRule::Conjugate) => {}
        (
            ValueKindFact::Logical | ValueKindFact::Character,
            NumericComponentRule::RealPart | NumericComponentRule::ImaginaryPart,
        )
        | (ValueKindFact::Character, NumericComponentRule::Conjugate) => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            materializes = true;
            changes_class = true;
        }
        (ValueKindFact::Unknown, _) => {
            let shape = output.shape.clone();
            output = ValueFact::unknown(DynamicReason::RuntimeValue);
            output.shape = shape;
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-COMPONENT-INPUT",
                format!(
                    "{} requires numeric, logical, or character input",
                    entry.identity.name
                ),
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }

    if materializes {
        output.storage = if output.is_scalar() {
            StorageFact::Scalar
        } else {
            StorageFact::Dense
        };
        output.layout = LayoutFact::ColumnMajor;
        output.contiguity = ContiguityFact::Contiguous;
        output.view = ViewFact::Materialized;
        output.alias = AliasFact::Unique;
        output.mutation = MutationFact::ValueSemantics;
    }
    if changes_class && matches!(output.residency, ResidencyFact::Device { .. }) {
        output.residency = ResidencyFact::Unknown;
    }

    finish_fixed(entry, request, output, diagnostics)
}

/// Infer the local result produced by one catalog-admitted partition-local
/// builtin invocation.
///
/// Distributed handles remain execution-owned. Static builtin rules operate
/// on the value fact carried by each handle, which is the same fact presented
/// to the builtin for every partition. The returned outputs therefore
/// describe partition payloads, not another distributed handle; the execution
/// service wraps those facts in a newly fenced handle after the map completes.
pub fn infer_partition_local_call(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
) -> CallInference {
    let mut projected = request.clone();
    projected.arguments = request
        .arguments
        .iter()
        .map(|argument| match &argument.kind {
            ValueKindFact::Distributed(distributed) => distributed.value.as_ref().clone(),
            _ => argument.clone(),
        })
        .collect();
    infer_catalog_call_local(entry, &projected)
}

fn infer_distributed_map(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    source: DistributedFact,
) -> CallInference {
    let mut inference = infer_partition_local_call(entry, request);
    inference.outputs = inference
        .outputs
        .into_iter()
        .map(|value| {
            ValueFact::scalar(ValueKindFact::Distributed(DistributedFact {
                id: source.id,
                owner: source.owner,
                scheme: source.scheme.clone(),
                value: Box::new(value),
                materializable: source.materializable,
            }))
        })
        .collect();
    inference
}

fn infer_distributed_scalar_like(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    source: DistributedFact,
) -> CallInference {
    let mut inference = infer_partition_local_call(entry, request);
    inference.outputs = inference
        .outputs
        .into_iter()
        .map(|value| {
            ValueFact::scalar(ValueKindFact::Distributed(DistributedFact {
                id: source.id,
                owner: source.owner,
                scheme: source.scheme.clone(),
                value: Box::new(value),
                materializable: source.materializable,
            }))
        })
        .collect();
    inference
}

fn infer_numeric_limit(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    rule: NumericLimitRule,
) -> CallInference {
    let mut diagnostics = Vec::new();
    let default_class = match rule {
        NumericLimitRule::Integer(_) => NumericClass::Int32,
        NumericLimitRule::Floating(_) => NumericClass::Double,
    };
    let mut output = numeric_limit_scalar(default_class, NumericDomain::Real);

    match request.arguments.as_slice() {
        [] => {}
        [class] => match request.literals.literal_args.first().and_then(literal_text) {
            Some(name) => match NumericClass::from_class_name(&name) {
                Some(class) if numeric_limit_accepts_class(rule, class) => {
                    output = numeric_limit_scalar(class, NumericDomain::Real);
                }
                _ => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-LIMIT-CLASS",
                        format!("{} does not support class `{name}`", entry.identity.name),
                        0,
                    ));
                    output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                }
            },
            None if matches!(
                class.kind,
                ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
            ) =>
            {
                output = ValueFact::unknown(DynamicReason::RuntimeValue);
            }
            None => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-NUMERIC-LIMIT-CLASS",
                    format!("{} requires a numeric class name", entry.identity.name),
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            }
        },
        [keyword, prototype] => {
            let keyword_literal = request.literals.literal_args.first().and_then(literal_text);
            match keyword_literal.as_deref() {
                Some(value) if value.eq_ignore_ascii_case("like") => {
                    output = infer_numeric_limit_like(rule, prototype, entry, &mut diagnostics);
                }
                Some(_) => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-LIMIT-LIKE",
                        format!("{} accepts only the `like` option", entry.identity.name),
                        0,
                    ));
                    output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                }
                None if matches!(
                    keyword.kind,
                    ValueKindFact::String | ValueKindFact::Character | ValueKindFact::Unknown
                ) =>
                {
                    output = ValueFact::unknown(DynamicReason::RuntimeValue);
                }
                None => {
                    diagnostics.push(argument_error(
                        "RM-CATALOG-NUMERIC-LIMIT-LIKE",
                        format!(
                            "{} requires `like` before its prototype",
                            entry.identity.name
                        ),
                        0,
                    ));
                    output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
                }
            }
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-LIMIT-ARITY",
                format!(
                    "{} accepts no input, a class name, or `like` and a prototype",
                    entry.identity.name
                ),
                request.arguments.len().saturating_sub(1),
            ));
            output = ValueFact::unknown(DynamicReason::RuntimeValue);
        }
    }

    finish_fixed(entry, request, output, diagnostics)
}

fn numeric_limit_accepts_class(rule: NumericLimitRule, class: NumericClass) -> bool {
    match rule {
        NumericLimitRule::Integer(_) => {
            !matches!(class, NumericClass::Double | NumericClass::Single)
        }
        NumericLimitRule::Floating(_) => {
            matches!(class, NumericClass::Double | NumericClass::Single)
        }
    }
}

fn numeric_limit_scalar(class: NumericClass, domain: NumericDomain) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Numeric(NumericFact { class, domain }))
}

fn infer_numeric_limit_like(
    rule: NumericLimitRule,
    prototype: &ValueFact,
    entry: &BuiltinCatalogEntry,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) -> ValueFact {
    let ValueKindFact::Numeric(numeric) = prototype.kind else {
        if !matches!(prototype.kind, ValueKindFact::Unknown) {
            diagnostics.push(argument_error(
                "RM-CATALOG-NUMERIC-LIMIT-PROTOTYPE",
                format!("{} requires a numeric prototype", entry.identity.name),
                1,
            ));
        }
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if !numeric_limit_accepts_class(rule, numeric.class)
        || matches!(rule, NumericLimitRule::Integer(_))
            && matches!(prototype.storage, StorageFact::Sparse)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-NUMERIC-LIMIT-PROTOTYPE",
            format!(
                "{} does not support this prototype representation",
                entry.identity.name
            ),
            1,
        ));
        return ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
    }

    let mut output = numeric_limit_scalar(numeric.class, numeric.domain);
    output.storage = if matches!(prototype.storage, StorageFact::Sparse) {
        StorageFact::Sparse
    } else {
        StorageFact::Scalar
    };
    output.residency = prototype.residency.clone();
    output
}

fn infer_parallel_data(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let BuiltinInferenceRule::Parallel(rule) = entry.contract.inference_rule else {
        unreachable!("parallel inference accepts only typed parallel rules")
    };
    let output = match rule {
        ParallelInferenceRule::Barrier | ParallelInferenceRule::Send => {
            ValueFact::scalar(ValueKindFact::Void)
        }
        ParallelInferenceRule::Probe => ValueFact::scalar(ValueKindFact::Logical),
        ParallelInferenceRule::SpmdIndex | ParallelInferenceRule::SpmdSize => {
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }))
        }
        ParallelInferenceRule::LocalPart => {
            match request.arguments.first().map(|fact| &fact.kind) {
                Some(ValueKindFact::Distributed(distributed)) => distributed.value.as_ref().clone(),
                _ => ValueFact::unknown(DynamicReason::RuntimeValue),
            }
        }
        ParallelInferenceRule::Redistribute => request
            .arguments
            .first()
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::GetCodistributor => {
            let class = request.arguments.first().and_then(|fact| match &fact.kind {
                ValueKindFact::Distributed(distributed) => distributed
                    .scheme
                    .as_ref()
                    .and_then(CodistributorClass::from_scheme),
                _ => None,
            });
            codistributor_fact(class)
        }
        ParallelInferenceRule::GlobalIndices => ValueFact::unknown(DynamicReason::RuntimeValue),
        ParallelInferenceRule::Codistributor1d => {
            codistributor_fact(Some(CodistributorClass::OneDimensional))
        }
        ParallelInferenceRule::Codistributor2dbc => {
            codistributor_fact(Some(CodistributorClass::TwoDimensionalBlockCyclic))
        }
        ParallelInferenceRule::Codistributor => codistributor_fact(None),
        ParallelInferenceRule::CodistributorIsComplete | ParallelInferenceRule::Iscodistributed => {
            ValueFact::scalar(ValueKindFact::Logical)
        }
        ParallelInferenceRule::Broadcast => request
            .arguments
            .get(1)
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::SendReceive => request
            .arguments
            .get(2)
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::Gplus => request
            .arguments
            .first()
            .cloned()
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::Cat => request
            .arguments
            .first()
            .cloned()
            .map(|mut fact| {
                fact.shape = ShapeFact::Unknown;
                fact
            })
            .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue)),
        ParallelInferenceRule::FunctionalReduce => ValueFact::unknown(DynamicReason::RuntimeValue),
        ParallelInferenceRule::Distributed
        | ParallelInferenceRule::Codistributed
        | ParallelInferenceRule::CodistributedBuild
        | ParallelInferenceRule::Receive => ValueFact::unknown(DynamicReason::RuntimeValue),
        ParallelInferenceRule::FetchNext
        | ParallelInferenceRule::FetchOutputs
        | ParallelInferenceRule::Gcp
        | ParallelInferenceRule::GetCurrentJob
        | ParallelInferenceRule::GetCurrentTask
        | ParallelInferenceRule::GetCurrentWorker
        | ParallelInferenceRule::Parfeval
        | ParallelInferenceRule::ParfevalOnAll
        | ParallelInferenceRule::Parpool => {
            unreachable!("parallel rule was routed to its dedicated inference handler")
        }
    };
    finish_fixed(entry, request, output, Vec::new())
}

fn infer_parallel_pool(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    current_pool: bool,
) -> CallInference {
    let nocreate = current_pool
        && request
            .literals
            .literal_args
            .first()
            .and_then(literal_text)
            .is_some_and(|value| value.eq_ignore_ascii_case("nocreate"));
    let output = if nocreate {
        ValueFact::unknown(DynamicReason::RuntimeValue)
    } else {
        ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Pool))
    };
    finish_fixed(entry, request, output, Vec::new())
}

fn infer_parallel_future(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let explicit_pool = matches!(
        request.arguments.first().map(|fact| &fact.kind),
        Some(ValueKindFact::Execution(ExecutionFact::Pool))
    );
    let callable_index = usize::from(explicit_pool);
    let output_count_index = callable_index + 1;
    let output = scheduled_callable_output(request, callable_index, output_count_index);
    let future = ValueFact::scalar(ValueKindFact::Execution(ExecutionFact::Future {
        output: Box::new(output),
        state: FutureStateFact::Unknown,
    }));
    finish_fixed(entry, request, future, Vec::new())
}

fn scheduled_callable_output(
    request: &CallRequest,
    callable_index: usize,
    output_count_index: usize,
) -> ValueFact {
    let Some(ValueKindFact::Callable(callable)) =
        request.arguments.get(callable_index).map(|fact| &fact.kind)
    else {
        return ValueFact::unknown(DynamicReason::UnresolvedCallable);
    };
    let Some(output_count) = request
        .literals
        .literal_args
        .get(output_count_index)
        .and_then(literal_output_count)
    else {
        return ValueFact::unknown(DynamicReason::RuntimeValue);
    };
    if output_count == 0 {
        return ValueFact::scalar(ValueKindFact::Void);
    }
    let outputs = (0..output_count)
        .map(|index| {
            callable
                .outputs
                .get(index)
                .cloned()
                .unwrap_or_else(|| ValueFact::unknown(DynamicReason::RuntimeValue))
        })
        .collect::<Vec<_>>();
    if output_count == 1 {
        outputs
            .into_iter()
            .next()
            .expect("one requested output was materialized")
    } else {
        ValueFact::scalar(ValueKindFact::OutputList(OutputListFact {
            outputs,
            variadic: !callable.outputs_complete || callable.variadic_outputs,
        }))
    }
}

fn literal_output_count(literal: &LiteralValue) -> Option<usize> {
    let LiteralValue::Number(value) = literal else {
        return None;
    };
    (value.is_finite() && *value >= 0.0 && value.fract() == 0.0 && *value <= usize::MAX as f64)
        .then_some(*value as usize)
}

fn infer_parallel_fetch(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    include_index: bool,
) -> CallInference {
    let payload = match request.arguments.first().map(|fact| &fact.kind) {
        Some(ValueKindFact::Execution(ExecutionFact::Future { output, .. })) => {
            output.as_ref().clone()
        }
        _ => ValueFact::unknown(DynamicReason::RuntimeValue),
    };
    let mut outputs = Vec::new();
    if include_index {
        outputs.push(default_double_scalar());
    }
    match payload.kind {
        ValueKindFact::OutputList(OutputListFact {
            outputs: payload_outputs,
            ..
        }) => outputs.extend(payload_outputs),
        ValueKindFact::Void => {}
        _ => outputs.push(payload),
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    infer_call(&contract, request)
}

fn infer_feval(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut contract = match request.arguments.first().map(|argument| &argument.kind) {
        Some(ValueKindFact::Callable(callable)) => CallContract {
            outputs: callable.outputs.clone(),
            variadic_output: (callable.variadic_outputs || !callable.outputs_complete)
                .then(|| Box::new(ValueFact::unknown(DynamicReason::RuntimeValue))),
            maximum_outputs: (callable.outputs_complete && !callable.variadic_outputs)
                .then_some(callable.outputs.len()),
            effects: Default::default(),
            capabilities: callable.capabilities.clone(),
            dynamic_reason: (!callable.outputs_complete || callable.variadic_outputs)
                .then_some(DynamicReason::RuntimeValue),
        },
        Some(_) => CallContract::dynamic(DynamicReason::RuntimeValue),
        None => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FEVAL-ARITY",
                "feval requires a function target",
                0,
            ));
            CallContract::dynamic(DynamicReason::RuntimeValue)
        }
    };
    contract.effects = entry.contract.effect_set();
    contract
        .capabilities
        .0
        .extend(entry.contract.capability_set().0);
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn infer_struct_builtin(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let output = match request.arguments.as_slice() {
        [] => struct_fact(BTreeMap::new(), true),
        [template] => match template.kind {
            ValueKindFact::Struct(_) => template.clone(),
            ValueKindFact::Unknown => ValueFact::unknown(DynamicReason::RuntimeValue),
            _ if template.shape.element_count() == Some(0) => struct_fact(BTreeMap::new(), true),
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-STRUCT-TEMPLATE",
                    "single-argument struct requires a struct or empty array template",
                    0,
                ));
                ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
            }
        },
        arguments if arguments.len() % 2 == 0 => {
            let mut fields = BTreeMap::new();
            let mut complete = true;
            for index in (0..arguments.len()).step_by(2) {
                if let Some(name) = request
                    .literals
                    .literal_args
                    .get(index)
                    .and_then(literal_text)
                {
                    fields.insert(name, arguments[index + 1].clone());
                } else {
                    complete = false;
                }
            }
            struct_fact(fields, complete)
        }
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-STRUCT-PAIRS",
                "struct requires complete field/value pairs",
                request.arguments.len().saturating_sub(1),
            ));
            ValueFact::unknown(DynamicReason::UnsupportedRepresentation)
        }
    };
    finish_fixed(entry, request, output, diagnostics)
}

fn struct_fact(fields: BTreeMap<String, ValueFact>, fields_complete: bool) -> ValueFact {
    ValueFact::scalar(ValueKindFact::Struct(StructFact {
        fields,
        fields_complete,
    }))
}

fn infer_gather(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.is_empty() {
        diagnostics.push(argument_error(
            "RM-CATALOG-GATHER-ARITY",
            "gather requires at least one input",
            0,
        ));
    }
    let outputs = request
        .arguments
        .iter()
        .cloned()
        .map(|mut fact| {
            fact.residency = ResidencyFact::Host;
            fact
        })
        .collect::<Vec<_>>();
    if let Some(requested) = request.outputs.requested.known_count() {
        let valid = requested == 0
            || (request.arguments.len() == 1 && requested == 1)
            || (request.arguments.len() > 1 && requested == request.arguments.len());
        if !valid {
            diagnostics.push(InferenceDiagnostic::error(
                "RM-CATALOG-GATHER-OUTPUTS",
                "gather output count must match its input count",
            ));
        }
    }
    let mut contract = CallContract::fixed(outputs);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn infer_gpu_array(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let Some(source) = request.arguments.first() else {
        diagnostics.push(argument_error(
            "RM-CATALOG-GPUARRAY-ARITY",
            "gpuArray requires an input value",
            0,
        ));
        return finish_fixed(
            entry,
            request,
            ValueFact::unknown(DynamicReason::RuntimeValue),
            diagnostics,
        );
    };

    let literals = &request.literals.literal_args;
    let like_index = literals
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(index, literal)| {
            literal_text(literal)
                .is_some_and(|value| value.eq_ignore_ascii_case("like"))
                .then_some(index)
        });
    let mut output = like_index
        .and_then(|index| request.arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(|| source.clone());

    let explicit_class = literals
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(index, _)| like_index.is_none_or(|like| *index != like && *index != like + 1))
        .filter_map(|(_, literal)| literal_text(literal))
        .find_map(|value| numeric_class_tag(&value));

    if let Some(kind) = explicit_class {
        output.kind = kind;
    } else if like_index.is_none() {
        match &output.kind {
            ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Unknown => {}
            ValueKindFact::Character | ValueKindFact::String => {
                output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
                output.storage = StorageFact::Dense;
            }
            _ => {
                diagnostics.push(argument_error(
                    "RM-CATALOG-GPUARRAY-INPUT",
                    "gpuArray requires a numeric, logical, character, or string input",
                    0,
                ));
                output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
            }
        }
    }

    if let Some(shape) = gpu_array_shape(request) {
        output.shape = shape;
        output.storage = StorageFact::Dense;
    }
    output.residency = ResidencyFact::Device { provider: None };
    finish_fixed(entry, request, output, diagnostics)
}

fn gpu_array_shape(request: &CallRequest) -> Option<ShapeFact> {
    let literals = &request.literals.literal_args;
    if literals.len() <= 1 {
        return None;
    }
    if let Some(dimensions) = request.literals.numeric_vector_at(1) {
        return Some(ShapeFact::from(dimensions));
    }

    let mut dimensions = Vec::new();
    for index in 1..request.arguments.len() {
        let literal = literals.get(index).unwrap_or(&LiteralValue::Unknown);
        if literal_text(literal).is_some() {
            break;
        }
        let dimension = LiteralContext::numeric_from_literal(literal).and_then(|value| {
            (value.is_finite() && value >= 0.0 && value.fract() == 0.0).then_some(value as usize)
        });
        let scalar_numeric = matches!(
            request.arguments[index].kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        ) && request.arguments[index].is_scalar();
        if dimension.is_none() && !scalar_numeric {
            break;
        }
        dimensions.push(dimension);
    }
    (!dimensions.is_empty()).then(|| ShapeFact::from(dimensions))
}

fn numeric_class_tag(value: &str) -> Option<ValueKindFact> {
    let kind = match value.to_ascii_lowercase().as_str() {
        "double" => numeric_kind(NumericClass::Double, NumericDomain::Real),
        "single" | "float32" => numeric_kind(NumericClass::Single, NumericDomain::Real),
        "logical" | "bool" | "boolean" => ValueKindFact::Logical,
        "int8" => numeric_kind(NumericClass::Int8, NumericDomain::Real),
        "int16" => numeric_kind(NumericClass::Int16, NumericDomain::Real),
        "int32" | "int" => numeric_kind(NumericClass::Int32, NumericDomain::Real),
        "int64" => numeric_kind(NumericClass::Int64, NumericDomain::Real),
        "uint8" => numeric_kind(NumericClass::UInt8, NumericDomain::Real),
        "uint16" => numeric_kind(NumericClass::UInt16, NumericDomain::Real),
        "uint32" => numeric_kind(NumericClass::UInt32, NumericDomain::Real),
        "uint64" => numeric_kind(NumericClass::UInt64, NumericDomain::Real),
        _ => return None,
    };
    Some(kind)
}

fn infer_abs(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = request.arguments.first().cloned().unwrap_or_else(|| {
        diagnostics.push(argument_error(
            "RM-CATALOG-ABS-ARITY",
            "abs requires exactly one input",
            0,
        ));
        ValueFact::unknown(DynamicReason::RuntimeValue)
    });
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-ABS-ARITY",
            "abs accepts exactly one input",
            1,
        ));
    }
    match &mut output.kind {
        ValueKindFact::Numeric(numeric) => numeric.domain = NumericDomain::Real,
        ValueKindFact::Logical | ValueKindFact::Character => {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
        }
        ValueKindFact::Symbolic | ValueKindFact::Unknown => {}
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-ABS-INPUT",
                "abs requires numeric, logical, character, or symbolic input",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn infer_zeros(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let literals = &request.literals.literal_args;
    let like_index = literals.iter().position(
        |literal| matches!(literal, LiteralValue::String(value) | LiteralValue::Character(value) | LiteralValue::Keyword(value) if value.eq_ignore_ascii_case("like")),
    );
    let trailing_class = literals
        .last()
        .and_then(literal_text)
        .filter(|value| !value.eq_ignore_ascii_case("like"));
    let dimension_end = like_index.unwrap_or_else(|| {
        trailing_class
            .as_ref()
            .map_or(literals.len(), |_| literals.len().saturating_sub(1))
    });

    let mut output = like_index
        .and_then(|index| request.arguments.get(index + 1))
        .cloned()
        .unwrap_or_else(default_double_scalar);

    if let Some(class) = trailing_class.as_deref() {
        if let Some(numeric_class) = NumericClass::from_class_name(class) {
            output.kind = numeric_kind(numeric_class, NumericDomain::Real);
        } else if class.eq_ignore_ascii_case("logical") {
            output.kind = ValueKindFact::Logical;
        } else if class.eq_ignore_ascii_case("gpuarray") {
            output.kind = numeric_kind(NumericClass::Double, NumericDomain::Real);
            output.residency = ResidencyFact::Device { provider: None };
        } else {
            diagnostics.push(argument_error(
                "RM-CATALOG-ZEROS-CLASS",
                "zeros class specifier is not recognized",
                literals.len().saturating_sub(1),
            ));
        }
    }

    if dimension_end > 0 {
        output.shape = zeros_shape(request, dimension_end);
        output.storage = StorageFact::Dense;
    } else if like_index.is_none() {
        output = default_double_scalar();
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn zeros_shape(request: &CallRequest, dimension_end: usize) -> ShapeFact {
    if dimension_end == 1 {
        if let Some(dims) = request.literals.numeric_vector_at(0) {
            return ShapeFact::from(dims);
        }
        if let Some(size) = request.literals.numeric_dims().first().copied().flatten() {
            return ShapeFact::from(vec![Some(size), Some(size)]);
        }
        return match request
            .arguments
            .first()
            .and_then(|argument| argument.shape.rank())
        {
            Some(2) if !request.arguments[0].is_scalar() => {
                let rank = request.arguments[0]
                    .shape
                    .element_count()
                    .unwrap_or(2)
                    .max(2);
                ShapeFact::Ranked { rank }
            }
            _ => ShapeFact::Ranked { rank: 2 },
        };
    }
    ShapeFact::from(
        request
            .literals
            .numeric_dims()
            .into_iter()
            .take(dimension_end)
            .collect::<Vec<_>>(),
    )
}

fn literal_text(literal: &LiteralValue) -> Option<String> {
    match literal {
        LiteralValue::String(value)
        | LiteralValue::Character(value)
        | LiteralValue::Keyword(value) => Some(value.clone()),
        _ => None,
    }
}

fn numeric_kind(class: NumericClass, domain: NumericDomain) -> ValueKindFact {
    ValueKindFact::Numeric(NumericFact { class, domain })
}

fn preserved_binary_residency(left: &ResidencyFact, right: &ResidencyFact) -> ResidencyFact {
    match (left, right) {
        (ResidencyFact::Host, ResidencyFact::Host) => ResidencyFact::Host,
        (
            ResidencyFact::Device {
                provider: left_owner,
            },
            ResidencyFact::Device {
                provider: right_owner,
            },
        ) if left_owner == right_owner => left.clone(),
        (ResidencyFact::Device { .. }, ResidencyFact::Host) => left.clone(),
        (ResidencyFact::Host, ResidencyFact::Device { .. }) => right.clone(),
        _ => ResidencyFact::Unknown,
    }
}

fn default_double_scalar() -> ValueFact {
    ValueFact::scalar(numeric_kind(NumericClass::Double, NumericDomain::Real))
}

fn infer_full(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    let mut output = request.arguments.first().cloned().unwrap_or_else(|| {
        diagnostics.push(argument_error(
            "RM-CATALOG-FULL-ARITY",
            "full requires exactly one input",
            0,
        ));
        ValueFact::unknown(DynamicReason::RuntimeValue)
    });
    if request.arguments.len() > 1 {
        diagnostics.push(argument_error(
            "RM-CATALOG-FULL-ARITY",
            "full accepts exactly one input",
            1,
        ));
    }
    match output.kind {
        ValueKindFact::Numeric(_) | ValueKindFact::Logical | ValueKindFact::Character => {
            if matches!(output.storage, StorageFact::Sparse) {
                output.storage = StorageFact::Dense;
            }
        }
        ValueKindFact::Unknown => {}
        _ => {
            diagnostics.push(argument_error(
                "RM-CATALOG-FULL-INPUT",
                "full requires a numeric, logical, or character array",
                0,
            ));
            output = ValueFact::unknown(DynamicReason::UnsupportedRepresentation);
        }
    }
    finish_fixed(entry, request, output, diagnostics)
}

fn unavailable_rule(entry: &BuiltinCatalogEntry, request: &CallRequest) -> CallInference {
    let mut contract = CallContract::dynamic(DynamicReason::UnsupportedRepresentation);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    if matches!(entry.contract.maturity, BuiltinContractMaturity::Complete) {
        inference.diagnostics.push(InferenceDiagnostic::error(
            "RM-CATALOG-INFERENCE-RULE",
            format!(
                "complete builtin contract `{:?}` has no registered inference rule",
                entry.contract.inference_rule
            ),
        ));
    }
    inference
}

fn finish_fixed(
    entry: &BuiltinCatalogEntry,
    request: &CallRequest,
    output: ValueFact,
    mut diagnostics: Vec<InferenceDiagnostic>,
) -> CallInference {
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    let mut inference = infer_call(&contract, request);
    diagnostics.append(&mut inference.diagnostics);
    inference.diagnostics = diagnostics;
    inference
}

fn argument_error(
    code: impl Into<String>,
    message: impl Into<String>,
    argument: usize,
) -> InferenceDiagnostic {
    let mut diagnostic = InferenceDiagnostic::error(code, message);
    diagnostic.argument = Some(argument);
    diagnostic
}
