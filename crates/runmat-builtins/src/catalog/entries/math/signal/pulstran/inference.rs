use crate::catalog::inference::argument_error;
use crate::BuiltinCatalogEntry;
use runmat_types::*;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut diagnostics = Vec::new();
    if request.arguments.len() < 3 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PULSTRAN-ARITY",
            "pulstran requires sample times, delays, and a pulse source",
            request.arguments.len(),
        ));
    }
    for (index, label) in [(0, "sample times"), (1, "delays")] {
        if let Some(value) = request.arguments.get(index) {
            real_dense(value, index, label, &mut diagnostics);
            if request.literals.integer_is_exact_binary64_at(index) == Some(false) {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PULSTRAN-INTEGER-EXACTNESS",
                    "integer waveform controls must convert exactly to double",
                    index,
                ));
            }
        }
    }
    if let Some(delays) = request.arguments.get(1) {
        if let ShapeFact::Shaped { dims } = &delays.shape {
            if dims.len() > 2
                || (dims.len() == 2
                    && matches!((&dims[0], &dims[1]),
                    (DimensionFact::Known(rows), DimensionFact::Known(cols))
                    if *rows != 1 && *cols != 1 && *cols != 2))
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PULSTRAN-DELAY-SHAPE",
                    "delays must be a vector or N-by-2 matrix",
                    1,
                ));
            }
        }
    }
    let sampled = request.arguments.get(2).is_some_and(|pulse| {
        matches!(
            pulse.kind,
            ValueKindFact::Numeric(_) | ValueKindFact::Logical
        )
    });
    if let Some(pulse) = request.arguments.get(2) {
        match &pulse.kind {
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Logical => {
                real_dense(pulse, 2, "sampled prototype", &mut diagnostics);
                if let ShapeFact::Shaped { dims } = &pulse.shape {
                    if dims.len() > 2
                        || pulse.shape.is_definitely_nonscalar()
                            && dims.len() == 2
                            && matches!((&dims[0], &dims[1]),
                            (DimensionFact::Known(rows), DimensionFact::Known(cols))
                            if *rows > 1 && *cols > 1)
                    {
                        diagnostics.push(argument_error(
                            "RM-CATALOG-PULSTRAN-PROTOTYPE-SHAPE",
                            "sampled prototype must be a vector",
                            2,
                        ));
                    }
                }
            }
            ValueKindFact::Character
            | ValueKindFact::String
            | ValueKindFact::Callable(_)
            | ValueKindFact::Unknown => {}
            _ => diagnostics.push(argument_error(
                "RM-CATALOG-PULSTRAN-PULSE",
                "pulse source must be a name, callable, or real numeric prototype",
                2,
            )),
        }
    }
    if sampled && request.arguments.len() > 4 {
        diagnostics.push(argument_error(
            "RM-CATALOG-PULSTRAN-PROTOTYPE-ARITY",
            "sampled prototype accepts at most one sample rate",
            4,
        ));
    }
    if sampled {
        if let Some(rate) = request.arguments.get(3) {
            real_dense(rate, 3, "sample rate", &mut diagnostics);
            if rate.shape.is_definitely_nonscalar()
                || request
                    .literals
                    .numeric_at(3)
                    .is_some_and(|n| !n.is_finite() || n <= 0.0)
            {
                diagnostics.push(argument_error(
                    "RM-CATALOG-PULSTRAN-SAMPLE-RATE",
                    "sample rate must be a positive finite scalar",
                    3,
                ));
            }
        }
    }
    let output = match request.arguments.first() {
        Some(times) if diagnostics.is_empty() => match times.kind {
            ValueKindFact::Numeric(NumericFact {
                domain: NumericDomain::Real,
                ..
            })
            | ValueKindFact::Logical => {
                let storage = match times.shape.element_count() {
                    Some(1) => StorageFact::Scalar,
                    Some(_) => StorageFact::Dense,
                    None => StorageFact::Unknown,
                };
                let mut value = ValueFact::proven(
                    ValueKindFact::Numeric(NumericFact {
                        class: NumericClass::Double,
                        domain: NumericDomain::Real,
                    }),
                    times.shape.clone(),
                    storage,
                );
                value.residency = ResidencyFact::Unknown;
                value
            }
            _ => ValueFact::unknown(DynamicReason::RuntimeValue),
        },
        _ => ValueFact::unknown(DynamicReason::UnsupportedRepresentation),
    };
    let mut contract = CallContract::fixed(vec![output]);
    contract.effects = entry.contract.effect_set();
    contract.capabilities = entry.contract.capability_set();
    match request.arguments.get(2).map(|pulse| &pulse.kind) {
        Some(ValueKindFact::Numeric(_) | ValueKindFact::Logical) => {
            // A sampled prototype is evaluated by the waveform evaluator, not feval.
            contract.effects.0.remove(&EffectKind::HostCallback);
            contract.effects.0.remove(&EffectKind::Unknown);
        }
        Some(ValueKindFact::Callable(callable)) => {
            contract
                .capabilities
                .0
                .extend(callable.capabilities.0.iter().copied());
            if direct_builtin_pulse(callable.identity.as_ref()) {
                contract.effects.0.remove(&EffectKind::HostCallback);
                contract.effects.0.remove(&EffectKind::Unknown);
            } else {
                let known_builtin = matches!(
                    &callable.identity,
                    Some(CallableIdentity::Builtin(identity))
                        if crate::builtin_catalog_entry_by_name(&identity.0).is_some()
                );
                if known_builtin {
                    contract.effects.0.remove(&EffectKind::Unknown);
                }
                let shifted_times = request.arguments.first().map_or_else(
                    || ValueFact::unknown(DynamicReason::RuntimeValue),
                    |times| {
                        let mut shifted = ValueFact::proven(
                            ValueKindFact::Numeric(NumericFact {
                                class: NumericClass::Double,
                                domain: NumericDomain::Real,
                            }),
                            times.shape.clone(),
                            times.storage,
                        );
                        shifted.residency = ResidencyFact::Host;
                        shifted
                    },
                );
                let callback = crate::infer_callable_call(
                    callable,
                    &CallRequest {
                        arguments: std::iter::once(shifted_times)
                            .chain(request.arguments.iter().skip(3).cloned())
                            .collect(),
                        literals: LiteralContext::default(),
                        outputs: OutputSelection::new(RequestedOutputCount::One),
                    },
                );
                contract.effects.0.extend(callback.effects.0);
                contract.capabilities.0.extend(callback.capabilities.0);
            }
        }
        Some(ValueKindFact::Character | ValueKindFact::String) => {
            if request
                .literals
                .literal_args
                .get(2)
                .and_then(crate::catalog::inference::literal_text)
                .is_some_and(|name| direct_pulse_name(&name))
            {
                contract.effects.0.remove(&EffectKind::HostCallback);
                contract.effects.0.remove(&EffectKind::Unknown);
            }
        }
        _ => {}
    }
    let mut result = infer_call(&contract, request);
    result.dynamic_outputs = request.outputs.requested.known_count().is_none();
    result.diagnostics.extend(diagnostics);
    result
}

fn direct_builtin_pulse(identity: Option<&CallableIdentity>) -> bool {
    matches!(identity, Some(CallableIdentity::Builtin(id)) if direct_pulse_name(&id.0))
}

// Mirrors the runtime's directly evaluated pulse family. Dynamic names are not
// resolved through the catalog: a source function may shadow a builtin name.
fn direct_pulse_name(name: &str) -> bool {
    let trimmed = name.trim();
    let normalized = trimmed.strip_prefix('@').unwrap_or(trimmed).trim();
    matches!(
        normalized.to_ascii_lowercase().as_str(),
        "rectpuls" | "tripuls" | "gauspuls"
    )
}

fn real_dense(
    value: &ValueFact,
    index: usize,
    label: &str,
    diagnostics: &mut Vec<InferenceDiagnostic>,
) {
    if !matches!(
        value.kind,
        ValueKindFact::Numeric(NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Unknown
    ) || matches!(value.storage, StorageFact::Sparse | StorageFact::Opaque)
    {
        diagnostics.push(argument_error(
            "RM-CATALOG-PULSTRAN-INPUT",
            format!("{label} must be a dense real numeric value"),
            index,
        ));
    }
}
