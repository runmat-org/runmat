use std::collections::HashMap;
use std::sync::Once;

use super::*;
use crate::builtins::fea::values::{finite_numeric_scalar, finite_numeric_vector};
use runmat_analysis_core::{GeneratorDomain, SignalSamplingPolicy};

pub(super) const DOMAIN_CLASS: &str = "fea.SignalDomain";
pub(super) const SAMPLING_CLASS: &str = "fea.SignalSampling";

const DOMAIN_FIELDS: &[&str] = &["domain"];
const SAMPLING_FIELDS: &[&str] = &["sampling"];
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const STUDY_PARAMETER_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "parameterId",
    BuiltinParamType::StringScalar,
    "Stable study-parameter identifier.",
)];
const UNIFORM_INPUTS: [BuiltinParamDescriptor; 3] = [
    required(
        "start",
        BuiltinParamType::NumericScalar,
        "Finite first canonical sample coordinate.",
    ),
    required(
        "end",
        BuiltinParamType::NumericScalar,
        "Finite last canonical sample coordinate.",
    ),
    required(
        "sampleCount",
        BuiltinParamType::NumericScalar,
        "Integer number of samples, at least two.",
    ),
];
const ADAPTIVE_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "start",
        BuiltinParamType::NumericScalar,
        "Finite first canonical sample coordinate.",
    ),
    required(
        "end",
        BuiltinParamType::NumericScalar,
        "Finite last canonical sample coordinate.",
    ),
    required(
        "maxAbsoluteError",
        BuiltinParamType::NumericScalar,
        "Finite positive absolute interpolation-error target.",
    ),
    required(
        "maxSamples",
        BuiltinParamType::NumericScalar,
        "Integer sample budget, at least two.",
    ),
];
const POINTS_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "points",
    BuiltinParamType::NumericArray,
    "Strictly increasing finite canonical sample coordinates.",
)];
const OUT_POLICY: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "policy",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Typed signal-generator policy.",
}];
const NO_NUMERIC_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "This constructor accepts no numeric values.",
};
const NUMERIC_INPUTS: [BuiltinIntegerInputCapability; 4] = [
    integer_input(
        "start",
        "Canonical coordinates are converted once to binary64.",
    ),
    integer_input(
        "end",
        "Canonical coordinates are converted once to binary64.",
    ),
    integer_input(
        "sampleCount",
        "Sample counts must be exact non-negative platform integers.",
    ),
    integer_input(
        "maxSamples",
        "Sample budgets must be exact non-negative platform integers.",
    ),
];
const NUMERIC_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "typed signal-generator sampling policy",
        inputs: &NUMERIC_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes:
            "Coordinates enter canonical binary64 storage; counts retain exact integer semantics.",
    }];

const fn required(
    name: &'static str,
    ty: BuiltinParamType,
    description: &'static str,
) -> BuiltinParamDescriptor {
    BuiltinParamDescriptor {
        name,
        ty,
        arity: BuiltinParamArity::Required,
        default: None,
        description,
    }
}

const fn integer_input(name: &'static str, notes: &'static str) -> BuiltinIntegerInputCapability {
    BuiltinIntegerInputCapability {
        name,
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DomainObject {
    domain: GeneratorDomain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SamplingObject {
    sampling: SignalSamplingPolicy,
}

macro_rules! no_input_domain_builtin {
    ($function:ident, $descriptor:ident, $signatures:ident, $name:literal, $method:literal, $summary:literal, $domain:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: concat!("domain = fea.SignalDomain.", $method, "()"),
            inputs: &NO_INPUTS,
            outputs: &OUT_POLICY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        #[runtime_builtin(
                                    name = $name,
                                    category = "fea",
                                    summary = $summary,
                                    keywords = "fea,signal,generator,domain,typed",
                                    descriptor($descriptor),
                                    integer_audit(NO_NUMERIC_AUDIT),
                                    builtin_path = "crate::builtins::fea::generator_policies"
                                )]
        pub async fn $function() -> BuiltinResult<Value> {
            domain_object($name, $domain)
        }
    };
}

no_input_domain_builtin!(
    time_domain,
    TIME_DOMAIN_DESCRIPTOR,
    TIME_DOMAIN_SIGNATURES,
    "fea.SignalDomain.time",
    "time",
    "Select time in seconds as a callable signal domain.",
    GeneratorDomain::TimeSeconds
);
no_input_domain_builtin!(
    frequency_domain,
    FREQUENCY_DOMAIN_DESCRIPTOR,
    FREQUENCY_DOMAIN_SIGNATURES,
    "fea.SignalDomain.frequency",
    "frequency",
    "Select frequency in hertz as a callable signal domain.",
    GeneratorDomain::FrequencyHertz
);

const STUDY_PARAMETER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "domain = fea.SignalDomain.studyParameter(parameterId)",
    inputs: &STUDY_PARAMETER_INPUTS,
    outputs: &OUT_POLICY,
}];
pub(super) const STUDY_PARAMETER_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STUDY_PARAMETER_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.SignalDomain.studyParameter",
    category = "fea",
    summary = "Select a named study parameter as a callable signal domain.",
    keywords = "fea,signal,generator,domain,parameter,typed",
    descriptor(STUDY_PARAMETER_DESCRIPTOR),
    integer_audit(NO_NUMERIC_AUDIT),
    builtin_path = "crate::builtins::fea::generator_policies"
)]
pub async fn study_parameter_domain(parameter_id: String) -> BuiltinResult<Value> {
    if parameter_id.trim().is_empty() {
        return Err(builtin_error(
            "fea.SignalDomain.studyParameter",
            &ERROR_INPUT,
            "study parameter id must not be empty",
        ));
    }
    domain_object(
        "fea.SignalDomain.studyParameter",
        GeneratorDomain::StudyParameter { parameter_id },
    )
}

const UNIFORM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "sampling = fea.SignalSampling.uniform(start, end, sampleCount)",
    inputs: &UNIFORM_INPUTS,
    outputs: &OUT_POLICY,
}];
pub(super) const UNIFORM_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &UNIFORM_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.SignalSampling.uniform",
    category = "fea",
    summary = "Create uniform callable signal sampling.",
    keywords = "fea,signal,generator,sampling,uniform,typed",
    descriptor(UNIFORM_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::generator_policies"
)]
pub async fn uniform_sampling(
    start: Value,
    end: Value,
    sample_count: Value,
) -> BuiltinResult<Value> {
    let builtin = "fea.SignalSampling.uniform";
    sampling_object(
        builtin,
        SignalSamplingPolicy::Uniform {
            start: finite_numeric_scalar(builtin, &start, "start")?,
            end: finite_numeric_scalar(builtin, &end, "end")?,
            sample_count: usize_from_value(builtin, &sample_count)?,
        },
    )
}

const ADAPTIVE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "sampling = fea.SignalSampling.adaptive(start, end, maxAbsoluteError, maxSamples)",
    inputs: &ADAPTIVE_INPUTS,
    outputs: &OUT_POLICY,
}];
pub(super) const ADAPTIVE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &ADAPTIVE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.SignalSampling.adaptive",
    category = "fea",
    summary = "Create error-bounded adaptive callable signal sampling.",
    keywords = "fea,signal,generator,sampling,adaptive,typed",
    descriptor(ADAPTIVE_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::generator_policies"
)]
pub async fn adaptive_sampling(
    start: Value,
    end: Value,
    max_absolute_error: Value,
    max_samples: Value,
) -> BuiltinResult<Value> {
    let builtin = "fea.SignalSampling.adaptive";
    sampling_object(
        builtin,
        SignalSamplingPolicy::Adaptive {
            start: finite_numeric_scalar(builtin, &start, "start")?,
            end: finite_numeric_scalar(builtin, &end, "end")?,
            max_absolute_error: finite_numeric_scalar(
                builtin,
                &max_absolute_error,
                "maxAbsoluteError",
            )?,
            max_samples: usize_from_value(builtin, &max_samples)?,
        },
    )
}

const POINTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "sampling = fea.SignalSampling.canonicalPoints(points)",
    inputs: &POINTS_INPUTS,
    outputs: &OUT_POLICY,
}];
pub(super) const POINTS_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &POINTS_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};

#[runtime_builtin(
    name = "fea.SignalSampling.canonicalPoints",
    category = "fea",
    summary = "Create explicit callable signal sample coordinates.",
    keywords = "fea,signal,generator,sampling,points,typed",
    descriptor(POINTS_DESCRIPTOR),
    integer_capabilities(NUMERIC_CAPABILITIES),
    builtin_path = "crate::builtins::fea::generator_policies"
)]
pub async fn canonical_point_sampling(points: Value) -> BuiltinResult<Value> {
    let builtin = "fea.SignalSampling.canonicalPoints";
    sampling_object(
        builtin,
        SignalSamplingPolicy::CanonicalPoints(finite_numeric_vector(builtin, &points, "points")?),
    )
}

fn domain_object(builtin: &'static str, domain: GeneratorDomain) -> BuiltinResult<Value> {
    serializable_to_object(
        builtin,
        &ERROR_INTERNAL,
        DOMAIN_CLASS,
        &DomainObject { domain },
    )
}

fn sampling_object(builtin: &'static str, sampling: SignalSamplingPolicy) -> BuiltinResult<Value> {
    validate_sampling(builtin, &sampling)?;
    serializable_to_object(
        builtin,
        &ERROR_INTERNAL,
        SAMPLING_CLASS,
        &SamplingObject { sampling },
    )
}

fn validate_sampling(builtin: &'static str, sampling: &SignalSamplingPolicy) -> BuiltinResult<()> {
    let valid = match sampling {
        SignalSamplingPolicy::Uniform {
            start,
            end,
            sample_count,
        } => start.is_finite() && end.is_finite() && start < end && *sample_count >= 2,
        SignalSamplingPolicy::Adaptive {
            start,
            end,
            max_absolute_error,
            max_samples,
        } => {
            start.is_finite()
                && end.is_finite()
                && start < end
                && max_absolute_error.is_finite()
                && *max_absolute_error > 0.0
                && *max_samples >= 2
        }
        SignalSamplingPolicy::CanonicalPoints(points) => {
            points.len() >= 2
                && points.iter().all(|point| point.is_finite())
                && points.windows(2).all(|pair| pair[0] < pair[1])
        }
    };
    if valid {
        Ok(())
    } else {
        Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "signal sampling must be finite, strictly ordered, and contain at least two points",
        ))
    }
}

pub(super) fn domain_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<GeneratorDomain> {
    object_payload::<DomainObject>(builtin, value, DOMAIN_CLASS).map(|value| value.domain)
}

pub(super) fn sampling_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<SignalSamplingPolicy> {
    let sampling = object_payload::<SamplingObject>(builtin, value, SAMPLING_CLASS)?.sampling;
    validate_sampling(builtin, &sampling)?;
    Ok(sampling)
}

pub(super) fn ensure_generator_policy_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(DOMAIN_CLASS);
    PORTABLE_SCHEMAS.call_once(|| {
        for (class_name, type_identity, fields) in [
            (DOMAIN_CLASS, "runmat.fea.signal-domain", DOMAIN_FIELDS),
            (
                SAMPLING_CLASS,
                "runmat.fea.signal-sampling",
                SAMPLING_FIELDS,
            ),
        ] {
            crate::execution::value_codec::register_immutable_value_class(
                crate::execution::value_codec::ImmutableValueClassSchema {
                    class_name,
                    type_identity,
                    schema_version: 1,
                    allowed_fields: fields,
                    required_fields: fields,
                    validate: validate_policy_object,
                },
            )
            .unwrap_or_else(|error| {
                panic!("failed to register immutable signal generator policy {class_name}: {error}")
            });
        }
    });
    RUNTIME_CLASSES.ensure(|| {
        register_policy_class(
            DOMAIN_CLASS,
            DOMAIN_FIELDS,
            &[
                ("time", "fea.SignalDomain.time"),
                ("frequency", "fea.SignalDomain.frequency"),
                ("studyParameter", "fea.SignalDomain.studyParameter"),
            ],
        );
        register_policy_class(
            SAMPLING_CLASS,
            SAMPLING_FIELDS,
            &[
                ("uniform", "fea.SignalSampling.uniform"),
                ("adaptive", "fea.SignalSampling.adaptive"),
                ("canonicalPoints", "fea.SignalSampling.canonicalPoints"),
            ],
        );
    });
}

fn register_policy_class(class_name: &str, fields: &[&str], methods: &[(&str, &str)]) {
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: class_name.into(),
        parent: None,
        properties: fields
            .iter()
            .map(|field| {
                (
                    (*field).into(),
                    crate::class_registry::RuntimeProperty {
                        name: (*field).into(),
                        is_static: false,
                        is_constant: true,
                        is_dependent: false,
                        get_access: MemberAccess::Public,
                        set_access: MemberAccess::Private,
                        default_value: None,
                    },
                )
            })
            .collect(),
        methods: methods
            .iter()
            .map(|(name, builtin)| {
                (
                    (*name).into(),
                    crate::class_registry::RuntimeMethod {
                        name: (*name).into(),
                        is_static: true,
                        is_abstract: false,
                        is_sealed: false,
                        access: MemberAccess::Public,
                        function_name: (*builtin).into(),
                        implicit_class_argument: None,
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
    });
}

fn validate_policy_object(object: &ObjectInstance) -> Result<(), String> {
    if object.properties.len() != 1 {
        return Err(format!("{} has malformed policy data", object.class_name));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn generator_policies_are_typed_portable_and_validate_bounds() {
        let domain = block_on(study_parameter_domain("load_scale".into())).unwrap();
        assert_eq!(
            domain_from_value("test", &domain).unwrap(),
            GeneratorDomain::StudyParameter {
                parameter_id: "load_scale".into()
            }
        );
        let sampling = block_on(uniform_sampling(
            Value::Num(0.0),
            Value::Num(1.0),
            Value::Int(IntValue::U64(5)),
        ))
        .unwrap();
        assert_eq!(
            sampling_from_value("test", &sampling).unwrap(),
            SignalSamplingPolicy::Uniform {
                start: 0.0,
                end: 1.0,
                sample_count: 5
            }
        );
        for value in [domain, sampling] {
            let encoded = crate::execution::value_codec::encode_inline_value(&value).unwrap();
            assert_eq!(
                crate::execution::value_codec::decode_inline_value(&encoded).unwrap(),
                value
            );
        }
        assert!(block_on(uniform_sampling(
            Value::Num(1.0),
            Value::Num(0.0),
            Value::Int(IntValue::U64(1)),
        ))
        .is_err());
    }
}
