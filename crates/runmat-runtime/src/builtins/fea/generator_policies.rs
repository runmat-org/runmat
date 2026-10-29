use std::sync::Once;

use super::*;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::values::{finite_numeric_scalar, finite_numeric_vector};
use crate::class_registry::declaration::{immutable_properties, static_methods};
use runmat_analysis_core::{GeneratorDomain, SignalSamplingPolicy};

use runmat_builtins::fea_contract::policy_types::classes;

pub(super) const DOMAIN_CLASS: &str = classes::SIGNAL_DOMAIN.display_name();
pub(super) const SAMPLING_CLASS: &str = classes::SIGNAL_SAMPLING.display_name();

const DOMAIN_FIELDS: &[&str] = &["domain"];
const SAMPLING_FIELDS: &[&str] = &["sampling"];

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
    ($function:ident, $name:literal, $domain:expr) => {
        #[runtime_builtin(name = $name, binding_variant = "default", builtin_path = "crate::builtins::fea::generator_policies")]
        pub async fn $function() -> BuiltinResult<Value> {
            domain_object($name, $domain)
        }
    };
}

no_input_domain_builtin!(
    time_domain,
    "fea.SignalDomain.time",
    GeneratorDomain::TimeSeconds
);
no_input_domain_builtin!(
    frequency_domain,
    "fea.SignalDomain.frequency",
    GeneratorDomain::FrequencyHertz
);

#[runtime_builtin(
    name = "fea.SignalDomain.studyParameter",
    binding_variant = "default",
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
        GeneratorDomain::StudyParameter {
            parameter_id,
            quantity: None,
        },
    )
}

#[runtime_builtin(
    name = "fea.SignalDomain.studyParameterWithUnits",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::generator_policies"
)]
pub async fn study_parameter_domain_with_units(
    parameter_id: String,
    canonical_units: String,
) -> BuiltinResult<Value> {
    const NAME: &str = "fea.SignalDomain.studyParameterWithUnits";
    if parameter_id.trim().is_empty() {
        return Err(builtin_error(
            NAME,
            &ERROR_INPUT,
            "study parameter id must not be empty",
        ));
    }
    let unit = runmat_types::physical::parse_unit_expression(&canonical_units)
        .map_err(|e| builtin_error(NAME, &ERROR_INPUT, e.to_string()))?;
    if unit.scale() != 1.0 || unit.offset() != 0.0 {
        return Err(builtin_error(NAME, &ERROR_INPUT,
            "sampling numbers are canonical; specify canonical units such as m, rad or K and convert the source values first"));
    }
    domain_object(
        NAME,
        GeneratorDomain::StudyParameter {
            parameter_id,
            quantity: Some(runmat_types::physical::DynamicDimension::from_unit(unit)),
        },
    )
}

#[runtime_builtin(
    name = "fea.SignalSampling.uniform",
    binding_variant = "default",
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

#[runtime_builtin(
    name = "fea.SignalSampling.adaptive",
    binding_variant = "default",
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

#[runtime_builtin(
    name = "fea.SignalSampling.canonicalPoints",
    binding_variant = "default",
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
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            DOMAIN_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let mut schemas = Vec::new();
        for (class_name, type_identity, fields) in [
            (DOMAIN_CLASS, "runmat.fea.signal-domain", DOMAIN_FIELDS),
            (
                SAMPLING_CLASS,
                "runmat.fea.signal-sampling",
                SAMPLING_FIELDS,
            ),
        ] {
            schemas.push(crate::execution::value_codec::ImmutableValueClassSchema {
                class_name,
                type_identity,
                schema_version: 1,
                allowed_fields: fields,
                required_fields: fields,
                validate: validate_policy_object,
            });
        }

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASSES.ensure(|| {
        register_value_class(
            DOMAIN_CLASS,
            immutable_properties(DOMAIN_FIELDS),
            static_methods([
                ("time", "fea.SignalDomain.time"),
                ("frequency", "fea.SignalDomain.frequency"),
                ("studyParameter", "fea.SignalDomain.studyParameter"),
                (
                    "studyParameterWithUnits",
                    "fea.SignalDomain.studyParameterWithUnits",
                ),
            ]),
        );
        register_value_class(
            SAMPLING_CLASS,
            immutable_properties(SAMPLING_FIELDS),
            static_methods([
                ("uniform", "fea.SignalSampling.uniform"),
                ("adaptive", "fea.SignalSampling.adaptive"),
                ("canonicalPoints", "fea.SignalSampling.canonicalPoints"),
            ]),
        );
    });
}

fn validate_policy_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    if object
        .properties()
        .keys()
        .filter(|name| *name != crate::builtins::common::json::typed_object_projection::PROPERTY)
        .count()
        != 1
    {
        return Err(format!("{} has malformed policy data", object.class_name()));
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
                parameter_id: "load_scale".into(),
                quantity: None,
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
