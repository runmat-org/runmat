use std::sync::Once;

use super::*;
use crate::builtins::fea::object_projection::register_value_class;
use crate::class_registry::declaration::{immutable_properties, static_methods};
use runmat_analysis_core::{
    SignalExtrapolation, SignalInterpolation, SignalPointContinuity, SignalSampleApproach,
};
use runmat_builtins::fea_contract::policy_choices::classes::SIGNAL_SAMPLE_APPROACH;
use runmat_builtins::fea_contract::policy_types::classes::{
    SIGNAL_EXTRAPOLATION, SIGNAL_INTERPOLATION, SIGNAL_POINT_CONTINUITY,
};

// Class spellings are owned by the catalog's typed class identities.
pub(super) const INTERPOLATION_CLASS: &str = SIGNAL_INTERPOLATION.display_name();
pub(super) const EXTRAPOLATION_CLASS: &str = SIGNAL_EXTRAPOLATION.display_name();
pub(super) const SAMPLE_APPROACH_CLASS: &str = SIGNAL_SAMPLE_APPROACH.display_name();
pub(super) const POINT_CONTINUITY_CLASS: &str = SIGNAL_POINT_CONTINUITY.display_name();

const INTERPOLATION_FIELDS: &[&str] = &["interpolation"];
const EXTRAPOLATION_FIELDS: &[&str] = &["extrapolation"];
const SAMPLE_APPROACH_FIELDS: &[&str] = &["approach"];
const POINT_CONTINUITY_FIELDS: &[&str] = &["continuity"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InterpolationObject {
    interpolation: SignalInterpolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtrapolationObject {
    extrapolation: SignalExtrapolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SampleApproachObject {
    approach: SignalSampleApproach,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PointContinuityObject {
    continuity: SignalPointContinuity,
}

macro_rules! policy_builtin {
    ($function:ident, $builtin:literal, $class:expr, $payload:expr $(,)?) => {
        #[runtime_builtin(
                                            name = $builtin,
                                            binding_variant = "default",
                                            builtin_path = "crate::builtins::fea::signal_policies"
                                        )]
        pub async fn $function() -> BuiltinResult<Value> {
            serializable_to_object($builtin, &ERROR_INTERNAL, $class, &$payload)
        }
    };
}

policy_builtin!(
    interpolation_left_endpoint,
    "fea.SignalInterpolation.leftEndpoint",
    INTERPOLATION_CLASS,
    InterpolationObject {
        interpolation: SignalInterpolation::LeftEndpoint
    },
);
policy_builtin!(
    point_continuity_left,
    "fea.SignalPointContinuity.left",
    POINT_CONTINUITY_CLASS,
    PointContinuityObject {
        continuity: SignalPointContinuity::Left
    },
);
policy_builtin!(
    point_continuity_right,
    "fea.SignalPointContinuity.right",
    POINT_CONTINUITY_CLASS,
    PointContinuityObject {
        continuity: SignalPointContinuity::Right
    },
);
policy_builtin!(
    sample_approach_at,
    "fea.SignalSampleApproach.at",
    SAMPLE_APPROACH_CLASS,
    SampleApproachObject {
        approach: SignalSampleApproach::At
    },
);
policy_builtin!(
    sample_approach_left_limit,
    "fea.SignalSampleApproach.leftLimit",
    SAMPLE_APPROACH_CLASS,
    SampleApproachObject {
        approach: SignalSampleApproach::LeftLimit
    },
);
policy_builtin!(
    sample_approach_right_limit,
    "fea.SignalSampleApproach.rightLimit",
    SAMPLE_APPROACH_CLASS,
    SampleApproachObject {
        approach: SignalSampleApproach::RightLimit
    },
);
policy_builtin!(
    interpolation_right_endpoint,
    "fea.SignalInterpolation.rightEndpoint",
    INTERPOLATION_CLASS,
    InterpolationObject {
        interpolation: SignalInterpolation::RightEndpoint
    },
);
policy_builtin!(
    interpolation_linear,
    "fea.SignalInterpolation.linear",
    INTERPOLATION_CLASS,
    InterpolationObject {
        interpolation: SignalInterpolation::Linear
    },
);
policy_builtin!(
    extrapolation_reject,
    "fea.SignalExtrapolation.reject",
    EXTRAPOLATION_CLASS,
    ExtrapolationObject {
        extrapolation: SignalExtrapolation::Reject
    },
);
policy_builtin!(
    extrapolation_hold_endpoint,
    "fea.SignalExtrapolation.holdEndpoint",
    EXTRAPOLATION_CLASS,
    ExtrapolationObject {
        extrapolation: SignalExtrapolation::HoldEndpoint
    },
);
policy_builtin!(
    extrapolation_zero,
    "fea.SignalExtrapolation.zero",
    EXTRAPOLATION_CLASS,
    ExtrapolationObject {
        extrapolation: SignalExtrapolation::Zero
    },
);

pub(super) fn interpolation_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<SignalInterpolation> {
    let payload: InterpolationObject = object_payload(builtin, value, INTERPOLATION_CLASS)?;
    Ok(payload.interpolation)
}

pub(super) fn extrapolation_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<SignalExtrapolation> {
    let payload: ExtrapolationObject = object_payload(builtin, value, EXTRAPOLATION_CLASS)?;
    Ok(payload.extrapolation)
}

pub(super) fn sample_approach_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<SignalSampleApproach> {
    let payload: SampleApproachObject = object_payload(builtin, value, SAMPLE_APPROACH_CLASS)?;
    Ok(payload.approach)
}

pub(super) fn point_continuity_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<SignalPointContinuity> {
    let payload: PointContinuityObject = object_payload(builtin, value, POINT_CONTINUITY_CLASS)?;
    Ok(payload.continuity)
}

pub(super) fn ensure_signal_policy_classes_registered() {
    static PORTABLE_SCHEMAS: Once = Once::new();
    static RUNTIME_CLASSES: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            INTERPOLATION_CLASS,
        ));
    PORTABLE_SCHEMAS.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let mut schemas = Vec::new();
        for (class_name, type_identity, fields) in [
            (
                INTERPOLATION_CLASS,
                "runmat.fea.signal-interpolation",
                INTERPOLATION_FIELDS,
            ),
            (
                EXTRAPOLATION_CLASS,
                "runmat.fea.signal-extrapolation",
                EXTRAPOLATION_FIELDS,
            ),
            (
                SAMPLE_APPROACH_CLASS,
                "runmat.fea.signal-sample-approach",
                SAMPLE_APPROACH_FIELDS,
            ),
            (
                POINT_CONTINUITY_CLASS,
                "runmat.fea.signal-point-continuity",
                POINT_CONTINUITY_FIELDS,
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
            POINT_CONTINUITY_CLASS,
            immutable_properties(POINT_CONTINUITY_FIELDS),
            static_methods([
                ("left", "fea.SignalPointContinuity.left"),
                ("right", "fea.SignalPointContinuity.right"),
            ]),
        );
        register_value_class(
            INTERPOLATION_CLASS,
            immutable_properties(INTERPOLATION_FIELDS),
            static_methods([
                ("leftEndpoint", "fea.SignalInterpolation.leftEndpoint"),
                ("rightEndpoint", "fea.SignalInterpolation.rightEndpoint"),
                ("linear", "fea.SignalInterpolation.linear"),
            ]),
        );
        register_value_class(
            SAMPLE_APPROACH_CLASS,
            immutable_properties(SAMPLE_APPROACH_FIELDS),
            static_methods([
                ("at", "fea.SignalSampleApproach.at"),
                ("leftLimit", "fea.SignalSampleApproach.leftLimit"),
                ("rightLimit", "fea.SignalSampleApproach.rightLimit"),
            ]),
        );
        register_value_class(
            EXTRAPOLATION_CLASS,
            immutable_properties(EXTRAPOLATION_FIELDS),
            static_methods([
                ("reject", "fea.SignalExtrapolation.reject"),
                ("holdEndpoint", "fea.SignalExtrapolation.holdEndpoint"),
                ("zero", "fea.SignalExtrapolation.zero"),
            ]),
        );
    });
}

fn validate_policy_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    let mut properties = object.properties().iter().filter(|(name, _)| {
        *name != crate::builtins::common::json::typed_object_projection::PROPERTY
    });
    if !matches!(properties.next(), Some((_, Value::String(_)))) || properties.next().is_some() {
        return Err(format!(
            "{} has a malformed policy value",
            object.class_name()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn policies_are_closed_typed_and_portable() {
        let interpolation = block_on(interpolation_linear()).unwrap();
        let extrapolation = block_on(extrapolation_reject()).unwrap();
        let approach = block_on(sample_approach_left_limit()).unwrap();
        let continuity = block_on(point_continuity_right()).unwrap();
        assert_eq!(
            interpolation_from_value("test", &interpolation).unwrap(),
            SignalInterpolation::Linear
        );
        assert_eq!(
            extrapolation_from_value("test", &extrapolation).unwrap(),
            SignalExtrapolation::Reject
        );
        assert_eq!(
            sample_approach_from_value("test", &approach).unwrap(),
            SignalSampleApproach::LeftLimit
        );
        assert_eq!(
            point_continuity_from_value("test", &continuity).unwrap(),
            SignalPointContinuity::Right
        );
        for value in [interpolation, extrapolation, approach, continuity] {
            let encoded = crate::execution::value_codec::encode_inline_value(&value).unwrap();
            assert_eq!(
                crate::execution::value_codec::decode_inline_value(&encoded).unwrap(),
                value
            );
        }
    }
}
