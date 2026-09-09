use std::collections::HashMap;
use std::sync::Once;

use super::*;
use runmat_analysis_core::{
    SignalExtrapolation, SignalInterpolation, SignalPointContinuity, SignalSampleApproach,
};

pub(super) const INTERPOLATION_CLASS: &str = "fea.SignalInterpolation";
pub(super) const EXTRAPOLATION_CLASS: &str = "fea.SignalExtrapolation";
pub(super) const SAMPLE_APPROACH_CLASS: &str = "fea.SignalSampleApproach";
pub(super) const POINT_CONTINUITY_CLASS: &str = "fea.SignalPointContinuity";

const INTERPOLATION_FIELDS: &[&str] = &["interpolation"];
const EXTRAPOLATION_FIELDS: &[&str] = &["extrapolation"];
const SAMPLE_APPROACH_FIELDS: &[&str] = &["approach"];
const POINT_CONTINUITY_FIELDS: &[&str] = &["continuity"];
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const POLICY_INTEGER_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes: "Signal-policy constructors accept no numeric input; the concrete method fixes the closed policy value.",
};

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
    (
        $function:ident, $descriptor:ident, $signatures:ident,
        $builtin:literal, $label:literal, $summary:literal, $class:expr,
        $payload:expr
    ) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
            label: $label,
            inputs: &NO_INPUTS,
            outputs: &OUT_ANY,
        }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::Fixed,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
        #[runtime_builtin(
                            name = $builtin,
                            category = "fea",
                            summary = $summary,
                            keywords = "fea,signal,interpolation,extrapolation,policy,typed",
                            descriptor($descriptor),
                            integer_audit(POLICY_INTEGER_AUDIT),
                            builtin_path = "crate::builtins::fea::signal_policies"
                        )]
        pub async fn $function() -> BuiltinResult<Value> {
            serializable_to_object($builtin, &ERROR_INTERNAL, $class, &$payload)
        }
    };
}

policy_builtin!(
    interpolation_left_endpoint,
    INTERPOLATION_LEFT_ENDPOINT_DESCRIPTOR,
    INTERPOLATION_LEFT_ENDPOINT_SIGNATURES,
    "fea.SignalInterpolation.leftEndpoint",
    "policy = fea.SignalInterpolation.leftEndpoint()",
    "Use the outgoing value of each interval's left endpoint.",
    INTERPOLATION_CLASS,
    InterpolationObject {
        interpolation: SignalInterpolation::LeftEndpoint
    }
);
policy_builtin!(
    point_continuity_left,
    POINT_CONTINUITY_LEFT_DESCRIPTOR,
    POINT_CONTINUITY_LEFT_SIGNATURES,
    "fea.SignalPointContinuity.left",
    "continuity = fea.SignalPointContinuity.left()",
    "Assign the value at a jump knot to its left-hand limit.",
    POINT_CONTINUITY_CLASS,
    PointContinuityObject {
        continuity: SignalPointContinuity::Left
    }
);
policy_builtin!(
    point_continuity_right,
    POINT_CONTINUITY_RIGHT_DESCRIPTOR,
    POINT_CONTINUITY_RIGHT_SIGNATURES,
    "fea.SignalPointContinuity.right",
    "continuity = fea.SignalPointContinuity.right()",
    "Assign the value at a jump knot to its right-hand limit.",
    POINT_CONTINUITY_CLASS,
    PointContinuityObject {
        continuity: SignalPointContinuity::Right
    }
);
policy_builtin!(
    sample_approach_at,
    SAMPLE_APPROACH_AT_DESCRIPTOR,
    SAMPLE_APPROACH_AT_SIGNATURES,
    "fea.SignalSampleApproach.at",
    "approach = fea.SignalSampleApproach.at()",
    "Sample the value assigned at an exact signal abscissa.",
    SAMPLE_APPROACH_CLASS,
    SampleApproachObject {
        approach: SignalSampleApproach::At
    }
);
policy_builtin!(
    sample_approach_left_limit,
    SAMPLE_APPROACH_LEFT_LIMIT_DESCRIPTOR,
    SAMPLE_APPROACH_LEFT_LIMIT_SIGNATURES,
    "fea.SignalSampleApproach.leftLimit",
    "approach = fea.SignalSampleApproach.leftLimit()",
    "Sample the limit approaching an exact knot from below.",
    SAMPLE_APPROACH_CLASS,
    SampleApproachObject {
        approach: SignalSampleApproach::LeftLimit
    }
);
policy_builtin!(
    sample_approach_right_limit,
    SAMPLE_APPROACH_RIGHT_LIMIT_DESCRIPTOR,
    SAMPLE_APPROACH_RIGHT_LIMIT_SIGNATURES,
    "fea.SignalSampleApproach.rightLimit",
    "approach = fea.SignalSampleApproach.rightLimit()",
    "Sample the limit approaching an exact knot from above.",
    SAMPLE_APPROACH_CLASS,
    SampleApproachObject {
        approach: SignalSampleApproach::RightLimit
    }
);
policy_builtin!(
    interpolation_right_endpoint,
    INTERPOLATION_RIGHT_ENDPOINT_DESCRIPTOR,
    INTERPOLATION_RIGHT_ENDPOINT_SIGNATURES,
    "fea.SignalInterpolation.rightEndpoint",
    "policy = fea.SignalInterpolation.rightEndpoint()",
    "Use the incoming value of each interval's right endpoint.",
    INTERPOLATION_CLASS,
    InterpolationObject {
        interpolation: SignalInterpolation::RightEndpoint
    }
);
policy_builtin!(
    interpolation_linear,
    INTERPOLATION_LINEAR_DESCRIPTOR,
    INTERPOLATION_LINEAR_SIGNATURES,
    "fea.SignalInterpolation.linear",
    "policy = fea.SignalInterpolation.linear()",
    "Select piecewise-linear interpolation.",
    INTERPOLATION_CLASS,
    InterpolationObject {
        interpolation: SignalInterpolation::Linear
    }
);
policy_builtin!(
    extrapolation_reject,
    EXTRAPOLATION_REJECT_DESCRIPTOR,
    EXTRAPOLATION_REJECT_SIGNATURES,
    "fea.SignalExtrapolation.reject",
    "policy = fea.SignalExtrapolation.reject()",
    "Reject evaluation outside the sampled signal domain.",
    EXTRAPOLATION_CLASS,
    ExtrapolationObject {
        extrapolation: SignalExtrapolation::Reject
    }
);
policy_builtin!(
    extrapolation_hold_endpoint,
    EXTRAPOLATION_HOLD_DESCRIPTOR,
    EXTRAPOLATION_HOLD_SIGNATURES,
    "fea.SignalExtrapolation.holdEndpoint",
    "policy = fea.SignalExtrapolation.holdEndpoint()",
    "Hold the nearest endpoint outside the sampled signal domain.",
    EXTRAPOLATION_CLASS,
    ExtrapolationObject {
        extrapolation: SignalExtrapolation::HoldEndpoint
    }
);
policy_builtin!(
    extrapolation_zero,
    EXTRAPOLATION_ZERO_DESCRIPTOR,
    EXTRAPOLATION_ZERO_SIGNATURES,
    "fea.SignalExtrapolation.zero",
    "policy = fea.SignalExtrapolation.zero()",
    "Use zero outside the sampled signal domain.",
    EXTRAPOLATION_CLASS,
    ExtrapolationObject {
        extrapolation: SignalExtrapolation::Zero
    }
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
        crate::class_registry::ClassRegistration::new(INTERPOLATION_CLASS);
    PORTABLE_SCHEMAS.call_once(|| {
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
                panic!("failed to register immutable signal policy {class_name}: {error}")
            });
        }
    });
    RUNTIME_CLASSES.ensure(|| {
        register_policy_class(
            POINT_CONTINUITY_CLASS,
            POINT_CONTINUITY_FIELDS,
            &[
                ("left", "fea.SignalPointContinuity.left"),
                ("right", "fea.SignalPointContinuity.right"),
            ],
        );
        register_policy_class(
            INTERPOLATION_CLASS,
            INTERPOLATION_FIELDS,
            &[
                ("leftEndpoint", "fea.SignalInterpolation.leftEndpoint"),
                ("rightEndpoint", "fea.SignalInterpolation.rightEndpoint"),
                ("linear", "fea.SignalInterpolation.linear"),
            ],
        );
        register_policy_class(
            SAMPLE_APPROACH_CLASS,
            SAMPLE_APPROACH_FIELDS,
            &[
                ("at", "fea.SignalSampleApproach.at"),
                ("leftLimit", "fea.SignalSampleApproach.leftLimit"),
                ("rightLimit", "fea.SignalSampleApproach.rightLimit"),
            ],
        );
        register_policy_class(
            EXTRAPOLATION_CLASS,
            EXTRAPOLATION_FIELDS,
            &[
                ("reject", "fea.SignalExtrapolation.reject"),
                ("holdEndpoint", "fea.SignalExtrapolation.holdEndpoint"),
                ("zero", "fea.SignalExtrapolation.zero"),
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
    if object.properties.len() != 1
        || !matches!(object.properties.values().next(), Some(Value::String(_)))
    {
        return Err(format!(
            "{} has a malformed policy value",
            object.class_name
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
