use std::collections::HashMap;
use std::sync::Once;

use super::*;
use crate::builtins::fea::constraints::support::frame::coordinate_frame_from_value;
use crate::builtins::fea::values::{
    finite_numeric_scalar, finite_numeric_vector, finite_numeric_vector3,
};
use runmat_analysis_core::{
    DirectionalCombination, ModalCombination, ModalTruncationPolicy, ShockResponseSpectrumMethod,
    SpectrumAbscissa, SpectrumConstruction, SpectrumDamping, SpectrumDirection, SpectrumEnvelope,
    SpectrumEvidence, SpectrumExtrapolation, SpectrumInterpolation, SpectrumResponse,
};

pub(super) const AXIS_CLASS: &str = "fea.SpectrumAxis";
pub(super) const DAMPING_CLASS: &str = "fea.SpectrumDamping";
pub(super) const DIRECTION_CLASS: &str = "fea.SpectrumDirection";
pub(super) const RESPONSE_CLASS: &str = "fea.SpectrumResponse";
pub(super) const ENVELOPE_CLASS: &str = "fea.SpectrumEnvelope";
pub(super) const INTERPOLATION_CLASS: &str = "fea.SpectrumInterpolation";
pub(super) const EXTRAPOLATION_CLASS: &str = "fea.SpectrumExtrapolation";
pub(super) const MODAL_COMBINATION_CLASS: &str = "fea.ModalCombination";
pub(super) const DIRECTIONAL_COMBINATION_CLASS: &str = "fea.DirectionalCombination";
pub(super) const TRUNCATION_CLASS: &str = "fea.ModalTruncation";
pub(super) const CONSTRUCTION_CLASS: &str = "fea.SpectrumConstruction";
pub(super) const EVIDENCE_CLASS: &str = "fea.SpectrumEvidence";

const VALUE_FIELDS: &[&str] = &["value"];
const NO_INPUTS: [BuiltinParamDescriptor; 0] = [];
const POLICY_AUDIT: BuiltinIntegerAuditDescriptor = BuiltinIntegerAuditDescriptor {
    kind: BuiltinIntegerAuditKind::NotApplicable,
    canonical_builtin: None,
    notes:
        "The concrete method fixes a closed response-spectrum policy and accepts no numeric input.",
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedValue<T> {
    value: T,
}

macro_rules! noarg_policy {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $label:literal,
     $summary:literal, $class:expr, $value:expr) => {
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
                            keywords = "fea,response,spectrum,policy,typed",
                            descriptor($descriptor),
                            integer_audit(POLICY_AUDIT),
                            builtin_path = "crate::builtins::fea::response_spectrum_policies::choices"
                        )]
        pub async fn $function() -> BuiltinResult<Value> {
            typed_value($builtin, $class, $value)
        }
    };
}

fn typed_value<T: Serialize>(
    builtin: &'static str,
    class_name: &'static str,
    value: T,
) -> BuiltinResult<Value> {
    serializable_to_object(builtin, &ERROR_INTERNAL, class_name, &TypedValue { value })
}

fn typed_value_from<T: DeserializeOwned>(
    builtin: &'static str,
    value: &Value,
    class_name: &'static str,
) -> BuiltinResult<T> {
    let value: TypedValue<T> = object_payload(builtin, value, class_name)?;
    Ok(value.value)
}

const VECTOR_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "values",
    BuiltinParamType::NumericArray,
    "Finite nonempty numeric row or column vector in the unit named by the constructor.",
)];
const SCALAR_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "value",
    BuiltinParamType::NumericScalar,
    "Finite scalar value.",
)];
const DIRECTION_INPUTS: [BuiltinParamDescriptor; 2] = [
    required("frame", BuiltinParamType::Any, "Typed fea.CoordinateFrame."),
    required(
        "vector",
        BuiltinParamType::NumericArray,
        "Finite nonzero three-component direction vector.",
    ),
];
const PERCENTAGE_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "primary",
        BuiltinParamType::NumericScalar,
        "Primary fraction.",
    ),
    required(
        "secondary",
        BuiltinParamType::NumericScalar,
        "Secondary fraction.",
    ),
];
const TRUNCATION_INPUTS: [BuiltinParamDescriptor; 4] = [
    required(
        "minimumMassRatio",
        BuiltinParamType::NumericScalar,
        "Required cumulative modal-mass fraction.",
    ),
    required(
        "includeResidualMode",
        BuiltinParamType::LogicalArray,
        "Include residual-mode correction.",
    ),
    required(
        "includeRigidResponse",
        BuiltinParamType::LogicalArray,
        "Include rigid response.",
    ),
    required(
        "includeMissingMass",
        BuiltinParamType::LogicalArray,
        "Include missing-mass correction.",
    ),
];
const CONSTRUCTION_SIGNAL_INPUTS: [BuiltinParamDescriptor; 1] = [required(
    "accelerationSignal",
    BuiltinParamType::Any,
    "Typed time-domain fea.AccelerationSignal source.",
)];
const EVIDENCE_INPUTS: [BuiltinParamDescriptor; 2] = [
    required(
        "samplingAdequacyRatio",
        BuiltinParamType::NumericScalar,
        "Finite nonnegative sampling adequacy ratio.",
    ),
    required(
        "maximumInterpolationError",
        BuiltinParamType::NumericScalar,
        "Finite nonnegative interpolation-error bound.",
    ),
];

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

const SCALAR_NUMERIC_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "value",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "The physical or dimensionless value is checked before conversion to canonical binary64 storage.",
}];
const SCALAR_NUMERIC_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "typed response-spectrum numeric component",
        inputs: &SCALAR_NUMERIC_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Response-spectrum authoring is a portable host construction boundary.",
    }];

macro_rules! numeric_capabilities {
    ($inputs:ident, $capabilities:ident, $form:literal, [$($name:literal),+ $(,)?]) => {
        const $inputs: [BuiltinIntegerInputCapability; numeric_capabilities!(@count $($name),+)] = [
            $(BuiltinIntegerInputCapability {
                name: $name,
                classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
                availability: BuiltinIntegerInputAvailability::Documented,
                scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
                notes: "Numeric input is checked before conversion to canonical binary64 storage.",
            }),+
        ];
        const $capabilities: [BuiltinIntegerCapabilityDescriptor; 1] =
            [BuiltinIntegerCapabilityDescriptor {
                form: $form,
                inputs: &$inputs,
                computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
                output_class: BuiltinIntegerOutputClassRule::NotApplicable,
                overflow: BuiltinIntegerOverflowRule::NotApplicable,
                backend: BuiltinIntegerBackendRule::HostOnly,
                overload: BuiltinIntegerOverloadKind::FunctionSpecific,
                notes: "Response-spectrum authoring is a portable host construction boundary.",
            }];
    };
    (@count $head:literal $(, $tail:literal)*) => {
        1usize $(+ numeric_capabilities!(@one $tail))*
    };
    (@one $value:literal) => { 1usize };
}

numeric_capabilities!(
    AXIS_NUMERIC_INPUTS,
    AXIS_NUMERIC_CAPABILITIES,
    "typed response-spectrum axis",
    ["values"]
);
numeric_capabilities!(
    DIRECTION_NUMERIC_INPUTS,
    DIRECTION_NUMERIC_CAPABILITIES,
    "typed response-spectrum direction",
    ["vector"]
);
numeric_capabilities!(
    PERCENTAGE_NUMERIC_INPUTS,
    PERCENTAGE_NUMERIC_CAPABILITIES,
    "typed response-spectrum directional percentage rule",
    ["primary", "secondary"]
);
numeric_capabilities!(
    TRUNCATION_NUMERIC_INPUTS,
    TRUNCATION_NUMERIC_CAPABILITIES,
    "typed response-spectrum modal truncation policy",
    ["minimumMassRatio"]
);
numeric_capabilities!(
    EVIDENCE_NUMERIC_INPUTS,
    EVIDENCE_NUMERIC_CAPABILITIES,
    "typed response-spectrum evidence",
    ["samplingAdequacyRatio", "maximumInterpolationError"]
);

macro_rules! unary_policy {
    ($function:ident, $descriptor:ident, $signatures:ident, $builtin:literal, $label:literal,
     $summary:literal, $class:expr, $body:expr) => {
        const $signatures: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor { label: $label, inputs: &SCALAR_INPUTS, outputs: &OUT_ANY }];
        pub(super) const $descriptor: BuiltinDescriptor = BuiltinDescriptor { signatures: &$signatures, output_mode: BuiltinOutputMode::Fixed, completion_policy: BuiltinCompletionPolicy::Public, errors: &ERRORS };
        #[runtime_builtin(name = $builtin, category = "fea", summary = $summary, keywords = "fea,response,spectrum,typed", descriptor($descriptor), integer_capabilities(SCALAR_NUMERIC_CAPABILITIES), builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters")]
        pub async fn $function(value: Value) -> BuiltinResult<Value> {
            let value = finite_numeric_scalar($builtin, &value, "value")?;
            typed_value($builtin, $class, ($body)(value)?)
        }
    };
}

pub(crate) mod choices;
pub(crate) mod parameters;

pub(super) fn axis(b: &'static str, v: &Value) -> BuiltinResult<SpectrumAbscissa> {
    typed_value_from(b, v, AXIS_CLASS)
}
pub(super) fn damping(b: &'static str, v: &Value) -> BuiltinResult<SpectrumDamping> {
    typed_value_from(b, v, DAMPING_CLASS)
}
pub(super) fn direction(b: &'static str, v: &Value) -> BuiltinResult<SpectrumDirection> {
    typed_value_from(b, v, DIRECTION_CLASS)
}
pub(super) fn response(b: &'static str, v: &Value) -> BuiltinResult<SpectrumResponse> {
    typed_value_from(b, v, RESPONSE_CLASS)
}
pub(super) fn envelope(b: &'static str, v: &Value) -> BuiltinResult<SpectrumEnvelope> {
    typed_value_from(b, v, ENVELOPE_CLASS)
}
pub(super) fn interpolation(b: &'static str, v: &Value) -> BuiltinResult<SpectrumInterpolation> {
    typed_value_from(b, v, INTERPOLATION_CLASS)
}
pub(super) fn extrapolation(b: &'static str, v: &Value) -> BuiltinResult<SpectrumExtrapolation> {
    typed_value_from(b, v, EXTRAPOLATION_CLASS)
}
pub(super) fn modal_combination(b: &'static str, v: &Value) -> BuiltinResult<ModalCombination> {
    typed_value_from(b, v, MODAL_COMBINATION_CLASS)
}
pub(super) fn directional_combination(
    b: &'static str,
    v: &Value,
) -> BuiltinResult<DirectionalCombination> {
    typed_value_from(b, v, DIRECTIONAL_COMBINATION_CLASS)
}
pub(super) fn truncation(b: &'static str, v: &Value) -> BuiltinResult<ModalTruncationPolicy> {
    typed_value_from(b, v, TRUNCATION_CLASS)
}
pub(super) fn construction(b: &'static str, v: &Value) -> BuiltinResult<SpectrumConstruction> {
    typed_value_from(b, v, CONSTRUCTION_CLASS)
}
pub(super) fn evidence(b: &'static str, v: &Value) -> BuiltinResult<SpectrumEvidence> {
    typed_value_from(b, v, EVIDENCE_CLASS)
}

type Schema = (
    &'static str,
    &'static str,
    &'static [(&'static str, &'static str)],
);
fn schemas() -> [Schema; 12] {
    [
        (
            AXIS_CLASS,
            "runmat.fea.spectrum-axis",
            &[
                ("frequencyHertz", "fea.SpectrumAxis.frequencyHertz"),
                ("periodSeconds", "fea.SpectrumAxis.periodSeconds"),
            ],
        ),
        (
            DAMPING_CLASS,
            "runmat.fea.spectrum-damping",
            &[
                ("ratio", "fea.SpectrumDamping.ratio"),
                ("qualityFactor", "fea.SpectrumDamping.qualityFactor"),
            ],
        ),
        (
            DIRECTION_CLASS,
            "runmat.fea.spectrum-direction",
            &[("new", "fea.SpectrumDirection.new")],
        ),
        (
            RESPONSE_CLASS,
            "runmat.fea.spectrum-response",
            &[
                (
                    "relativeDisplacement",
                    "fea.SpectrumResponse.relativeDisplacement",
                ),
                ("relativeVelocity", "fea.SpectrumResponse.relativeVelocity"),
                (
                    "relativeAcceleration",
                    "fea.SpectrumResponse.relativeAcceleration",
                ),
                (
                    "absoluteAcceleration",
                    "fea.SpectrumResponse.absoluteAcceleration",
                ),
                ("pseudoVelocity", "fea.SpectrumResponse.pseudoVelocity"),
                (
                    "pseudoAcceleration",
                    "fea.SpectrumResponse.pseudoAcceleration",
                ),
            ],
        ),
        (
            ENVELOPE_CLASS,
            "runmat.fea.spectrum-envelope",
            &[
                ("positive", "fea.SpectrumEnvelope.positive"),
                ("negative", "fea.SpectrumEnvelope.negative"),
                ("maximax", "fea.SpectrumEnvelope.maximax"),
            ],
        ),
        (
            INTERPOLATION_CLASS,
            "runmat.fea.spectrum-interpolation",
            &[
                (
                    "linearFrequency",
                    "fea.SpectrumInterpolation.linearFrequency",
                ),
                ("linearPeriod", "fea.SpectrumInterpolation.linearPeriod"),
                ("logLog", "fea.SpectrumInterpolation.logLog"),
            ],
        ),
        (
            EXTRAPOLATION_CLASS,
            "runmat.fea.spectrum-extrapolation",
            &[
                ("reject", "fea.SpectrumExtrapolation.reject"),
                ("holdEndpoint", "fea.SpectrumExtrapolation.holdEndpoint"),
            ],
        ),
        (
            MODAL_COMBINATION_CLASS,
            "runmat.fea.modal-combination",
            &[
                ("srss", "fea.ModalCombination.srss"),
                (
                    "completeQuadratic",
                    "fea.ModalCombination.completeQuadratic",
                ),
                ("tenPercent", "fea.ModalCombination.tenPercent"),
                ("absoluteSum", "fea.ModalCombination.absoluteSum"),
            ],
        ),
        (
            DIRECTIONAL_COMBINATION_CLASS,
            "runmat.fea.directional-combination",
            &[
                ("srss", "fea.DirectionalCombination.srss"),
                ("absoluteSum", "fea.DirectionalCombination.absoluteSum"),
                ("percentage", "fea.DirectionalCombination.percentage"),
            ],
        ),
        (
            TRUNCATION_CLASS,
            "runmat.fea.modal-truncation",
            &[("new", "fea.ModalTruncation.new")],
        ),
        (
            CONSTRUCTION_CLASS,
            "runmat.fea.spectrum-construction",
            &[
                ("authoredDesign", "fea.SpectrumConstruction.authoredDesign"),
                (
                    "fromAccelerationHistoryExact",
                    "fea.SpectrumConstruction.fromAccelerationHistoryExact",
                ),
                (
                    "fromAccelerationHistoryRecursive",
                    "fea.SpectrumConstruction.fromAccelerationHistoryRecursive",
                ),
            ],
        ),
        (
            EVIDENCE_CLASS,
            "runmat.fea.spectrum-evidence",
            &[("bounded", "fea.SpectrumEvidence.bounded")],
        ),
    ]
}

pub(super) fn ensure_spectrum_policy_classes_registered() {
    static PORTABLE: Once = Once::new();
    static RUNTIME: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(AXIS_CLASS);
    PORTABLE.call_once(|| {
        for (class_name, type_identity, _) in schemas() {
            crate::execution::value_codec::register_immutable_value_class(
                crate::execution::value_codec::ImmutableValueClassSchema {
                    class_name,
                    type_identity,
                    schema_version: 1,
                    allowed_fields: VALUE_FIELDS,
                    required_fields: VALUE_FIELDS,
                    validate: validate_value_object,
                },
            )
            .unwrap_or_else(|error| panic!("failed to register {class_name}: {error}"));
        }
    });
    RUNTIME.ensure(|| {
        for (class_name, _, methods) in schemas() {
            register_class(class_name, methods);
        }
    });
}

fn register_class(class_name: &str, methods: &[(&str, &str)]) {
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        name: class_name.into(),
        parent: None,
        properties: HashMap::from([(
            "value".into(),
            crate::class_registry::RuntimeProperty {
                name: "value".into(),
                is_static: false,
                is_constant: true,
                is_dependent: false,
                get_access: MemberAccess::Public,
                set_access: MemberAccess::Private,
                default_value: None,
            },
        )]),
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
            .collect(),
    });
}

fn validate_value_object(object: &ObjectInstance) -> Result<(), String> {
    if object.properties.contains_key("value") {
        Ok(())
    } else {
        Err(format!("{}.value is required", object.class_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::constraints::support::frame::coordinate_frame_global_builtin;
    use crate::execution::value_codec::{decode_inline_value, encode_inline_value};

    #[tokio::test]
    async fn direction_is_normalized_and_portable() {
        let value = parameters::spectrum_direction(
            coordinate_frame_global_builtin()
                .await
                .expect("global frame"),
            Tensor::new(vec![3.0, 4.0, 0.0], vec![1, 3])
                .expect("direction tensor")
                .into(),
        )
        .await
        .expect("direction");
        let decoded =
            decode_inline_value(&encode_inline_value(&value).expect("encode")).expect("decode");
        assert_eq!(decoded, value);
        let direction = direction("test", &decoded).expect("typed direction");
        assert_eq!(direction.vector, [0.6, 0.8, 0.0]);
    }

    #[tokio::test]
    async fn construction_and_evidence_reject_invalid_sources_and_ranges() {
        let constant = super::super::signals::acceleration_constant(
            "constant".into(),
            Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3])
                .expect("acceleration tensor")
                .into(),
        )
        .await
        .expect("constant acceleration");
        let error = parameters::derived_exact(constant)
            .await
            .expect_err("constant acceleration is not a history");
        assert!(error.to_string().contains("time-domain acceleration"));

        let error = parameters::spectrum_evidence(1.01.into(), 0.0.into())
            .await
            .expect_err("sampling adequacy above one must fail");
        assert!(error.to_string().contains("[0, 1]"));
    }
}
