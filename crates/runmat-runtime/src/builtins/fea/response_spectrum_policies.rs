use std::sync::Once;

use super::*;
use crate::builtins::fea::constraints::support::frame::coordinate_frame_from_value;
use crate::builtins::fea::object_projection::register_value_class;
use crate::builtins::fea::values::{
    finite_numeric_scalar, finite_numeric_vector, finite_numeric_vector3,
};
use crate::class_registry::declaration::{immutable_properties, static_methods};
use runmat_analysis_core::{
    DirectionalCombination, ModalCombination, ModalTruncationCorrection, ModalTruncationPolicy,
    RigidResponsePolicy, ShockResponseSpectrumMethod, SpectrumAbscissa, SpectrumConstruction,
    SpectrumDamping, SpectrumDirection, SpectrumEnvelope, SpectrumEvidence, SpectrumExtrapolation,
    SpectrumInterpolation, SpectrumResponse,
};

use runmat_builtins::fea_contract::policy_types::classes;

pub(super) const AXIS_CLASS: &str = classes::SPECTRUM_AXIS.display_name();
pub(super) const DAMPING_CLASS: &str = classes::SPECTRUM_DAMPING.display_name();
pub(super) const DIRECTION_CLASS: &str = classes::SPECTRUM_DIRECTION.display_name();
pub(super) const RESPONSE_CLASS: &str = classes::SPECTRUM_RESPONSE.display_name();
pub(super) const ENVELOPE_CLASS: &str = classes::SPECTRUM_ENVELOPE.display_name();
pub(super) const INTERPOLATION_CLASS: &str = classes::SPECTRUM_INTERPOLATION.display_name();
pub(super) const EXTRAPOLATION_CLASS: &str = classes::SPECTRUM_EXTRAPOLATION.display_name();
pub(super) const MODAL_COMBINATION_CLASS: &str = classes::MODAL_COMBINATION.display_name();
pub(super) const DIRECTIONAL_COMBINATION_CLASS: &str =
    classes::DIRECTIONAL_COMBINATION.display_name();
pub(super) const TRUNCATION_CLASS: &str = classes::MODAL_TRUNCATION.display_name();
pub(super) const TRUNCATION_CORRECTION_CLASS: &str =
    classes::MODAL_TRUNCATION_CORRECTION.display_name();
pub(super) const RIGID_RESPONSE_CLASS: &str = classes::RIGID_RESPONSE.display_name();
pub(super) const CONSTRUCTION_CLASS: &str = classes::SPECTRUM_CONSTRUCTION.display_name();
pub(super) const EVIDENCE_CLASS: &str = classes::SPECTRUM_EVIDENCE.display_name();

const VALUE_FIELDS: &[&str] = &["value"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TypedValue<T> {
    value: T,
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

macro_rules! unary_policy {
    ($function:ident, $builtin:literal, $class:expr, $body:expr) => {
        #[runtime_builtin(name = $builtin, binding_variant = "default", builtin_path = "crate::builtins::fea::response_spectrum_policies::parameters")]
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
pub(super) fn rigid_response(b: &'static str, v: &Value) -> BuiltinResult<RigidResponsePolicy> {
    typed_value_from(b, v, RIGID_RESPONSE_CLASS)
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
fn schemas() -> [Schema; 14] {
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
            TRUNCATION_CORRECTION_CLASS,
            "runmat.fea.modal-truncation-correction",
            &[
                ("none", "fea.ModalTruncationCorrection.none"),
                ("residualMode", "fea.ModalTruncationCorrection.residualMode"),
                ("missingMass", "fea.ModalTruncationCorrection.missingMass"),
            ],
        ),
        (
            RIGID_RESPONSE_CLASS,
            "runmat.fea.rigid-response",
            &[
                ("disabled", "fea.RigidResponse.disabled"),
                ("gupta", "fea.RigidResponse.gupta"),
                ("lindleyYow", "fea.RigidResponse.lindleyYow"),
            ],
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
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            AXIS_CLASS,
        ));
    PORTABLE.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let mut registrations = Vec::new();
        for (class_name, type_identity, _) in schemas() {
            registrations.push(crate::execution::value_codec::ImmutableValueClassSchema {
                class_name,
                type_identity,
                schema_version: 1,
                allowed_fields: VALUE_FIELDS,
                required_fields: VALUE_FIELDS,
                validate: validate_value_object,
            });
        }

        crate::builtins::fea::object_projection::register_schemas(
            &PROJECTION_FIELDS,
            registrations,
        );
    });
    RUNTIME.ensure(|| {
        for (class_name, _, methods) in schemas() {
            register_value_class(
                class_name,
                immutable_properties(VALUE_FIELDS),
                static_methods(methods.iter().copied()),
            );
        }
    });
}

fn validate_value_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    if object.properties().contains_key("value") {
        Ok(())
    } else {
        Err(format!("{}.value is required", object.class_name()))
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
            coordinate_frame_global_builtin(None)
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
