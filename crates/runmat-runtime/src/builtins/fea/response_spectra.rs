use std::collections::HashMap;
use std::sync::Once;

use super::*;
use runmat_analysis_core::{ResponseSpectrum, ResponseSpectrumId};

pub(super) const RESPONSE_SPECTRUM_CLASS: &str = "fea.ResponseSpectrum";
const BUILTIN: &str = "fea.ResponseSpectrum.new";
const FIELDS: &[&str] = &[
    "spectrum_id",
    "abscissa",
    "ordinates",
    "damping",
    "response",
    "envelope",
    "interpolation",
    "extrapolation",
    "direction",
    "modal_combination",
    "directional_combination",
    "truncation",
    "rigid_response",
    "construction",
    "evidence",
    "provenance_digest",
];

const INPUTS: [BuiltinParamDescriptor; 16] = [
    required(
        "id",
        BuiltinParamType::StringScalar,
        "Stable response-spectrum identifier.",
    ),
    required("axis", BuiltinParamType::Any, "Typed fea.SpectrumAxis."),
    required(
        "ordinates",
        BuiltinParamType::NumericArray,
        "Finite nonnegative spectrum ordinates.",
    ),
    required(
        "damping",
        BuiltinParamType::Any,
        "Typed fea.SpectrumDamping.",
    ),
    required(
        "response",
        BuiltinParamType::Any,
        "Typed fea.SpectrumResponse.",
    ),
    required(
        "envelope",
        BuiltinParamType::Any,
        "Typed fea.SpectrumEnvelope.",
    ),
    required(
        "interpolation",
        BuiltinParamType::Any,
        "Typed fea.SpectrumInterpolation.",
    ),
    required(
        "extrapolation",
        BuiltinParamType::Any,
        "Typed fea.SpectrumExtrapolation.",
    ),
    required(
        "direction",
        BuiltinParamType::Any,
        "Typed fea.SpectrumDirection.",
    ),
    required(
        "modalCombination",
        BuiltinParamType::Any,
        "Typed fea.ModalCombination.",
    ),
    required(
        "directionalCombination",
        BuiltinParamType::Any,
        "Typed fea.DirectionalCombination.",
    ),
    required(
        "truncation",
        BuiltinParamType::Any,
        "Typed fea.ModalTruncation.",
    ),
    required(
        "rigidResponse",
        BuiltinParamType::Any,
        "Typed fea.RigidResponse policy.",
    ),
    required(
        "construction",
        BuiltinParamType::Any,
        "Typed fea.SpectrumConstruction.",
    ),
    required(
        "evidence",
        BuiltinParamType::Any,
        "Typed fea.SpectrumEvidence.",
    ),
    required(
        "provenanceDigest",
        BuiltinParamType::StringScalar,
        "Nonempty provenance digest for the spectrum data.",
    ),
];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "spectrum = fea.ResponseSpectrum.new(id, axis, ordinates, damping, response, envelope, interpolation, extrapolation, direction, modalCombination, directionalCombination, truncation, rigidResponse, construction, evidence, provenanceDigest)",
    inputs: &INPUTS,
    outputs: &OUT_ANY,
}];
pub(super) const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &ERRORS,
};
const INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "ordinates",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::Documented,
    scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
    notes: "Spectrum ordinates are checked before conversion to canonical binary64 storage.",
}];
const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "typed response spectrum",
        inputs: &INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Response-spectrum authoring produces an immutable portable host value.",
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

#[runtime_builtin(
    name = "fea.ResponseSpectrum.new",
    category = "fea",
    summary = "Create an immutable typed response spectrum.",
    keywords = "fea,response,spectrum,srs,modal,typed",
    descriptor(crate::builtins::fea::response_spectra::DESCRIPTOR),
    integer_capabilities(INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::fea::response_spectra"
)]
pub async fn response_spectrum(args: Vec<Value>) -> BuiltinResult<Value> {
    let [id, axis, ordinates, damping, response, envelope, interpolation, extrapolation, direction, modal_combination, directional_combination, truncation, rigid_response, construction, evidence, provenance_digest] =
        <[Value; 16]>::try_from(args).map_err(|args| {
            builtin_error(
                BUILTIN,
                &ERROR_INPUT,
                format!("{BUILTIN} requires 16 arguments; got {}", args.len()),
            )
        })?;
    let id = scalar_string(&id, BUILTIN, &ERROR_INPUT)?;
    let provenance_digest = scalar_string(&provenance_digest, BUILTIN, &ERROR_INPUT)?;
    if id.trim().is_empty() || provenance_digest.trim().is_empty() {
        return Err(builtin_error(
            BUILTIN,
            &ERROR_INPUT,
            "response-spectrum identifier and provenance digest must not be empty",
        ));
    }
    let abscissa = super::response_spectrum_policies::axis(BUILTIN, &axis)?;
    let ordinates = super::values::finite_numeric_vector(BUILTIN, &ordinates, "ordinates")?;
    let interpolation = super::response_spectrum_policies::interpolation(BUILTIN, &interpolation)?;
    let spectrum = ResponseSpectrum {
        spectrum_id: ResponseSpectrumId(id),
        abscissa,
        ordinates,
        damping: super::response_spectrum_policies::damping(BUILTIN, &damping)?,
        response: super::response_spectrum_policies::response(BUILTIN, &response)?,
        envelope: super::response_spectrum_policies::envelope(BUILTIN, &envelope)?,
        interpolation,
        extrapolation: super::response_spectrum_policies::extrapolation(BUILTIN, &extrapolation)?,
        direction: super::response_spectrum_policies::direction(BUILTIN, &direction)?,
        modal_combination: super::response_spectrum_policies::modal_combination(
            BUILTIN,
            &modal_combination,
        )?,
        directional_combination: super::response_spectrum_policies::directional_combination(
            BUILTIN,
            &directional_combination,
        )?,
        truncation: super::response_spectrum_policies::truncation(BUILTIN, &truncation)?,
        rigid_response: super::response_spectrum_policies::rigid_response(
            BUILTIN,
            &rigid_response,
        )?,
        construction: super::response_spectrum_policies::construction(BUILTIN, &construction)?,
        evidence: super::response_spectrum_policies::evidence(BUILTIN, &evidence)?,
        provenance_digest,
    };
    spectrum
        .validate_intrinsic()
        .map_err(|detail| builtin_error(BUILTIN, &ERROR_INPUT, detail))?;
    serializable_to_object(BUILTIN, &ERROR_INTERNAL, RESPONSE_SPECTRUM_CLASS, &spectrum)
}

pub(super) fn response_spectrum_vec_from_value(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<ResponseSpectrum>> {
    super::modeling::object_vec_from_value(builtin, value, RESPONSE_SPECTRUM_CLASS)
}

pub(super) fn ensure_response_spectrum_class_registered() {
    static PORTABLE_SCHEMA: Once = Once::new();
    static RUNTIME_CLASS: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(RESPONSE_SPECTRUM_CLASS);
    PORTABLE_SCHEMA.call_once(|| {
        crate::execution::value_codec::register_immutable_value_class(
            crate::execution::value_codec::ImmutableValueClassSchema {
                class_name: RESPONSE_SPECTRUM_CLASS,
                type_identity: "runmat.fea.response-spectrum",
                schema_version: 1,
                allowed_fields: FIELDS,
                required_fields: FIELDS,
                validate: validate_object,
            },
        )
        .unwrap_or_else(|error| panic!("failed to register {RESPONSE_SPECTRUM_CLASS}: {error}"));
    });
    RUNTIME_CLASS.ensure(|| {
        crate::class_registry::register_class(crate::class_registry::RuntimeClass {
            name: RESPONSE_SPECTRUM_CLASS.into(),
            parent: None,
            properties: FIELDS
                .iter()
                .map(|name| {
                    (
                        (*name).into(),
                        crate::class_registry::RuntimeProperty {
                            name: (*name).into(),
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
            methods: HashMap::from([(
                "new".into(),
                crate::class_registry::RuntimeMethod {
                    name: "new".into(),
                    is_static: true,
                    is_abstract: false,
                    is_sealed: false,
                    access: MemberAccess::Public,
                    function_name: BUILTIN.into(),
                    implicit_class_argument: None,
                },
            )]),
        });
    });
}

fn validate_object(object: &ObjectInstance) -> Result<(), String> {
    let missing = FIELDS
        .iter()
        .filter(|field| !object.properties.contains_key(**field))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "{}.{} is required",
            object.class_name,
            missing.join(", ")
        ));
    }
    let spectrum: ResponseSpectrum = object_payload(
        BUILTIN,
        &Value::Object(object.clone()),
        RESPONSE_SPECTRUM_CLASS,
    )
    .map_err(|error| error.to_string())?;
    spectrum.validate_intrinsic()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::fea::constraints::support::frame::coordinate_frame_global_builtin;
    use crate::execution::value_codec::{decode_inline_value, encode_inline_value};

    async fn valid_spectrum() -> BuiltinResult<Value> {
        let axis = super::super::response_spectrum_policies::parameters::frequency_axis(
            Tensor::new(vec![5.0, 10.0, 20.0], vec![1, 3])?.into(),
        )
        .await?;
        let damping =
            super::super::response_spectrum_policies::parameters::damping_ratio(0.05.into())
                .await?;
        let response =
            super::super::response_spectrum_policies::choices::response_absolute_acceleration()
                .await?;
        let envelope =
            super::super::response_spectrum_policies::choices::envelope_maximax().await?;
        let interpolation =
            super::super::response_spectrum_policies::choices::interpolation_frequency().await?;
        let extrapolation =
            super::super::response_spectrum_policies::choices::extrapolation_reject().await?;
        let direction = super::super::response_spectrum_policies::parameters::spectrum_direction(
            coordinate_frame_global_builtin().await?,
            Tensor::new(vec![1.0, 0.0, 0.0], vec![1, 3])?.into(),
        )
        .await?;
        let modal = super::super::response_spectrum_policies::choices::modal_cqc().await?;
        let directional =
            super::super::response_spectrum_policies::choices::directional_srss().await?;
        let correction =
            super::super::response_spectrum_policies::parameters::truncation_missing_mass(
                9.0.into(),
            )
            .await?;
        let truncation = super::super::response_spectrum_policies::parameters::modal_truncation(
            0.9.into(),
            correction,
        )
        .await?;
        let rigid_response =
            super::super::response_spectrum_policies::parameters::rigid_response_lindley_yow(
                9.0.into(),
            )
            .await?;
        let construction =
            super::super::response_spectrum_policies::parameters::authored_design().await?;
        let evidence = super::super::response_spectrum_policies::parameters::spectrum_evidence(
            0.95.into(),
            0.01.into(),
        )
        .await?;
        response_spectrum(vec![
            Value::String("launch".into()),
            axis,
            Tensor::new(vec![12.0, 18.0, 9.0], vec![1, 3])?.into(),
            damping,
            response,
            envelope,
            interpolation,
            extrapolation,
            direction,
            modal,
            directional,
            truncation,
            rigid_response,
            construction,
            evidence,
            Value::String("sha256:launch-spectrum".into()),
        ])
        .await
    }

    #[tokio::test]
    async fn typed_spectrum_round_trips_portably() {
        let value = valid_spectrum().await.expect("typed spectrum");
        let decoded =
            decode_inline_value(&encode_inline_value(&value).expect("encode")).expect("decode");
        let spectra = response_spectrum_vec_from_value(BUILTIN, &decoded).expect("decode class");
        assert_eq!(spectra.len(), 1);
        assert_eq!(spectra[0].spectrum_id.0, "launch");
        assert_eq!(spectra[0].ordinates, vec![12.0, 18.0, 9.0]);
    }
}
