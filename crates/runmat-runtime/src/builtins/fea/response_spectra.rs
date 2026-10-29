use std::sync::Once;

use super::*;
use crate::builtins::fea::object_projection::register_value_class;
use crate::class_registry::declaration::{immutable_properties, static_methods};
use runmat_analysis_core::{ResponseSpectrum, ResponseSpectrumId};

pub(super) const RESPONSE_SPECTRUM_CLASS: &str =
    runmat_builtins::fea_contract::policy_types::classes::RESPONSE_SPECTRUM.display_name();
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
];

#[runtime_builtin(
    name = "fea.ResponseSpectrum.new",
    binding_variant = "default",
    builtin_path = "crate::builtins::fea::response_spectra"
)]
pub async fn response_spectrum(args: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    let args: Vec<Value> = args.into_iter().collect();
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
        ordinates: runmat_analysis_core::NumericalDataSource::authored(
            ordinates,
            provenance_digest,
        ),
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
        crate::class_registry::ClassRegistration::new(runmat_types::StaticClassIdentity::new(
            RESPONSE_SPECTRUM_CLASS,
        ));
    PORTABLE_SCHEMA.call_once(|| {
        static PROJECTION_FIELDS: crate::builtins::fea::object_projection::ProjectionFields =
            std::sync::OnceLock::new();
        let schemas = vec![crate::execution::value_codec::ImmutableValueClassSchema {
            class_name: RESPONSE_SPECTRUM_CLASS,
            type_identity: "runmat.fea.response-spectrum",
            schema_version: 1,
            allowed_fields: FIELDS,
            required_fields: FIELDS,
            validate: validate_object,
        }];

        crate::builtins::fea::object_projection::register_schemas(&PROJECTION_FIELDS, schemas);
    });
    RUNTIME_CLASS.ensure(|| {
        register_value_class(
            RESPONSE_SPECTRUM_CLASS,
            immutable_properties(FIELDS),
            static_methods([("new", BUILTIN)]),
        );
    });
}

fn validate_object(object: &ObjectInstance) -> Result<(), String> {
    crate::builtins::fea::object_projection::validate(object)?;
    let missing = FIELDS
        .iter()
        .filter(|field| !object.properties().contains_key(field))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "{}.{} is required",
            object.class_name(),
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
            coordinate_frame_global_builtin(None).await?,
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
        response_spectrum(crate::builtins::common::test_support::builtin_arguments(
            vec![
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
            ],
        ))
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
        assert_eq!(
            spectra[0].resolved_ordinates().expect("authored ordinates"),
            [12.0, 18.0, 9.0]
        );
    }
}
