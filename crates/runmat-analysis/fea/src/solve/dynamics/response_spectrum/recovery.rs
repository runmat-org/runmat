use runmat_analysis_core::{ModalTruncationCorrection, SpectrumResponse};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary, contracts::FeaRunError, pipeline::modal_basis::ModalBasisArtifact,
    post::fields::StructuralLinearFieldQuantity, solve::preconditioner::SpdPreconditionerKind,
};

use super::{
    ModalResponseSpectrumResult, ModalResponseSpectrumResultIdentity,
    ResponseSpectrumDisplacementConversion, ResponseSpectrumResultSemantics,
};

const FIELD_RECOVERY_PREFIX: &[u8] = b"runmat-modal-response-spectrum-field-recovery\0";
const FIELD_RECOVERY_LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 65_536, 24));

mod deformation;
mod nodal;

pub use nodal::{
    ResponseSpectrumNodalEnvelope, ResponseSpectrumNodalField, ResponseSpectrumNodalQuantity,
};

use deformation::recover_deformation_fields;
use nodal::recover_nodal_envelope;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseSpectrumDeformationRecoveryUnavailability {
    ResponseQuantityNotDisplacementConvertible { response: SpectrumResponse },
    MissingMassStaticCorrectionRequiresPseudoAcceleration { response: SpectrumResponse },
}

impl std::fmt::Display for ResponseSpectrumDeformationRecoveryUnavailability {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResponseQuantityNotDisplacementConvertible { response } => write!(
                formatter,
                "{response:?} response cannot be converted into a signed displacement contribution for linear field recovery"
            ),
            Self::MissingMassStaticCorrectionRequiresPseudoAcceleration { response } => write!(
                formatter,
                "missing-mass deformation recovery requires pseudo-acceleration ordinates, not {response:?}",
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseSpectrumDeformationRecovery {
    Recovered {
        conversion: ResponseSpectrumDisplacementConversion,
        /// Preconditioner prepared for missing-mass correction; an all-zero
        /// correction can bypass the solve while retaining this selection.
        missing_mass_static_preconditioner: Option<SpdPreconditionerKind>,
        fields: Vec<ResponseSpectrumLinearField>,
    },
    Unavailable {
        reason: ResponseSpectrumDeformationRecoveryUnavailability,
    },
}

impl ResponseSpectrumDeformationRecovery {
    pub const fn missing_mass_static_preconditioner(&self) -> Option<SpdPreconditionerKind> {
        match self {
            Self::Recovered {
                missing_mass_static_preconditioner,
                ..
            } => *missing_mass_static_preconditioner,
            Self::Unavailable { .. } => None,
        }
    }

    pub const fn conversion(&self) -> Option<ResponseSpectrumDisplacementConversion> {
        match self {
            Self::Recovered { conversion, .. } => Some(*conversion),
            Self::Unavailable { .. } => None,
        }
    }

    pub fn fields(&self) -> Option<&[ResponseSpectrumLinearField]> {
        match self {
            Self::Recovered { fields, .. } => Some(fields),
            Self::Unavailable { .. } => None,
        }
    }

    pub fn field(
        &self,
        quantity: StructuralLinearFieldQuantity,
    ) -> Option<&ResponseSpectrumLinearField> {
        self.fields()?
            .iter()
            .find(|field| field.quantity == quantity)
    }

    pub const fn unavailability(
        &self,
    ) -> Option<ResponseSpectrumDeformationRecoveryUnavailability> {
        match self {
            Self::Recovered { .. } => None,
            Self::Unavailable { reason } => Some(*reason),
        }
    }
}

/// One statistically combined component envelope of a signed linear recovery quantity.
///
/// `values` are nonnegative component magnitudes. They do not describe a simultaneous signed
/// state and therefore cannot be fed back into ordinary deformation recovery.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseSpectrumLinearField {
    quantity: StructuralLinearFieldQuantity,
    shape: Vec<usize>,
    values: Vec<f64>,
}

impl ResponseSpectrumLinearField {
    pub const fn quantity(&self) -> StructuralLinearFieldQuantity {
        self.quantity
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    fn validate(&self) -> Result<(), FeaRunError> {
        let element_count = self
            .shape
            .iter()
            .try_fold(1_usize, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| FeaRunError::Solve("response-spectrum field shape overflows".into()))?;
        if self.shape.is_empty()
            || self.shape.contains(&0)
            || element_count != self.values.len()
            || self
                .values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(FeaRunError::Solve(
                "response-spectrum linear field is not a finite component envelope".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResponseSpectrumFieldRecoveryIdentity(Digest);

impl ResponseSpectrumFieldRecoveryIdentity {
    pub const fn digest(self) -> Digest {
        self.0
    }
}

/// Canonical typed nodal envelope and optional deformation recovery for one response-spectrum result.
///
/// Nodal values retain the response quantity exactly. Deformation fields are a separate contract
/// because only displacement and pseudo-response ordinates admit a qualified displacement
/// conversion. Every value remains a nonconcurrent statistical component magnitude.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseSpectrumFieldRecovery {
    identity: ResponseSpectrumFieldRecoveryIdentity,
    source_result: ModalResponseSpectrumResultIdentity,
    response: SpectrumResponse,
    semantics: ResponseSpectrumResultSemantics,
    nodal: ResponseSpectrumNodalEnvelope,
    deformation: ResponseSpectrumDeformationRecovery,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseSpectrumFieldRecoveryWire {
    identity: ResponseSpectrumFieldRecoveryIdentity,
    source_result: ModalResponseSpectrumResultIdentity,
    response: SpectrumResponse,
    semantics: ResponseSpectrumResultSemantics,
    nodal: ResponseSpectrumNodalEnvelope,
    deformation: ResponseSpectrumDeformationRecovery,
}

impl TryFrom<ResponseSpectrumFieldRecoveryWire> for ResponseSpectrumFieldRecovery {
    type Error = FeaRunError;

    fn try_from(wire: ResponseSpectrumFieldRecoveryWire) -> Result<Self, Self::Error> {
        let envelope = Self {
            identity: wire.identity,
            source_result: wire.source_result,
            response: wire.response,
            semantics: wire.semantics,
            nodal: wire.nodal,
            deformation: wire.deformation,
        };
        envelope.validate()?;
        Ok(envelope)
    }
}

impl<'de> Deserialize<'de> for ResponseSpectrumFieldRecovery {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        ResponseSpectrumFieldRecoveryWire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl ResponseSpectrumFieldRecovery {
    pub fn identity(&self) -> Result<Digest, FeaRunError> {
        self.validate()?;
        Ok(self.canonical_identity().digest())
    }

    pub const fn canonical_identity(&self) -> ResponseSpectrumFieldRecoveryIdentity {
        self.identity
    }

    pub const fn source_result(&self) -> ModalResponseSpectrumResultIdentity {
        self.source_result
    }

    pub const fn semantics(&self) -> ResponseSpectrumResultSemantics {
        self.semantics
    }

    pub const fn response(&self) -> SpectrumResponse {
        self.response
    }

    pub const fn nodal(&self) -> &ResponseSpectrumNodalEnvelope {
        &self.nodal
    }

    pub const fn deformation(&self) -> &ResponseSpectrumDeformationRecovery {
        &self.deformation
    }

    pub fn fields(&self) -> Option<&[ResponseSpectrumLinearField]> {
        self.deformation.fields()
    }

    pub fn field(
        &self,
        quantity: StructuralLinearFieldQuantity,
    ) -> Option<&ResponseSpectrumLinearField> {
        self.deformation.field(quantity)
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, FeaRunError> {
        self.validate()?;
        encode_contract(
            FIELD_RECOVERY_PREFIX,
            "modal-response-spectrum-field-recovery",
            self,
            FIELD_RECOVERY_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, FeaRunError> {
        decode_contract(
            FIELD_RECOVERY_PREFIX,
            "modal-response-spectrum-field-recovery",
            bytes,
            FIELD_RECOVERY_LIMITS,
        )
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }

    pub fn validate_for_result(
        &self,
        result: &ModalResponseSpectrumResult,
    ) -> Result<(), FeaRunError> {
        self.validate()?;
        if self.source_result != result.identity()
            || self.response != result.response()
            || self.semantics != result.semantics()
            || self.deformation.unavailability()
                != response_spectrum_deformation_recovery_unavailability(result)
            || self
                .deformation
                .missing_mass_static_preconditioner()
                .is_some()
                != (self.deformation.unavailability().is_none()
                    && has_missing_mass_correction(result))
        {
            return Err(FeaRunError::Solve(
                "response-spectrum field recovery differs from its numerical result".into(),
            ));
        }
        Ok(())
    }

    pub fn validate_exact_for(
        &self,
        summary: &AssemblySummary,
        basis: &ModalBasisArtifact,
        result: &ModalResponseSpectrumResult,
    ) -> Result<(), FeaRunError> {
        self.validate_for_result(result)?;
        if self != &recover_modal_response_spectrum_fields(summary, basis, result)? {
            return Err(FeaRunError::Solve(
                "response-spectrum field envelope differs from its operator, basis, or numerical result"
                    .into(),
            ));
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), FeaRunError> {
        self.nodal.validate_for(self.response)?;
        let deformation_is_valid = match &self.deformation {
            ResponseSpectrumDeformationRecovery::Recovered {
                conversion, fields, ..
            } => {
                Some(*conversion)
                    == ResponseSpectrumDisplacementConversion::for_response(self.response)
                    && !fields.is_empty()
                    && fields.iter().all(|field| field.validate().is_ok())
                    && !fields
                        .windows(2)
                        .any(|pair| pair[0].quantity >= pair[1].quantity)
            }
            ResponseSpectrumDeformationRecovery::Unavailable { reason } => {
                matches!(
                    reason,
                ResponseSpectrumDeformationRecoveryUnavailability::ResponseQuantityNotDisplacementConvertible { response }
                        if *response == self.response
                ) || matches!(
                    reason,
                    ResponseSpectrumDeformationRecoveryUnavailability::MissingMassStaticCorrectionRequiresPseudoAcceleration { response }
                        if *response == self.response && *response != SpectrumResponse::PseudoAcceleration
                )
            }
        };
        if self.semantics != ResponseSpectrumResultSemantics::StatisticalComponentMagnitudeEnvelope
            || !deformation_is_valid
            || self.identity != self.derive_identity()?
        {
            return Err(FeaRunError::Solve(
                "response-spectrum field envelope is not canonical".into(),
            ));
        }
        Ok(())
    }

    fn derive_identity(&self) -> Result<ResponseSpectrumFieldRecoveryIdentity, FeaRunError> {
        encode_contract(
            FIELD_RECOVERY_PREFIX,
            "modal-response-spectrum-field-recovery-identity",
            &(
                self.source_result,
                self.response,
                self.semantics,
                &self.nodal,
                &self.deformation,
            ),
            FIELD_RECOVERY_LIMITS,
        )
        .map(Digest::sha256)
        .map(ResponseSpectrumFieldRecoveryIdentity)
        .map_err(|error| FeaRunError::Solve(error.to_string()))
    }
}

pub fn recover_modal_response_spectrum_fields(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    result: &ModalResponseSpectrumResult,
) -> Result<ResponseSpectrumFieldRecovery, FeaRunError> {
    basis.validate_for(summary)?;
    result.validate()?;
    if result.basis() != basis.identity() {
        return Err(FeaRunError::Solve(
            "response-spectrum result names another modal basis".into(),
        ));
    }
    let nodal = recover_nodal_envelope(summary, result)?;
    let deformation = match response_spectrum_deformation_recovery_unavailability(result) {
        Some(reason) => ResponseSpectrumDeformationRecovery::Unavailable { reason },
        None => {
            let conversion = result
                .displacement_conversion()
                .expect("availability checked displacement conversion");
            let (fields, missing_mass_static_preconditioner) =
                recover_deformation_fields(summary, basis, result, conversion)?;
            ResponseSpectrumDeformationRecovery::Recovered {
                conversion,
                missing_mass_static_preconditioner,
                fields,
            }
        }
    };

    let mut envelope = ResponseSpectrumFieldRecovery {
        identity: ResponseSpectrumFieldRecoveryIdentity(Digest::sha256([])),
        source_result: result.identity(),
        response: result.response(),
        semantics: result.semantics(),
        nodal,
        deformation,
    };
    envelope.identity = envelope.derive_identity()?;
    envelope.validate()?;
    Ok(envelope)
}

pub fn response_spectrum_deformation_recovery_unavailability(
    result: &ModalResponseSpectrumResult,
) -> Option<ResponseSpectrumDeformationRecoveryUnavailability> {
    if has_missing_mass_correction(result)
        && result.response() != SpectrumResponse::PseudoAcceleration
    {
        return Some(
            ResponseSpectrumDeformationRecoveryUnavailability::MissingMassStaticCorrectionRequiresPseudoAcceleration {
                response: result.response(),
            },
        );
    }
    result.displacement_conversion().is_none().then_some(
        ResponseSpectrumDeformationRecoveryUnavailability::ResponseQuantityNotDisplacementConvertible {
            response: result.response(),
        },
    )
}

fn has_missing_mass_correction(result: &ModalResponseSpectrumResult) -> bool {
    result.contributions().iter().any(|contribution| {
        matches!(
            contribution.truncation.correction,
            ModalTruncationCorrection::MissingMass { .. }
        )
    })
}
