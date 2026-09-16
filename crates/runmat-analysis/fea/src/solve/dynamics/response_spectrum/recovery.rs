use std::collections::BTreeMap;
use std::f64::consts::TAU;

use runmat_analysis_core::{
    AnalysisField, AnalysisFieldValues, ModalTruncationCorrection, SpectrumResponse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    contracts::FeaRunError,
    pipeline::modal_basis::ModalBasisArtifact,
    post::fields::{recover_deformation_result_fields, StructuralLinearFieldQuantity},
};

use super::{
    combine::{
        combine_directional_responses, combine_modal_responses,
        combine_periodic_and_rigid_responses,
    },
    validation::vectors_close,
    ModalResponseSpectrumContribution, ModalResponseSpectrumModeContribution,
    ModalResponseSpectrumModeKind, ModalResponseSpectrumResult,
    ModalResponseSpectrumResultIdentity, ResponseSpectrumDisplacementConversion,
    ResponseSpectrumResultSemantics,
};

const FIELD_RECOVERY_PREFIX: &[u8] = b"runmat-modal-response-spectrum-field-recovery\0";
const FIELD_RECOVERY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(1024 * 1024 * 1024, 40_000_000, 65_536, 24);

mod nodal;

pub use nodal::{
    ResponseSpectrumNodalEnvelope, ResponseSpectrumNodalField, ResponseSpectrumNodalQuantity,
};

use nodal::recover_nodal_envelope;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseSpectrumDeformationRecoveryUnavailability {
    ResponseQuantityNotDisplacementConvertible { response: SpectrumResponse },
    MissingMassStaticCorrectionNotQualified,
}

impl std::fmt::Display for ResponseSpectrumDeformationRecoveryUnavailability {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ResponseQuantityNotDisplacementConvertible { response } => write!(
                formatter,
                "{response:?} response cannot be converted into a signed displacement contribution for linear field recovery"
            ),
            Self::MissingMassStaticCorrectionNotQualified => formatter.write_str(
                "missing-mass response requires a separately qualified static-correction field recovery",
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseSpectrumDeformationRecovery {
    Recovered {
        conversion: ResponseSpectrumDisplacementConversion,
        fields: Vec<ResponseSpectrumLinearField>,
    },
    Unavailable {
        reason: ResponseSpectrumDeformationRecoveryUnavailability,
    },
}

impl ResponseSpectrumDeformationRecovery {
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
            ResponseSpectrumDeformationRecovery::Recovered { conversion, fields } => {
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
                ResponseSpectrumDeformationRecoveryUnavailability::MissingMassStaticCorrectionNotQualified
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
            ResponseSpectrumDeformationRecovery::Recovered {
                conversion,
                fields: recover_deformation_fields(summary, basis, result, conversion)?,
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

fn recover_deformation_fields(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    result: &ModalResponseSpectrumResult,
    conversion: ResponseSpectrumDisplacementConversion,
) -> Result<Vec<ResponseSpectrumLinearField>, FeaRunError> {
    let directional = result
        .contributions()
        .iter()
        .map(|contribution| recover_direction(summary, basis, contribution, conversion))
        .collect::<Result<Vec<_>, _>>()?;
    let first = directional.first().ok_or_else(|| {
        FeaRunError::Solve("response-spectrum field recovery has no directions".into())
    })?;
    let mut fields = Vec::with_capacity(first.len());
    for quantity in StructuralLinearFieldQuantity::ALL {
        let Some(reference) = first.get(&quantity) else {
            continue;
        };
        let entries = directional
            .iter()
            .map(|direction| {
                direction.get(&quantity).ok_or_else(|| {
                    FeaRunError::Solve(
                        "response-spectrum directions produced different linear field inventories"
                            .into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if entries.iter().any(|field| field.shape != reference.shape) {
            return Err(FeaRunError::Solve(
                "response-spectrum directions produced incompatible linear field shapes".into(),
            ));
        }
        let values = combine_directional_responses(
            &entries
                .iter()
                .map(|field| field.values.as_slice())
                .collect::<Vec<_>>(),
            result.directional_combination(),
        )?;
        fields.push(ResponseSpectrumLinearField {
            quantity,
            shape: reference.shape.clone(),
            values,
        });
    }
    if directional
        .iter()
        .any(|direction| direction.len() != fields.len())
    {
        return Err(FeaRunError::Solve(
            "response-spectrum directions produced different linear field inventories".into(),
        ));
    }
    Ok(fields)
}

pub fn response_spectrum_deformation_recovery_unavailability(
    result: &ModalResponseSpectrumResult,
) -> Option<ResponseSpectrumDeformationRecoveryUnavailability> {
    if result.contributions().iter().any(|contribution| {
        matches!(
            contribution.truncation.correction,
            ModalTruncationCorrection::MissingMass { .. }
        )
    }) {
        return Some(
            ResponseSpectrumDeformationRecoveryUnavailability::MissingMassStaticCorrectionNotQualified,
        );
    }
    result.displacement_conversion().is_none().then_some(
        ResponseSpectrumDeformationRecoveryUnavailability::ResponseQuantityNotDisplacementConvertible {
            response: result.response(),
        },
    )
}

fn recover_direction(
    summary: &AssemblySummary,
    basis: &ModalBasisArtifact,
    contribution: &ModalResponseSpectrumContribution,
    conversion: ResponseSpectrumDisplacementConversion,
) -> Result<BTreeMap<StructuralLinearFieldQuantity, ResponseSpectrumLinearField>, FeaRunError> {
    let mode_fields = contribution
        .modes
        .iter()
        .map(|mode| {
            let shape = mode_shape(basis, contribution, mode)?;
            let scale = displacement_scale(mode, conversion)?;
            let displacement = shape.iter().map(|value| scale * value).collect::<Vec<_>>();
            recover_linear_fields(summary, &displacement)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = mode_fields.first().ok_or_else(|| {
        FeaRunError::Solve("response-spectrum direction selected no modal contributions".into())
    })?;
    let frequencies = contribution
        .modes
        .iter()
        .map(|mode| mode.frequency_hz)
        .collect::<Vec<_>>();
    let mut combined = BTreeMap::new();
    for quantity in StructuralLinearFieldQuantity::ALL {
        let Some(reference) = first.get(&quantity) else {
            continue;
        };
        let entries = mode_fields
            .iter()
            .map(|fields| {
                fields.get(&quantity).ok_or_else(|| {
                    FeaRunError::Solve(
                        "modal contributions produced different linear field inventories".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if entries.iter().any(|field| field.shape != reference.shape) {
            return Err(FeaRunError::Solve(
                "modal contributions produced incompatible linear field shapes".into(),
            ));
        }
        let vectors = entries
            .iter()
            .map(|field| field.values.clone())
            .collect::<Vec<_>>();
        let (periodic, rigid) = if contribution.rigid_response_policy
            == runmat_analysis_core::RigidResponsePolicy::Disabled
        {
            (
                combine_modal_responses(
                    &vectors,
                    &frequencies,
                    contribution.damping_ratio,
                    contribution.modal_combination,
                )?,
                vec![0.0; reference.values.len()],
            )
        } else {
            combine_periodic_and_rigid_responses(
                &vectors,
                &frequencies,
                &contribution.rigid_coefficients,
                contribution.damping_ratio,
                contribution.modal_combination,
            )?
        };
        let values = periodic
            .iter()
            .zip(rigid)
            .map(|(periodic, rigid)| (periodic * periodic + rigid * rigid).sqrt())
            .collect();
        combined.insert(
            quantity,
            ResponseSpectrumLinearField {
                quantity,
                shape: reference.shape.clone(),
                values,
            },
        );
    }
    if mode_fields
        .iter()
        .any(|fields| fields.len() != combined.len())
    {
        return Err(FeaRunError::Solve(
            "modal contributions produced different linear field inventories".into(),
        ));
    }
    Ok(combined)
}

fn displacement_scale(
    mode: &ModalResponseSpectrumModeContribution,
    conversion: ResponseSpectrumDisplacementConversion,
) -> Result<f64, FeaRunError> {
    let angular_frequency = TAU * mode.frequency_hz;
    let divisor = match conversion {
        ResponseSpectrumDisplacementConversion::Direct => 1.0,
        ResponseSpectrumDisplacementConversion::DivideByAngularFrequency => angular_frequency,
        ResponseSpectrumDisplacementConversion::DivideByAngularFrequencySquared => {
            angular_frequency * angular_frequency
        }
    };
    let scale = mode.generalized_peak / divisor;
    if !scale.is_finite() {
        return Err(FeaRunError::Solve(
            "response-spectrum modal displacement conversion is nonfinite".into(),
        ));
    }
    Ok(scale)
}

fn mode_shape<'a>(
    basis: &'a ModalBasisArtifact,
    contribution: &ModalResponseSpectrumContribution,
    mode: &ModalResponseSpectrumModeContribution,
) -> Result<&'a [f64], FeaRunError> {
    match mode.kind {
        ModalResponseSpectrumModeKind::Eigenmode => basis
            .modes()
            .get(mode.mode_index.checked_sub(1).ok_or_else(|| {
                FeaRunError::Solve("response-spectrum eigenmode index is zero".into())
            })?)
            .map(|basis_mode| basis_mode.dof_shape())
            .ok_or_else(|| FeaRunError::Solve("response-spectrum eigenmode is absent".into())),
        ModalResponseSpectrumModeKind::ResidualVector => basis
            .residual_vectors()
            .iter()
            .find(|residual| vectors_close(&residual.direction(), &contribution.direction))
            .map(|residual| residual.dof_shape())
            .ok_or_else(|| {
                FeaRunError::Solve(
                    "response-spectrum residual contribution has no matching basis vector".into(),
                )
            }),
    }
}

fn recover_linear_fields(
    summary: &AssemblySummary,
    displacement: &[f64],
) -> Result<BTreeMap<StructuralLinearFieldQuantity, ResponseSpectrumLinearField>, FeaRunError> {
    let recovered = recover_deformation_result_fields(summary, displacement)
        .map_err(|error| FeaRunError::Assembly(error.to_string()))?;
    let mut fields = BTreeMap::new();
    for quantity in StructuralLinearFieldQuantity::ALL {
        let Some(field) = recovered
            .iter()
            .find(|field| field.field_id == quantity.source_field_id() && !field.is_empty())
        else {
            continue;
        };
        let values = host_values(field)?;
        fields.insert(
            quantity,
            ResponseSpectrumLinearField {
                quantity,
                shape: field.shape.clone(),
                values,
            },
        );
    }
    Ok(fields)
}

fn host_values(field: &AnalysisField) -> Result<Vec<f64>, FeaRunError> {
    match &field.values {
        AnalysisFieldValues::HostF64(values) => {
            if values.len() != field.element_count()
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(FeaRunError::Solve(
                    "linear field recovery returned malformed host storage".into(),
                ));
            }
            Ok(values.clone())
        }
        AnalysisFieldValues::DeviceRef(_) => Err(FeaRunError::Solve(
            "response-spectrum linear recovery requires materialized field values".into(),
        )),
    }
}
