//! Common optional control for surface contracts; wire forms and codec limits stay unchanged.
use super::{ExactSurfaceMeshError, ExactSurfaceMeshErrorKind};
use runmat_canonical_codec::{CanonicalCodecError, CanonicalLimits};
use runmat_meshing_core::{MeshingCancellationSignal, MeshingCanonicalCodecControl};
use serde::{de::DeserializeOwned, Serialize};

pub(super) fn encode<T: Serialize>(
    prefix: &[u8],
    domain: &str,
    value: &T,
    limits: CanonicalLimits,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<Vec<u8>, ExactSurfaceMeshError> {
    match control {
        Some(signal) => {
            let control = MeshingCanonicalCodecControl::new(signal);
            runmat_canonical_codec::encode_contract_with_control(
                prefix, domain, value, limits, &control,
            )
            .map_err(|error| map_error(error, control.observed_cancellation()))
        }
        None => runmat_canonical_codec::encode_contract(prefix, domain, value, limits)
            .map_err(|error| map_error(error, false)),
    }
}

pub(super) fn decode<T: DeserializeOwned>(
    prefix: &[u8],
    domain: &str,
    bytes: &[u8],
    limits: CanonicalLimits,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<T, ExactSurfaceMeshError> {
    match control {
        Some(signal) => {
            let control = MeshingCanonicalCodecControl::new(signal);
            runmat_canonical_codec::decode_contract_with_control(
                prefix, domain, bytes, limits, &control,
            )
            .map_err(|error| map_error(error, control.observed_cancellation()))
        }
        None => runmat_canonical_codec::decode_contract(prefix, domain, bytes, limits)
            .map_err(|error| map_error(error, false)),
    }
}

fn map_error(error: CanonicalCodecError, cancelled: bool) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(
        if cancelled {
            ExactSurfaceMeshErrorKind::Cancelled
        } else {
            ExactSurfaceMeshErrorKind::InvalidEncoding
        },
        format!("{}: {}", error.field, error.reason),
    )
}
