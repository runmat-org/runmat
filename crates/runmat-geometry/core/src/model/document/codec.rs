use runmat_canonical_codec::CanonicalLimits;

use super::GeometryDocument;
use crate::GeometryContractError;

const DOCUMENT_LIMITS: CanonicalLimits =
    CanonicalLimits::new(4 * 1024 * 1024, 100_000, 1 << 20, 64);

pub fn encode_geometry_document(
    document: &GeometryDocument,
) -> Result<Vec<u8>, GeometryContractError> {
    document.validate()?;
    crate::model::canonical::encode("analysis.geometry.document/v2", document, DOCUMENT_LIMITS)
}

pub fn decode_geometry_document(bytes: &[u8]) -> Result<GeometryDocument, GeometryContractError> {
    let document =
        crate::model::canonical::decode("analysis.geometry.document/v2", bytes, DOCUMENT_LIMITS)?;
    GeometryDocument::validate(&document)?;
    Ok(document)
}
