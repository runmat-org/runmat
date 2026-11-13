use runmat_canonical_codec::CanonicalLimits;

use super::FacetedSolid;
use crate::{FacetedSolidModel, GeometryContractError};

const FACETED_SOLID_LIMITS: CanonicalLimits =
    CanonicalLimits::new(512 * 1024 * 1024, 20_000_000, 8 * 1024 * 1024, 64);

pub fn encode_faceted_solid(
    solid: &FacetedSolid,
    model: &FacetedSolidModel,
) -> Result<Vec<u8>, GeometryContractError> {
    solid.validate_against(model)?;
    crate::model::canonical::encode(
        "analysis.geometry.faceted-solid/v2",
        solid,
        FACETED_SOLID_LIMITS,
    )
}

pub fn decode_faceted_solid(
    bytes: &[u8],
    model: &FacetedSolidModel,
) -> Result<FacetedSolid, GeometryContractError> {
    let solid = crate::model::canonical::decode(
        "analysis.geometry.faceted-solid/v2",
        bytes,
        FACETED_SOLID_LIMITS,
    )?;
    FacetedSolid::validate_against(&solid, model)?;
    Ok(solid)
}

pub fn decode_faceted_solid_with_control(
    bytes: &[u8],
    model: &FacetedSolidModel,
    control: &dyn crate::GeometryCanonicalControl,
) -> Result<FacetedSolid, GeometryContractError> {
    let solid: FacetedSolid = crate::model::canonical::decode_with_control(
        "analysis.geometry.faceted-solid/v2",
        bytes,
        FACETED_SOLID_LIMITS,
        control,
    )?;
    solid.validate_against_with_control(model, control)?;
    Ok(solid)
}
