use super::{decode_faceted_solid, encode_faceted_solid, EncodedFacetedSolidClosure, FacetedSolid};
use crate::{
    FacetedSolidModel, GeometryContractError, GeometryDocument, GeometryModel, GeometryObjectRef,
    FACETED_SOLID_MEDIA_TYPE, GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
};

pub fn build_faceted_solid_closure(
    mut document: GeometryDocument,
    solid: FacetedSolid,
) -> Result<EncodedFacetedSolidClosure, GeometryContractError> {
    let GeometryModel::FacetedSolid { model } = &mut document.model else {
        return Err(GeometryContractError::invalid(
            "faceted solid closure",
            "exact geometry cannot be packaged as a faceted solid",
        ));
    };
    model.vertex_count = solid.vertices.len() as u64;
    model.triangle_count = solid.triangles.len() as u64;
    model.shell_count = solid.shells.len() as u64;
    model.is_watertight = true;
    model.is_oriented = true;
    let solid_bytes = encode_faceted_solid(&solid, model)?;
    model.artifact = GeometryObjectRef {
        digest: crate::model::canonical::digest(&solid_bytes)?,
        encoded_length: solid_bytes.len() as u64,
        media_type: FACETED_SOLID_MEDIA_TYPE.into(),
        schema_version: GEOMETRY_PRIMARY_ARTIFACT_SCHEMA_VERSION,
    };
    document.validate()?;
    // encode_faceted_solid has validated the original typed solid. Its newly
    // constructed immutable bytes/reference need no second typed decode here.
    Ok(EncodedFacetedSolidClosure {
        document,
        solid,
        solid_bytes,
    })
}

pub fn admit_faceted_solid(
    document: &GeometryDocument,
    solid_bytes: &[u8],
) -> Result<FacetedSolid, GeometryContractError> {
    document.validate()?;
    let GeometryModel::FacetedSolid { model } = &document.model else {
        return Err(GeometryContractError::invalid(
            "faceted solid admission",
            "exact geometry has no faceted solid payload",
        ));
    };
    verify_reference(model, solid_bytes)?;
    decode_faceted_solid(solid_bytes, model)
}

fn verify_reference(model: &FacetedSolidModel, bytes: &[u8]) -> Result<(), GeometryContractError> {
    if model.artifact.encoded_length != bytes.len() as u64
        || model.artifact.digest != crate::model::canonical::digest(bytes)?
    {
        return Err(GeometryContractError::invalid(
            "faceted solid artifact",
            "payload length or digest does not match the geometry document",
        ));
    }
    Ok(())
}

pub fn admit_faceted_solid_with_control(
    document: &GeometryDocument,
    bytes: &[u8],
    control: &dyn crate::GeometryCanonicalControl,
) -> Result<FacetedSolid, GeometryContractError> {
    document.validate()?;
    let GeometryModel::FacetedSolid { model } = &document.model else {
        return Err(GeometryContractError::invalid(
            "faceted solid admission",
            "exact geometry has no faceted solid payload",
        ));
    };
    verify_reference(model, bytes)?;
    super::decode_faceted_solid_with_control(bytes, model, control)
}
