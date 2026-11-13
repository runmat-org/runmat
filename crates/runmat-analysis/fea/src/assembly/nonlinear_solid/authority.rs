//! The original physical basis identity schema, shared with storage verification.
use super::*;

type ElementGeometryAuthority<'a> = (StableDigest, &'a [[usize; 3]], &'a [PreparedPoint]);

#[derive(Serialize)]
pub(super) struct BasisAuthority<'a> {
    pub integration_identity: Digest,
    pub tangent_pattern_identity: Digest,
    pub properties: &'a CompiledPropertyMapIdentity,
    pub layout: &'a super::super::dofs::StructuralDofLayout,
    pub elements: Vec<ElementGeometryAuthority<'a>>,
}
