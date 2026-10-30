//! Composition of the original authority's actual Serde schemas and exact inventories.
//! Inventory values are for profiling only; they never enter geometry or physics.
use super::*;
use crate::assembly::dofs::{StructuralDofAddress, StructuralDofLayout, StructuralNodeDofSet};
use runmat_canonical_codec::{
    profile_canonical_encoding, CanonicalLimits, CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT,
};

/// The authority is encoded directly (`encode_contract`), never projected into a value tree.
/// Its construction workspace is the direct encoder's own profile: the reserved output, held
/// out-of-order buffers and sorted-map metadata. Each component charges its share of that
/// workspace; the enclosing contract charges the shared envelope once.
fn encoding_workspace<T: Serialize>(value: &T) -> Result<u64, NonlinearSolidError> {
    Ok(profile_canonical_encoding(
        value,
        crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::new(0, 0, 4096, 64)),
    )
    .map_err(|e| invalid(&e.to_string()))?
    .resident_workspace_bytes_estimate)
}

fn incremental<T: Serialize>(value: &T) -> Result<u64, NonlinearSolidError> {
    encoding_workspace(value)?
        .checked_sub(CANONICAL_MATERIALIZATION_FIXED_BYTES_PER_CONTRACT)
        .ok_or_else(overflow)
}

pub(super) fn construction_bytes(
    plan: &CanonicalSolidIntegrationPlan,
    properties: &CompiledPropertyMapIdentity,
    dofs: usize,
) -> Result<u64, NonlinearSolidError> {
    let empty_layout = StructuralDofLayout::default();
    let enclosing = super::super::authority::BasisAuthority {
        integration_identity: Digest::sha256([]),
        tangent_pattern_identity: Digest::sha256([]),
        properties,
        layout: &empty_layout,
        elements: Vec::new(),
    };
    let outer = encoding_workspace(&enclosing)?;
    // The admitted uncoupled solid basis has three translations per node.
    // Profile the actual row/node schema without allocating a layout inventory.
    let node = incremental(&StructuralNodeDofSet::translational())?;
    let row = incremental(&StructuralDofAddress {
        node_index: usize::MAX,
        kind: StructuralDofKind::Ux,
    })?;
    let mut total = sum(&[
        outer,
        mul(node, dofs.div_ceil(3) as u64)?,
        mul(row, dofs as u64)?,
    ])?;
    let mut profiles = std::collections::BTreeMap::new();
    for element in plan.elements() {
        let (tuple_bytes, point_bytes) = if let Some(value) = profiles.get(&element.node_count) {
            *value
        } else {
            if !matches!(element.node_count, 4 | 10) {
                return Err(invalid("unsupported nonlinear authority node inventory"));
            }
            let identity = StableDigest([255; 32]);
            let rows = vec![[usize::MAX; 3]; element.node_count];
            let tuple = (identity, rows.as_slice(), &[] as &[PreparedPoint]);
            let point = PreparedPoint {
                identity: NonlinearSolidPointIdentity {
                    element_identity: identity,
                    quadrature_ordinal: usize::MAX,
                },
                position_m: [f64::MAX; 3],
                weight_m3: f64::MAX,
                gradients: vec![[f64::MAX; 3]; element.node_count],
            };
            let value = (incremental(&tuple)?, incremental(&point)?);
            profiles.insert(element.node_count, value);
            value
        };
        total = sum(&[
            total,
            tuple_bytes,
            mul(point_bytes, element.point_count as u64)?,
        ])?;
    }
    Ok(total)
}
