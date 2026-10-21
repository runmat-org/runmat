//! Composition of the original authority's actual Serde schemas and exact inventories.
//! Inventory values are for profiling only; they never enter geometry or physics.
use super::*;
use crate::assembly::dofs::{StructuralDofAddress, StructuralDofLayout, StructuralNodeDofSet};
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};

/// The authority is encoded directly (`encode_contract`), never projected into a value tree:
/// its construction holds the canonical wire bytes plus at most one growth reallocation.
fn incremental<T: Serialize>(value: &T) -> Result<u64, NonlinearSolidError> {
    profile_serialized(
        value,
        CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 4096, 64),
    )
    .map_err(|e| invalid(&e.to_string()))?
    .encoded_bytes_upper_bound
    .checked_mul(2)
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
    let outer = profile_serialized(
        &enclosing,
        CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 4096, 64),
    )
    .map_err(|e| invalid(&e.to_string()))?
    .encoded_bytes_upper_bound
    .checked_mul(2)
    .ok_or_else(overflow)?;
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
