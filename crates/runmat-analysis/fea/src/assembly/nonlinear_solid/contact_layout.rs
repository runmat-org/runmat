//! Contact consumes exactly the same original nodal geometry and translation rows.
use super::*;
use crate::physics::structural::contact::CanonicalContactOperator;
impl CanonicalNonlinearSolidBasis {
    pub(crate) fn contact_pattern(&self) -> &SparsePattern {
        &self.tangent_pattern
    }
    pub(crate) fn validate_contact_layout(
        &self,
        contact: &CanonicalContactOperator<'_>,
    ) -> Result<(), NonlinearSolidError> {
        let projection = contact.projection();
        if projection.mesh_digest != self.mesh_digest
            || projection.nodes.len() != self.nodes.len()
            || contact.node_rows().len() != self.nodes.len()
            || contact.pattern().rows() != self.dof_count
        {
            return Err(NonlinearSolidError::Authority(
                "contact/body geometry or layout differs".into(),
            ));
        }
        for ((original, node), rows) in self
            .nodes
            .iter()
            .zip(&projection.nodes)
            .zip(contact.node_rows())
        {
            if original.identity != node.stable_identity
                || original.coordinates_m.canonical_value() != node.coordinates_m.canonical_value()
                || original.rows != *rows
            {
                return Err(NonlinearSolidError::Authority(
                    "contact/body original node authority differs".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn contact_positions(
        &self,
        u: &[f64],
    ) -> Result<Vec<[f64; 3]>, NonlinearSolidError> {
        if crate::is_fea_cancelled() {
            return Err(NonlinearSolidError::Cancelled);
        }
        if u.len() != self.dof_count || u.iter().any(|v| !v.is_finite()) {
            return Err(NonlinearSolidError::Authority(
                "invalid original body displacement".into(),
            ));
        }
        let mut positions = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            if crate::is_fea_cancelled() {
                return Err(NonlinearSolidError::Cancelled);
            }
            let p = std::array::from_fn(|axis| {
                node.coordinates_m.canonical_value()[axis] + u[node.rows[axis]]
            });
            if p.iter().any(|v| !v.is_finite()) {
                return Err(NonlinearSolidError::Authority(
                    "current body position overflow".into(),
                ));
            }
            positions.push(p);
        }
        Ok(positions)
    }
}

pub(super) fn prepare_nodes(
    projection: &AnalysisProjection,
    canonical_nodes: &BTreeMap<StableDigest, usize>,
    initial: &AssemblySummary,
) -> Result<Vec<PreparedBodyNode>, NonlinearSolidError> {
    projection
        .nodes
        .iter()
        .map(|n| {
            let index = canonical_nodes[&n.stable_identity];
            let mut rows = [0; 3];
            for (axis, kind) in [
                StructuralDofKind::Ux,
                StructuralDofKind::Uy,
                StructuralDofKind::Uz,
            ]
            .into_iter()
            .enumerate()
            {
                rows[axis] = initial
                    .structural_dof_layout
                    .index(index, kind)
                    .ok_or_else(|| {
                        NonlinearSolidError::Authority("missing original translation row".into())
                    })?;
            }
            Ok(PreparedBodyNode {
                identity: n.stable_identity,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(n.coordinates_m.canonical_value()),
                rows,
            })
        })
        .collect::<Result<Vec<_>, NonlinearSolidError>>()
}
