use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralDofKind {
    Ux,
    Uy,
    Uz,
    Rx,
    Ry,
    Rz,
}

impl StructuralDofKind {
    pub const ORDER: [StructuralDofKind; 6] = [
        StructuralDofKind::Ux,
        StructuralDofKind::Uy,
        StructuralDofKind::Uz,
        StructuralDofKind::Rx,
        StructuralDofKind::Ry,
        StructuralDofKind::Rz,
    ];

    pub fn is_translational(self) -> bool {
        matches!(
            self,
            StructuralDofKind::Ux | StructuralDofKind::Uy | StructuralDofKind::Uz
        )
    }

    pub fn is_rotational(self) -> bool {
        matches!(
            self,
            StructuralDofKind::Rx | StructuralDofKind::Ry | StructuralDofKind::Rz
        )
    }

    /// State and work-conjugate quantity kinds are different for translational
    /// and rotational rows. Equal numeric storage does not erase this distinction.
    pub const fn state_quantity(self) -> runmat_types::physical::QuantityKind {
        match self {
            Self::Ux | Self::Uy | Self::Uz => runmat_types::physical::QuantityKind::Displacement,
            Self::Rx | Self::Ry | Self::Rz => runmat_types::physical::QuantityKind::Rotation,
        }
    }

    pub const fn conjugate_quantity(self) -> runmat_types::physical::QuantityKind {
        match self {
            Self::Ux | Self::Uy | Self::Uz => runmat_types::physical::QuantityKind::Force,
            Self::Rx | Self::Ry | Self::Rz => runmat_types::physical::QuantityKind::Moment,
        }
    }

    pub(crate) const fn from_constraint(dof: crate::constraints::CanonicalStructuralDof) -> Self {
        use crate::constraints::CanonicalStructuralDof;
        use runmat_analysis_core::{StructuralRotationDof, StructuralTranslationDof};

        match dof {
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Ux) => Self::Ux,
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Uy) => Self::Uy,
            CanonicalStructuralDof::Translation(StructuralTranslationDof::Uz) => Self::Uz,
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rx) => Self::Rx,
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Ry) => Self::Ry,
            CanonicalStructuralDof::Rotation(StructuralRotationDof::Rz) => Self::Rz,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StructuralNodeDofSet {
    active: Vec<StructuralDofKind>,
}

impl<'de> Deserialize<'de> for StructuralNodeDofSet {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct EncodedSet {
            active: Vec<StructuralDofKind>,
        }
        let encoded = EncodedSet::deserialize(deserializer)?;
        if !StructuralDofKind::ORDER
            .into_iter()
            .filter(|kind| encoded.active.contains(kind))
            .eq(encoded.active.iter().copied())
        {
            return Err(serde::de::Error::custom(
                "structural DOF sets must be ordered and unique",
            ));
        }
        Ok(Self {
            active: encoded.active,
        })
    }
}

impl StructuralNodeDofSet {
    pub fn translational() -> Self {
        Self::from_kinds([
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ])
    }

    pub fn translational_rotational() -> Self {
        Self::from_kinds(StructuralDofKind::ORDER)
    }

    pub fn from_kinds<const N: usize>(kinds: [StructuralDofKind; N]) -> Self {
        let mut active = Vec::with_capacity(N);
        for kind in StructuralDofKind::ORDER {
            if kinds.contains(&kind) {
                active.push(kind);
            }
        }
        Self { active }
    }

    pub fn contains(&self, kind: StructuralDofKind) -> bool {
        self.active.contains(&kind)
    }

    pub fn active(&self) -> &[StructuralDofKind] {
        &self.active
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StructuralDofAddress {
    pub node_index: usize,
    pub kind: StructuralDofKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StructuralDofLayout {
    node_sets: Vec<StructuralNodeDofSet>,
    rows: Vec<StructuralDofAddress>,
    reference_point_count: usize,
}

impl<'de> Deserialize<'de> for StructuralDofLayout {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct EncodedLayout {
            node_sets: Vec<StructuralNodeDofSet>,
            rows: Vec<StructuralDofAddress>,
            reference_point_count: usize,
        }
        let encoded = EncodedLayout::deserialize(deserializer)?;
        let layout = Self {
            node_sets: encoded.node_sets,
            rows: encoded.rows,
            reference_point_count: encoded.reference_point_count,
        };
        layout.validate().map_err(serde::de::Error::custom)?;
        Ok(layout)
    }
}

impl Default for StructuralDofLayout {
    fn default() -> Self {
        Self::from_node_sets(Vec::new())
    }
}

impl StructuralDofLayout {
    pub fn validate(&self) -> Result<(), &'static str> {
        let mut row = 0;
        for (node_index, set) in self.node_sets.iter().enumerate() {
            if !StructuralDofKind::ORDER
                .into_iter()
                .filter(|kind| set.contains(*kind))
                .eq(set.active.iter().copied())
            {
                return Err("structural DOF sets must be ordered and unique");
            }
            for kind in &set.active {
                if self.rows.get(row)
                    != Some(&StructuralDofAddress {
                        node_index,
                        kind: *kind,
                    })
                {
                    return Err("structural DOF rows disagree with node sets");
                }
                row += 1;
            }
        }
        if row != self.rows.len()
            || self
                .reference_point_count
                .checked_mul(6)
                .and_then(|count| count.checked_add(row))
                .is_none()
        {
            return Err("structural DOF rows or reference-point layout are inconsistent");
        }
        Ok(())
    }
    pub fn from_node_sets(node_sets: Vec<StructuralNodeDofSet>) -> Self {
        let rows = node_sets
            .iter()
            .enumerate()
            .flat_map(|(node_index, set)| {
                set.active()
                    .iter()
                    .copied()
                    .map(move |kind| StructuralDofAddress { node_index, kind })
            })
            .collect();
        Self {
            node_sets,
            rows,
            reference_point_count: 0,
        }
    }

    pub fn with_reference_points(mut self, reference_point_count: usize) -> Self {
        self.reference_point_count = reference_point_count;
        self
    }

    pub fn reduced_translational_rows(dof_count: usize) -> Self {
        if dof_count == 0 {
            return Self::from_node_sets(Vec::new());
        }
        let node_count = dof_count.div_ceil(3);
        let mut kinds_by_node = vec![Vec::new(); node_count];
        for row in 0..dof_count {
            let kind = match row % 3 {
                0 => StructuralDofKind::Ux,
                1 => StructuralDofKind::Uy,
                _ => StructuralDofKind::Uz,
            };
            kinds_by_node[row / 3].push(kind);
        }
        let node_sets = kinds_by_node
            .into_iter()
            .map(|kinds| {
                let mut active = Vec::new();
                for kind in StructuralDofKind::ORDER {
                    if kinds.contains(&kind) {
                        active.push(kind);
                    }
                }
                StructuralNodeDofSet { active }
            })
            .collect();
        Self::from_node_sets(node_sets)
    }

    pub fn total_dof_count(&self) -> usize {
        self.rows.len() + self.reference_point_count * StructuralDofKind::ORDER.len()
    }

    pub fn node_dof_count(&self) -> usize {
        self.rows.len()
    }

    pub fn node_count(&self) -> usize {
        self.node_sets.len()
    }

    pub fn rotation_node_count(&self) -> usize {
        self.node_sets
            .iter()
            .filter(|set| set.active().iter().any(|kind| kind.is_rotational()))
            .count()
            + self.reference_point_count
    }

    pub fn translational_dof_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|address| address.kind.is_translational())
            .count()
            + self.reference_point_count * 3
    }

    pub fn rotational_dof_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|address| address.kind.is_rotational())
            .count()
            + self.reference_point_count * 3
    }

    pub fn index(&self, node_index: usize, kind: StructuralDofKind) -> Option<usize> {
        // Construction and validate() establish canonical node/component
        // ordering. Reuse that ordering rather than scanning all model rows
        // at every boundary quadrature point and transient sample.
        self.rows
            .binary_search(&StructuralDofAddress { node_index, kind })
            .ok()
    }

    pub fn reference_point_index(
        &self,
        reference_point_index: usize,
        kind: StructuralDofKind,
    ) -> Option<usize> {
        if reference_point_index >= self.reference_point_count {
            return None;
        }
        StructuralDofKind::ORDER
            .iter()
            .position(|candidate| *candidate == kind)
            .map(|component| {
                self.rows.len() + reference_point_index * StructuralDofKind::ORDER.len() + component
            })
    }

    pub fn reference_point_count(&self) -> usize {
        self.reference_point_count
    }

    pub fn address(&self, row: usize) -> Option<StructuralDofAddress> {
        self.rows.get(row).copied()
    }

    /// Includes appended reference-point rows; those rows share the same six-DOF ordering.
    pub fn row_kind(&self, row: usize) -> Option<StructuralDofKind> {
        if let Some(address) = self.rows.get(row) {
            return Some(address.kind);
        }
        (row < self.total_dof_count())
            .then(|| StructuralDofKind::ORDER[(row - self.rows.len()) % 6])
    }

    /// A mixed structural matrix has dimensions per row/column pair, never one
    /// scalar matrix dimension. Derivative order 0/1/2 selects stiffness/damping/mass.
    pub fn operator_entry_dimension(
        &self,
        row: usize,
        column: usize,
        derivative_order: u8,
    ) -> Option<runmat_types::PhysicalDimension> {
        if derivative_order > 2 {
            return None;
        }
        Some(
            self.operator_entry_descriptor(row, column, derivative_order)?
                .dimension(),
        )
    }

    /// Full coefficient descriptor, including inverse radians for rotational
    /// state columns. This is output/input times the derivative-time role.
    pub fn operator_entry_descriptor(
        &self,
        row: usize,
        column: usize,
        derivative_order: u8,
    ) -> Option<runmat_types::physical::DynamicDimension> {
        use runmat_types::physical::DynamicDimension;
        if derivative_order > 2 {
            return None;
        }
        let load = DynamicDimension::quantity_type(self.row_kind(row)?.conjugate_quantity());
        let state = DynamicDimension::quantity_type(self.row_kind(column)?.state_quantity());
        load.divide(state)?.multiply(
            DynamicDimension::from(runmat_types::PhysicalDimension::TIME)
                .power(derivative_order as i8)?,
        )
    }

    pub fn state_component(
        &self,
        row: usize,
        value: f64,
    ) -> Option<runmat_types::physical::AnyQuantity> {
        runmat_types::physical::AnyQuantity::try_new(self.row_kind(row)?.state_quantity(), value)
            .ok()
    }

    pub fn load_component(
        &self,
        row: usize,
        value: f64,
    ) -> Option<runmat_types::physical::AnyQuantity> {
        runmat_types::physical::AnyQuantity::try_new(
            self.row_kind(row)?.conjugate_quantity(),
            value,
        )
        .ok()
    }

    pub fn node_set(&self, node_index: usize) -> Option<&StructuralNodeDofSet> {
        self.node_sets.get(node_index)
    }

    pub fn has_rotational_dofs(&self) -> bool {
        self.rotational_dof_count() > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_dof_layout_indexes_full_rotational_nodes_deterministically() {
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational_rotational(),
            StructuralNodeDofSet::translational(),
        ]);

        assert_eq!(layout.total_dof_count(), 9);
        assert_eq!(layout.node_count(), 2);
        assert_eq!(layout.translational_dof_count(), 6);
        assert_eq!(layout.rotational_dof_count(), 3);
        assert_eq!(layout.rotation_node_count(), 1);
        assert_eq!(layout.index(0, StructuralDofKind::Ux), Some(0));
        assert_eq!(layout.index(0, StructuralDofKind::Rz), Some(5));
        assert_eq!(layout.index(1, StructuralDofKind::Ux), Some(6));
        assert_eq!(layout.index(1, StructuralDofKind::Rz), None);
        assert_eq!(layout.index(usize::MAX, StructuralDofKind::Ux), None);
        let encoded = serde_json::to_value(&layout).unwrap();
        assert_eq!(
            serde_json::from_value::<StructuralDofLayout>(encoded.clone()).unwrap(),
            layout
        );
        let mut reordered = encoded;
        reordered["rows"].as_array_mut().unwrap().reverse();
        assert!(serde_json::from_value::<StructuralDofLayout>(reordered).is_err());
        assert!(serde_json::from_value::<StructuralNodeDofSet>(
            serde_json::json!({"active": ["uz", "ux"]})
        )
        .is_err());
        assert!(StructuralDofKind::ORDER
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
        for node_index in 0..=layout.node_count() {
            for kind in StructuralDofKind::ORDER {
                assert_eq!(
                    layout.index(node_index, kind),
                    layout
                        .rows
                        .iter()
                        .position(|row| row.node_index == node_index && row.kind == kind)
                );
            }
        }
        assert_eq!(
            layout.address(4),
            Some(StructuralDofAddress {
                node_index: 0,
                kind: StructuralDofKind::Ry,
            })
        );
    }

    #[test]
    fn mixed_operator_dimensions_include_reference_points() {
        use runmat_types::{physical::QuantityKind, PhysicalDimension as D};
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational_rotational(),
        ])
        .with_reference_points(1);
        assert_eq!(layout.row_kind(6), Some(StructuralDofKind::Ux));
        assert_eq!(layout.row_kind(11), Some(StructuralDofKind::Rz));
        assert_eq!(layout.row_kind(12), None);
        assert_eq!(
            layout.state_component(11, 0.5).unwrap().kind(),
            QuantityKind::Rotation
        );
        assert_eq!(
            layout.load_component(11, 2.0).unwrap().kind(),
            QuantityKind::Moment
        );
        assert_eq!(
            layout.operator_entry_dimension(0, 0, 0),
            D::FORCE.divide(D::LENGTH)
        );
        assert_eq!(layout.operator_entry_dimension(3, 3, 0), Some(D::MOMENT));
        assert_eq!(layout.operator_entry_dimension(0, 0, 2), Some(D::MASS));
        assert_ne!(
            layout.operator_entry_dimension(0, 3, 0),
            layout.operator_entry_dimension(3, 3, 0)
        );
    }

    #[test]
    fn reduced_translational_layout_preserves_requested_row_count() {
        let layout = StructuralDofLayout::reduced_translational_rows(5);

        assert_eq!(layout.total_dof_count(), 5);
        assert_eq!(layout.node_count(), 2);
        assert_eq!(layout.translational_dof_count(), 5);
        assert_eq!(layout.rotational_dof_count(), 0);
        assert_eq!(layout.index(1, StructuralDofKind::Ux), Some(3));
        assert_eq!(layout.index(1, StructuralDofKind::Uz), None);
    }
}

/// Layout views distinguish state derivatives from their work-conjugate loads.
#[derive(Debug, Clone, Copy)]
pub struct StructuralQuantityLayout<'a> {
    pub dofs: &'a StructuralDofLayout,
    pub lane: StructuralQuantityLane,
}
impl<'a, 'b> PartialEq<StructuralQuantityLayout<'b>> for StructuralQuantityLayout<'a> {
    fn eq(&self, other: &StructuralQuantityLayout<'b>) -> bool {
        self.dofs == other.dofs && self.lane == other.lane
    }
}
impl Eq for StructuralQuantityLayout<'_> {}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralQuantityLane {
    State { derivative_order: u8 },
    Load,
}
impl runmat_types::physical::QuantityLayout for StructuralQuantityLayout<'_> {
    fn component_count(&self) -> usize {
        self.dofs.total_dof_count()
    }
    fn component_descriptor(
        &self,
        index: usize,
    ) -> Option<runmat_types::physical::DynamicDimension> {
        use runmat_types::physical::DynamicDimension;
        let row = self.dofs.row_kind(index)?;
        match self.lane {
            StructuralQuantityLane::Load => Some(
                DynamicDimension::quantity_type(row.conjugate_quantity())
                    .with_semantic_kind(Some(row.conjugate_quantity()))
                    .ok()?,
            ),
            StructuralQuantityLane::State { derivative_order } => {
                if derivative_order > 2 {
                    return None;
                }
                let state = DynamicDimension::quantity_type(row.state_quantity());
                if derivative_order == 0 {
                    state.with_semantic_kind(Some(row.state_quantity())).ok()
                } else {
                    state.divide(
                        DynamicDimension::from(runmat_types::PhysicalDimension::TIME)
                            .power(derivative_order as i8)?,
                    )
                }
            }
        }
    }
}
impl StructuralDofLayout {
    pub fn physical_field<'a>(
        &'a self,
        lane: StructuralQuantityLane,
        values: &'a [f64],
    ) -> Result<
        runmat_types::physical::QuantityField<StructuralQuantityLayout<'a>, &'a [f64]>,
        runmat_types::physical::QuantityFieldError,
    > {
        runmat_types::physical::QuantityField::try_new(
            StructuralQuantityLayout { dofs: self, lane },
            values,
        )
    }
}
