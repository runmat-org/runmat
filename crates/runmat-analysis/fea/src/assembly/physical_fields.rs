//! Checked full structural numerical seams. Reduced/modal operators must retain
//! their own basis and cannot be rebound to this full-DOF view by matching size.
use super::{
    dofs::{StructuralQuantityLane, StructuralQuantityLayout},
    AssemblySummary, StructuralDofLayout,
};
use crate::algebra::{CanonicalRealOperators, RealSparseMatrix};
use runmat_types::physical::{DynamicDimension, Quantity, QuantityField};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralOperatorRole {
    Stiffness,
    Damping,
    Mass,
}
impl StructuralOperatorRole {
    pub fn derivative_order(self) -> u8 {
        match self {
            Self::Stiffness => 0,
            Self::Damping => 1,
            Self::Mass => 2,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuralPhysicalFieldError {
    ShapeMismatch,
    CoordinateRoleMismatch,
    InvalidComponent,
}

pub struct StructuralPhysicalOperator<'a> {
    layout: &'a StructuralDofLayout,
    operators: &'a CanonicalRealOperators,
}
impl<'a> StructuralPhysicalOperator<'a> {
    /// The caller binds an admitted full structural assembly, never a projected
    /// matrix. Shape equality alone is not an authorization to change its basis.
    pub fn from_assembly(
        summary: &'a AssemblySummary,
    ) -> Result<Self, StructuralPhysicalFieldError> {
        summary
            .structural_dof_layout
            .validate()
            .map_err(|_| StructuralPhysicalFieldError::ShapeMismatch)?;
        if summary.operator.dof_count() != summary.structural_dof_layout.total_dof_count() {
            return Err(StructuralPhysicalFieldError::ShapeMismatch);
        }
        Ok(Self {
            layout: &summary.structural_dof_layout,
            operators: summary.operator.operators(),
        })
    }
    fn operator(
        &self,
        role: StructuralOperatorRole,
    ) -> Result<
        runmat_types::physical::QuantityLinearOperator<
            StructuralQuantityLayout<'a>,
            StructuralQuantityLayout<'a>,
            &'a RealSparseMatrix,
        >,
        StructuralPhysicalFieldError,
    > {
        let matrix = match role {
            StructuralOperatorRole::Stiffness => self.operators.stiffness(),
            StructuralOperatorRole::Damping => self.operators.damping(),
            StructuralOperatorRole::Mass => self.operators.mass(),
        };
        runmat_types::physical::QuantityLinearOperator::try_new(
            StructuralQuantityLayout {
                dofs: self.layout,
                lane: StructuralQuantityLane::State {
                    derivative_order: role.derivative_order(),
                },
            },
            StructuralQuantityLayout {
                dofs: self.layout,
                lane: StructuralQuantityLane::Load,
            },
            matrix,
        )
        .map_err(|_| StructuralPhysicalFieldError::ShapeMismatch)
    }
    pub fn coefficient(
        &self,
        role: StructuralOperatorRole,
        row: usize,
        column: usize,
    ) -> Option<Quantity<DynamicDimension>> {
        self.operator(role).ok()?.coefficient(row, column)
    }
    pub fn apply<S: AsRef<[f64]>>(
        &self,
        role: StructuralOperatorRole,
        state: &QuantityField<StructuralQuantityLayout<'_>, S>,
    ) -> Result<QuantityField<StructuralQuantityLayout<'a>, Vec<f64>>, StructuralPhysicalFieldError>
    {
        self.operator(role)?
            .apply(state)
            .map_err(|error| match error {
                runmat_types::physical::QuantityOperatorError::InputLayoutMismatch => {
                    StructuralPhysicalFieldError::CoordinateRoleMismatch
                }
                _ => StructuralPhysicalFieldError::InvalidComponent,
            })
    }
}
impl AssemblySummary {
    pub fn physical_load_field(
        &self,
    ) -> Result<QuantityField<StructuralQuantityLayout<'_>, &[f64]>, StructuralPhysicalFieldError>
    {
        self.structural_dof_layout
            .physical_field(StructuralQuantityLane::Load, &self.physical_load)
            .map_err(|_| StructuralPhysicalFieldError::InvalidComponent)
    }
    pub fn state_field<'a>(
        &'a self,
        derivative_order: u8,
        values: &'a [f64],
    ) -> Result<QuantityField<StructuralQuantityLayout<'a>, &'a [f64]>, StructuralPhysicalFieldError>
    {
        self.structural_dof_layout
            .physical_field(StructuralQuantityLane::State { derivative_order }, values)
            .map_err(|_| StructuralPhysicalFieldError::InvalidComponent)
    }
}

/// Actual coordinate basis of a structural solve. A reduced system retains its
/// transformation instead of claiming that its rows are the original DOFs.
#[derive(Clone)]
pub enum StructuralSolveLayout<'a> {
    Full(&'a StructuralDofLayout),
    Affine {
        full: &'a StructuralDofLayout,
        reduction: std::sync::Arc<crate::constraints::AffineConstraintReduction>,
    },
}
impl StructuralSolveLayout<'_> {
    pub fn validate(&self) -> Result<(), StructuralPhysicalFieldError> {
        let full = match self {
            Self::Full(full) | Self::Affine { full, .. } => full,
        };
        full.validate()
            .map_err(|_| StructuralPhysicalFieldError::ShapeMismatch)?;
        if let Self::Affine { full, reduction } = self {
            reduction
                .validate()
                .map_err(|_| StructuralPhysicalFieldError::ShapeMismatch)?;
            if reduction.full_dof_count != full.total_dof_count() {
                return Err(StructuralPhysicalFieldError::ShapeMismatch);
            }
        }
        Ok(())
    }
    pub fn row_kind(&self, index: usize) -> Option<super::dofs::StructuralDofKind> {
        match self {
            Self::Full(full) => full.row_kind(index),
            Self::Affine { full, reduction } => {
                full.row_kind(*reduction.free_dof_indices.get(index)?)
            }
        }
    }
    pub fn load_descriptor(&self, index: usize) -> Option<DynamicDimension> {
        let kind = self.row_kind(index)?.conjugate_quantity();
        DynamicDimension::quantity_type(kind)
            .with_semantic_kind(Some(kind))
            .ok()
    }
    pub fn transform_descriptor(&self, row: usize, column: usize) -> Option<DynamicDimension> {
        use runmat_types::physical::QuantityLayout;
        let Self::Affine { full, .. } = self else {
            return None;
        };
        DynamicDimension::quantity_type(full.row_kind(row)?.state_quantity())
            .divide(self.component_descriptor(column)?)
    }
}
impl runmat_types::physical::QuantityLayout for StructuralSolveLayout<'_> {
    fn component_count(&self) -> usize {
        match self {
            Self::Full(full) => full.total_dof_count(),
            Self::Affine { reduction, .. } => reduction.reduced_dof_count,
        }
    }
    fn component_descriptor(&self, index: usize) -> Option<DynamicDimension> {
        let kind = self.row_kind(index)?.state_quantity();
        DynamicDimension::quantity_type(kind)
            .with_semantic_kind(Some(kind))
            .ok()
    }
}

impl StructuralPhysicalOperator<'_> {
    pub fn elastic_energy<S: AsRef<[f64]>, T: AsRef<[f64]>>(
        &self,
        state: &QuantityField<StructuralQuantityLayout<'_>, S>,
        load: &QuantityField<StructuralQuantityLayout<'_>, T>,
    ) -> Result<runmat_types::physical::Energy, StructuralPhysicalFieldError> {
        use runmat_types::physical::{kind, Energy};
        if state.layout().dofs != self.layout
            || load.layout().dofs != self.layout
            || state.layout().lane
                != (StructuralQuantityLane::State {
                    derivative_order: 0,
                })
            || load.layout().lane != StructuralQuantityLane::Load
        {
            return Err(StructuralPhysicalFieldError::CoordinateRoleMismatch);
        }
        let mut energy = Energy::new(0.0);
        for row in 0..state.len() {
            let displacement = state
                .component(row)
                .ok_or(StructuralPhysicalFieldError::InvalidComponent)?;
            let force = load
                .component(row)
                .ok_or(StructuralPhysicalFieldError::InvalidComponent)?;
            let work = if self
                .layout
                .row_kind(row)
                .ok_or(StructuralPhysicalFieldError::InvalidComponent)?
                .is_translational()
            {
                force
                    .resolve_kind::<kind::Force>()
                    .map_err(|_| StructuralPhysicalFieldError::CoordinateRoleMismatch)?
                    .work_over(
                        displacement
                            .resolve_kind::<kind::Displacement>()
                            .map_err(|_| StructuralPhysicalFieldError::CoordinateRoleMismatch)?,
                    )
            } else {
                force
                    .resolve_kind::<kind::Moment>()
                    .map_err(|_| StructuralPhysicalFieldError::CoordinateRoleMismatch)?
                    .work_over(
                        displacement
                            .resolve_kind::<kind::Rotation>()
                            .map_err(|_| StructuralPhysicalFieldError::CoordinateRoleMismatch)?,
                    )
            };
            energy += work * 0.5;
        }
        if !energy.canonical_value().is_finite() {
            return Err(StructuralPhysicalFieldError::InvalidComponent);
        }
        Ok(energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algebra::{MatrixProperties, OperatorRole, RealSparseMatrix},
        assembly::dofs::StructuralNodeDofSet,
    };
    #[test]
    fn mixed_structural_work_contracts_translation_and_rotation_as_energy() {
        let layout = StructuralDofLayout::from_node_sets(vec![
            StructuralNodeDofSet::translational_rotational(),
        ]);
        let stiffness = RealSparseMatrix::diagonal(
            vec![1000.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            OperatorRole::Stiffness,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();
        let mass = RealSparseMatrix::diagonal(
            vec![0.0; 6],
            OperatorRole::Mass,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();
        let damping = RealSparseMatrix::diagonal(
            vec![0.0; 6],
            OperatorRole::Damping,
            MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
        )
        .unwrap();
        let operators = CanonicalRealOperators::try_new(stiffness, mass, damping).unwrap();
        let physical = StructuralPhysicalOperator {
            layout: &layout,
            operators: &operators,
        };
        let values = [0.01, 0.0, 0.0, 0.2, 0.0, 0.0];
        let state = layout
            .physical_field(
                StructuralQuantityLane::State {
                    derivative_order: 0,
                },
                &values,
            )
            .unwrap();
        let load = physical
            .apply(StructuralOperatorRole::Stiffness, &state)
            .unwrap();
        assert_eq!(
            load.component(3).unwrap().semantic_kind(),
            Some(runmat_types::physical::QuantityKind::Moment)
        );
        assert_eq!(
            physical
                .coefficient(StructuralOperatorRole::Stiffness, 3, 3)
                .unwrap()
                .angle_power(),
            -1
        );
        assert!(
            (physical
                .elastic_energy(&state, &load)
                .unwrap()
                .canonical_value()
                - 0.25)
                .abs()
                < 1e-12
        );
        let velocity = layout
            .physical_field(
                StructuralQuantityLane::State {
                    derivative_order: 1,
                },
                &values,
            )
            .unwrap();
        assert!(matches!(
            physical.apply(StructuralOperatorRole::Stiffness, &velocity),
            Err(StructuralPhysicalFieldError::CoordinateRoleMismatch)
        ));
    }
}

/// The coordinate transform of an operator that has already been reduced.
/// This is distinct from constraints still awaiting elimination.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralReducedBasis(std::sync::Arc<crate::constraints::AffineConstraintReduction>);
impl StructuralReducedBasis {
    pub fn try_new(
        full: &StructuralDofLayout,
        reduction: crate::constraints::AffineConstraintReduction,
    ) -> Result<Self, StructuralPhysicalFieldError> {
        let shared = std::sync::Arc::new(reduction);
        StructuralSolveLayout::Affine {
            full,
            reduction: std::sync::Arc::clone(&shared),
        }
        .validate()?;
        Ok(Self(shared))
    }
}
impl serde::Serialize for StructuralReducedBasis {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(self.0.as_ref(), serializer)
    }
}
impl<'de> serde::Deserialize<'de> for StructuralReducedBasis {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let reduction =
            <crate::constraints::AffineConstraintReduction as serde::Deserialize>::deserialize(
                deserializer,
            )?;
        reduction.validate().map_err(serde::de::Error::custom)?;
        Ok(Self(std::sync::Arc::new(reduction)))
    }
}

impl AssemblySummary {
    pub(crate) fn structural_solve_layout(&self) -> StructuralSolveLayout<'_> {
        match &self.structural_reduced_basis {
            Some(basis) => StructuralSolveLayout::Affine {
                full: &self.structural_dof_layout,
                reduction: std::sync::Arc::clone(&basis.0),
            },
            None => StructuralSolveLayout::Full(&self.structural_dof_layout),
        }
    }
}
