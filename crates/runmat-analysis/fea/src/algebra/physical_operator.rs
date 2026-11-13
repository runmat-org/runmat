//! One shared physical-operator adapter over the existing sparse numerical kernel.
use super::{spmv, RealSparseMatrix};
impl runmat_types::physical::QuantityOperatorStorage for RealSparseMatrix {
    fn rows(&self) -> usize {
        self.rows()
    }
    fn columns(&self) -> usize {
        self.columns()
    }
    fn all_finite(&self) -> bool {
        // RealSparseMatrix::try_new (including serde admission) validates every
        // entry; private immutable Arc storage has no mutation escape hatch.
        true
    }
    fn coefficient(&self, row: usize, column: usize) -> f64 {
        self.get(row, column)
    }
    fn apply_canonical(&self, input: &[f64]) -> Result<Vec<f64>, &'static str> {
        spmv(self, input)
    }
}

/// Actual prescribed rows contain the state value, while free rows contain the
/// physical equation residual. Identity constraints do not have forcing units.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScalarDirichletLayout<'a, K: runmat_types::physical::QuantityMarker> {
    prescribed: &'a [bool],
    free_equation: runmat_types::physical::DynamicDimension,
    _state: std::marker::PhantomData<K>,
}
impl<'a, K: runmat_types::physical::QuantityMarker> ScalarDirichletLayout<'a, K> {
    pub fn new(
        prescribed: &'a [bool],
        free_equation: runmat_types::physical::DynamicDimension,
    ) -> Self {
        Self {
            prescribed,
            free_equation,
            _state: std::marker::PhantomData,
        }
    }
}
impl<K: runmat_types::physical::QuantityMarker> runmat_types::physical::QuantityLayout
    for ScalarDirichletLayout<'_, K>
{
    fn component_count(&self) -> usize {
        self.prescribed.len()
    }
    fn component_descriptor(
        &self,
        index: usize,
    ) -> Option<runmat_types::physical::DynamicDimension> {
        Some(if *self.prescribed.get(index)? {
            runmat_types::physical::DynamicDimension::quantity_type(K::KIND)
                .with_semantic_kind(Some(K::KIND))
                .ok()?
        } else {
            self.free_equation
        })
    }
}

/// A physics-owned scalar equation, including its actual mixed prescribed/free
/// RHS layout. Generic solver preparation is an explicit numerical boundary.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(bound(serialize = ""))]
pub(crate) struct PhysicalScalarSystem<K: runmat_types::physical::QuantityMarker> {
    schema_version: u16,
    state_quantity: runmat_types::physical::QuantityKind,
    free_equation: runmat_types::physical::DynamicDimension,
    numerical: super::AssembledRealSystem,
    #[serde(skip)]
    _state: std::marker::PhantomData<K>,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicalScalarSystemData {
    schema_version: u16,
    state_quantity: runmat_types::physical::QuantityKind,
    free_equation: runmat_types::physical::DynamicDimension,
    numerical: super::AssembledRealSystem,
}
impl<'de, K: runmat_types::physical::QuantityMarker> serde::Deserialize<'de>
    for PhysicalScalarSystem<K>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let data = <PhysicalScalarSystemData as serde::Deserialize>::deserialize(deserializer)?;
        if data.schema_version != 1 || data.state_quantity != K::KIND {
            return Err(serde::de::Error::custom(
                "scalar physical system state/schema mismatch",
            ));
        }
        Self::try_new(data.numerical, data.free_equation).map_err(serde::de::Error::custom)
    }
}
impl<K: runmat_types::physical::QuantityMarker> PhysicalScalarSystem<K> {
    pub fn try_new(
        numerical: super::AssembledRealSystem,
        free_equation: runmat_types::physical::DynamicDimension,
    ) -> Result<Self, &'static str> {
        let system = Self {
            schema_version: 1,
            state_quantity: K::KIND,
            free_equation,
            numerical,
            _state: std::marker::PhantomData,
        };
        system.rhs_field()?;
        Ok(system)
    }
    pub fn require_equation(
        &self,
        expected: runmat_types::physical::DynamicDimension,
    ) -> Result<(), &'static str> {
        if self.free_equation != expected {
            return Err("scalar equation quantity differs from its physics owner");
        }
        Ok(())
    }
    pub fn rhs_field(
        &self,
    ) -> Result<
        runmat_types::physical::QuantityField<ScalarDirichletLayout<'_, K>, &[f64]>,
        &'static str,
    > {
        runmat_types::physical::QuantityField::try_new(
            ScalarDirichletLayout::new(self.numerical.constrained_dofs(), self.free_equation),
            self.numerical.load(),
        )
        .map_err(|_| "invalid scalar physical RHS")
    }
    pub fn physical_operator(
        &self,
    ) -> Result<
        runmat_types::physical::QuantityLinearOperator<
            runmat_types::physical::UniformQuantityLayout<K>,
            runmat_types::physical::UniformQuantityLayout<runmat_types::physical::DynamicDimension>,
            &RealSparseMatrix,
        >,
        runmat_types::physical::QuantityOperatorError,
    > {
        runmat_types::physical::QuantityLinearOperator::try_new(
            runmat_types::physical::UniformQuantityLayout::<K>::new(self.numerical.dof_count()),
            runmat_types::physical::UniformQuantityLayout::from_descriptor(
                self.numerical.dof_count(),
                self.free_equation,
            ),
            self.numerical.operators().stiffness(),
        )
    }
    pub fn with_canonical_system<T>(
        &self,
        kernel: impl FnOnce(&super::AssembledRealSystem) -> T,
    ) -> T {
        kernel(&self.numerical)
    }
    pub fn admit_solution(
        &self,
        result: crate::solve::linear::LinearSolveResult,
    ) -> Result<
        crate::solve::linear::LinearSolveResult<runmat_types::physical::Quantity<K, Vec<f64>>>,
        &'static str,
    > {
        result.try_map_solution(|values| {
            if values.len() != self.numerical.dof_count() {
                return Err("scalar solution coordinate count differs");
            }
            runmat_types::physical::Quantity::try_new(values)
                .map_err(|_| "scalar solution contains nonfinite values")
        })
    }
    pub fn dof_count(&self) -> usize {
        self.numerical.dof_count()
    }
    // Planning and residual-normalization operate on numerical storage, while
    // rhs_field/physical_operator retain its physical interpretation for execution.
    pub fn operators(&self) -> &super::CanonicalRealOperators {
        self.numerical.operators()
    }
    pub fn linear_rhs(&self) -> Vec<f64> {
        self.numerical.linear_rhs()
    }
    #[cfg(test)]
    pub fn load(&self) -> &[f64] {
        self.numerical.load()
    }
    #[cfg(test)]
    pub fn prepare_linear(
        &self,
    ) -> Result<super::PreparedRealLinearSystem, super::RealSystemError> {
        self.numerical.prepare_linear()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{
        AssembledRealSystem, CanonicalRealOperators, MatrixProperties, OperatorRole,
    };
    use runmat_types::physical::{kind, DynamicDimension, QuantityKind};
    #[test]
    fn scalar_system_preserves_mixed_constraint_units_and_validates_persisted_state_kind() {
        let matrix = |values, role| {
            RealSparseMatrix::diagonal(
                values,
                role,
                MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
            )
            .unwrap()
        };
        let operators = CanonicalRealOperators::try_new(
            matrix(vec![2., 3.], OperatorRole::Conductivity),
            matrix(vec![1., 1.], OperatorRole::Capacity),
            matrix(vec![0., 0.], OperatorRole::Damping),
        )
        .unwrap();
        let numerical =
            AssembledRealSystem::try_new(operators, vec![true, false], vec![10., 60.]).unwrap();
        let power = DynamicDimension::quantity_type(QuantityKind::Power)
            .with_semantic_kind(Some(QuantityKind::Power))
            .unwrap();
        let system =
            PhysicalScalarSystem::<kind::TemperatureDifference>::try_new(numerical, power).unwrap();
        let rhs = system.rhs_field().unwrap();
        assert_eq!(
            rhs.component(0).unwrap().semantic_kind(),
            Some(QuantityKind::TemperatureDifference)
        );
        assert_eq!(
            rhs.component(1).unwrap().semantic_kind(),
            Some(QuantityKind::Power)
        );
        assert!(system.require_equation(power).is_ok());
        let encoded = serde_json::to_value(&system).unwrap();
        assert!(
            serde_json::from_value::<PhysicalScalarSystem<kind::ElectricPotential>>(
                encoded.clone()
            )
            .is_err()
        );
        let decoded =
            serde_json::from_value::<PhysicalScalarSystem<kind::TemperatureDifference>>(encoded)
                .unwrap();
        assert!(decoded.require_equation(power).is_ok());
        assert_eq!(
            decoded
                .physical_operator()
                .unwrap()
                .coefficient(0, 0)
                .unwrap()
                .dimension(),
            QuantityKind::Power
                .dimension()
                .divide(QuantityKind::TemperatureDifference.dimension())
                .unwrap()
        );
    }
}
