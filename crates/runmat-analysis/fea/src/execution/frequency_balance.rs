use serde::{Deserialize, Serialize};

use super::{AnalysisExecutionError, AnalysisExecutionResult};

/// One complex structural vector represented without an implicit storage convention.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "StructuralComplexVectorWire")]
pub struct StructuralComplexVector {
    real: Vec<f64>,
    imaginary: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StructuralComplexVectorWire {
    real: Vec<f64>,
    imaginary: Vec<f64>,
}

impl TryFrom<StructuralComplexVectorWire> for StructuralComplexVector {
    type Error = AnalysisExecutionError;

    fn try_from(wire: StructuralComplexVectorWire) -> Result<Self, Self::Error> {
        Self::new(wire.real, wire.imaginary)
    }
}

impl StructuralComplexVector {
    pub fn new(real: Vec<f64>, imaginary: Vec<f64>) -> AnalysisExecutionResult<Self> {
        let value = Self { real, imaginary };
        value.validate_nonempty()?;
        Ok(value)
    }

    pub fn real(&self) -> &[f64] {
        &self.real
    }

    pub fn imaginary(&self) -> &[f64] {
        &self.imaginary
    }

    pub fn len(&self) -> usize {
        self.real.len()
    }

    pub fn is_empty(&self) -> bool {
        self.real.is_empty()
    }

    fn validate_nonempty(&self) -> AnalysisExecutionResult<()> {
        if self.real.is_empty()
            || self.real.len() != self.imaginary.len()
            || self
                .real
                .iter()
                .chain(&self.imaginary)
                .any(|value| !value.is_finite())
        {
            return Err(AnalysisExecutionError::Invalid(
                "complex structural vector has invalid shape or values".into(),
            ));
        }
        Ok(())
    }

    pub(super) fn validate_shape(&self, expected: usize) -> AnalysisExecutionResult<()> {
        self.validate_nonempty()?;
        if self.len() != expected {
            return Err(AnalysisExecutionError::Invalid(
                "complex structural vectors have inconsistent dimensions".into(),
            ));
        }
        Ok(())
    }
}

/// Full-order complex dynamic equilibrium at one physical response frequency.
///
/// The sign convention is
/// `inertia + damping + elastic - applied = residual`, matching transient structural balance.
/// Reactions are recovered separately because generalized constraint rows need not map to a
/// simple per-DOF subtraction. The constructor owns residual and relative-error calculation so
/// serialized evidence cannot assert inconsistent arithmetic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, try_from = "ComplexStructuralBalanceWire")]
pub struct ComplexStructuralBalance {
    applied_load: StructuralComplexVector,
    inertia_force: StructuralComplexVector,
    damping_force: StructuralComplexVector,
    elastic_force: StructuralComplexVector,
    equilibrium_residual: StructuralComplexVector,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ComplexStructuralBalanceWire {
    applied_load: StructuralComplexVector,
    inertia_force: StructuralComplexVector,
    damping_force: StructuralComplexVector,
    elastic_force: StructuralComplexVector,
    equilibrium_residual: StructuralComplexVector,
}

impl TryFrom<ComplexStructuralBalanceWire> for ComplexStructuralBalance {
    type Error = AnalysisExecutionError;

    fn try_from(wire: ComplexStructuralBalanceWire) -> Result<Self, Self::Error> {
        let balance = Self {
            applied_load: wire.applied_load,
            inertia_force: wire.inertia_force,
            damping_force: wire.damping_force,
            elastic_force: wire.elastic_force,
            equilibrium_residual: wire.equilibrium_residual,
        };
        balance.validate()?;
        Ok(balance)
    }
}

impl ComplexStructuralBalance {
    pub fn new(
        applied_load: StructuralComplexVector,
        inertia_force: StructuralComplexVector,
        damping_force: StructuralComplexVector,
        elastic_force: StructuralComplexVector,
    ) -> AnalysisExecutionResult<Self> {
        Self::new_unchecked(applied_load, inertia_force, damping_force, elastic_force)
    }

    pub fn applied_load(&self) -> &StructuralComplexVector {
        &self.applied_load
    }

    pub fn inertia_force(&self) -> &StructuralComplexVector {
        &self.inertia_force
    }

    pub fn damping_force(&self) -> &StructuralComplexVector {
        &self.damping_force
    }

    pub fn elastic_force(&self) -> &StructuralComplexVector {
        &self.elastic_force
    }

    pub fn equilibrium_residual(&self) -> &StructuralComplexVector {
        &self.equilibrium_residual
    }

    pub fn validate_for_dof_count(&self, dof_count: usize) -> AnalysisExecutionResult<()> {
        self.validate()?;
        if self.applied_load.len() != dof_count {
            return Err(AnalysisExecutionError::Invalid(
                "complex structural balance differs from the physical DOF count".into(),
            ));
        }
        Ok(())
    }

    fn validate(&self) -> AnalysisExecutionResult<()> {
        let expected = Self::new_unchecked(
            self.applied_load.clone(),
            self.inertia_force.clone(),
            self.damping_force.clone(),
            self.elastic_force.clone(),
        )?;
        if !same_vector_bits(&self.equilibrium_residual, &expected.equilibrium_residual) {
            return Err(AnalysisExecutionError::Identity(
                "complex structural balance does not contain its canonical residual",
            ));
        }
        Ok(())
    }

    fn new_unchecked(
        applied_load: StructuralComplexVector,
        inertia_force: StructuralComplexVector,
        damping_force: StructuralComplexVector,
        elastic_force: StructuralComplexVector,
    ) -> AnalysisExecutionResult<Self> {
        let dof_count = applied_load.len();
        applied_load.validate_shape(dof_count)?;
        inertia_force.validate_shape(dof_count)?;
        damping_force.validate_shape(dof_count)?;
        elastic_force.validate_shape(dof_count)?;
        let equilibrium_residual = residual(
            &applied_load,
            &inertia_force,
            &damping_force,
            &elastic_force,
        )?;
        Ok(Self {
            applied_load,
            inertia_force,
            damping_force,
            elastic_force,
            equilibrium_residual,
        })
    }

    pub(crate) fn closure_metrics(
        &self,
        reaction: &StructuralComplexVector,
    ) -> AnalysisExecutionResult<(f64, f64)> {
        self.validate()?;
        reaction.validate_shape(self.applied_load.len())?;
        let closure = combine_vectors(&[&self.equilibrium_residual], &[reaction])?;
        let internal = sum_three(
            &self.inertia_force,
            &self.damping_force,
            &self.elastic_force,
        )?;
        let residual_norm = complex_norm(&closure);
        let relative = residual_norm
            / complex_norm(&self.applied_load)
                .max(complex_norm(&internal))
                .max(complex_norm(reaction))
                .max(f64::MIN_POSITIVE);
        if !residual_norm.is_finite() || !relative.is_finite() {
            return Err(AnalysisExecutionError::Invalid(
                "complex structural balance closure metrics are nonfinite".into(),
            ));
        }
        Ok((residual_norm, relative))
    }
}

fn residual(
    applied: &StructuralComplexVector,
    inertia: &StructuralComplexVector,
    damping: &StructuralComplexVector,
    elastic: &StructuralComplexVector,
) -> AnalysisExecutionResult<StructuralComplexVector> {
    combine_vectors(&[inertia, damping, elastic], &[applied])
}

fn sum_three(
    first: &StructuralComplexVector,
    second: &StructuralComplexVector,
    third: &StructuralComplexVector,
) -> AnalysisExecutionResult<StructuralComplexVector> {
    combine_vectors(&[first, second, third], &[])
}

fn combine_vectors(
    positive: &[&StructuralComplexVector],
    negative: &[&StructuralComplexVector],
) -> AnalysisExecutionResult<StructuralComplexVector> {
    let size = positive
        .first()
        .or_else(|| negative.first())
        .ok_or_else(|| AnalysisExecutionError::Invalid("complex vector sum is empty".into()))?
        .len();
    for value in positive.iter().chain(negative) {
        value.validate_shape(size)?;
    }
    let component = |imaginary: bool| {
        (0..size)
            .map(|index| {
                let value = |vector: &&StructuralComplexVector| {
                    if imaginary {
                        vector.imaginary[index]
                    } else {
                        vector.real[index]
                    }
                };
                positive.iter().map(value).sum::<f64>() - negative.iter().map(value).sum::<f64>()
            })
            .collect::<Vec<_>>()
    };
    StructuralComplexVector::new(component(false), component(true))
}

fn complex_norm(value: &StructuralComplexVector) -> f64 {
    value
        .real
        .iter()
        .chain(&value.imaginary)
        .fold(0.0_f64, |norm, component| norm.hypot(*component))
}

fn same_vector_bits(left: &StructuralComplexVector, right: &StructuralComplexVector) -> bool {
    left.real
        .iter()
        .zip(&right.real)
        .chain(left.imaginary.iter().zip(&right.imaginary))
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(real: &[f64], imaginary: &[f64]) -> StructuralComplexVector {
        StructuralComplexVector::new(real.to_vec(), imaginary.to_vec()).unwrap()
    }

    #[test]
    fn balance_derives_complex_residual_and_rejects_tampering() {
        let balance = ComplexStructuralBalance::new(
            vector(&[5.0, 1.0], &[2.0, -1.0]),
            vector(&[1.0, 0.0], &[0.0, -0.5]),
            vector(&[0.5, 0.0], &[0.5, 0.0]),
            vector(&[3.5, 1.0], &[1.5, -0.5]),
        )
        .unwrap();
        assert_eq!(balance.equilibrium_residual().real(), &[0.0, 0.0]);
        assert_eq!(balance.equilibrium_residual().imaginary(), &[0.0, 0.0]);
        assert_eq!(
            balance
                .closure_metrics(&vector(&[0.0, 0.0], &[0.0, 0.0]))
                .unwrap(),
            (0.0, 0.0)
        );

        let mut tampered = serde_json::to_value(&balance).unwrap();
        tampered["equilibrium_residual"]["real"][0] = serde_json::json!(0.5);
        assert!(serde_json::from_value::<ComplexStructuralBalance>(tampered).is_err());
        assert!(
            serde_json::from_value::<StructuralComplexVector>(serde_json::json!({
                "real": [1.0],
                "imaginary": [1.0, 2.0]
            }))
            .is_err()
        );
    }

    #[test]
    fn closure_norm_remains_finite_for_large_finite_components() {
        let large = f64::MAX / 4.0;
        let balance = ComplexStructuralBalance::new(
            vector(&[large], &[large]),
            vector(&[large], &[large]),
            vector(&[0.0], &[0.0]),
            vector(&[0.0], &[0.0]),
        )
        .unwrap();
        let (residual, relative) = balance.closure_metrics(&vector(&[0.0], &[0.0])).unwrap();
        assert_eq!(residual, 0.0);
        assert_eq!(relative, 0.0);
    }
}
