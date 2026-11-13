//! Original row identities and positive barrier generations. Fixed bounds are
//! exact zero-direction equalities, not two impossible positive slacks.
use super::super::{
    model::{Geometry, Linearization, Role},
    workspace,
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use runmat_builtins::BuiltinCatalogIdentity;

#[derive(Clone, Copy)]
pub(super) enum Row {
    Equality(usize),
    Inequality(usize),
    Fixed,
}
pub(super) struct Layout {
    pub rows: Vec<Row>,
    pub fixed: Vec<usize>,
    pub original_equalities: usize,
    pub inequalities: usize,
}
impl Layout {
    pub async fn new(
        original: &Linearization,
        geometry: &Geometry,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut rows = workspace::capacity(original.roles.len(), owner)?;
        let mut fixed = workspace::capacity(geometry.variables, owner)?;
        for variable in 0..geometry.variables {
            if geometry.lower[variable] == geometry.upper[variable] {
                fixed.push(variable);
            }
            cooperation.completed(1).await?;
        }
        let mut inequalities = 0usize;
        for (index, role) in original.roles.iter().enumerate() {
            let row = if index < original.equalities {
                Row::Equality(index)
            } else {
                match role {
                    Role::Lower(variable) | Role::Upper(variable)
                        if geometry.lower[*variable] == geometry.upper[*variable] =>
                    {
                        Row::Fixed
                    }
                    _ => {
                        let next = Row::Inequality(inequalities);
                        inequalities = inequalities
                            .checked_add(1)
                            .ok_or_else(|| super::super::invalid("barrier row extent overflow"))?;
                        next
                    }
                }
            };
            rows.push(row);
            cooperation.completed(1).await?;
        }
        Ok(Self {
            rows,
            fixed,
            original_equalities: original.equalities,
            inequalities,
        })
    }
    pub fn equality_count(&self) -> Option<usize> {
        self.original_equalities.checked_add(self.fixed.len())
    }
    pub async fn original_multipliers(
        &self,
        original: &Linearization,
        gradient: &[f64],
        state: &State,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        if self.rows.len() != original.b.len()
            || state.z.len() != self.inequalities
            || state.y.len() != self.original_equalities
        {
            return Ok(None);
        }
        let mut values = workspace::capacity(original.b.len(), owner)?;
        for row in &self.rows {
            values.push(match *row {
                Row::Equality(index) => state.y[index],
                Row::Inequality(index) => state.z[index],
                Row::Fixed => 0.0,
            });
            cooperation.completed(1).await?;
        }
        let Some(residual) =
            super::super::validation::lagrangian(gradient, original, &values, owner, cooperation)
                .await?
        else {
            return Ok(None);
        };
        for (index, role) in original.roles.iter().enumerate() {
            if matches!(self.rows[index], Row::Fixed) {
                values[index] = match *role {
                    Role::Lower(variable) => residual[variable].max(0.0),
                    Role::Upper(variable) => (-residual[variable]).max(0.0),
                    _ => return Ok(None),
                };
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(values))
    }
}

pub(super) struct State {
    pub s: Vec<f64>,
    pub z: Vec<f64>,
    pub y: Vec<f64>,
    pub mu: f64,
}
impl State {
    pub async fn new(
        layout: &Layout,
        original: &Linearization,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Self>> {
        let mut s = zeros(layout.inequalities, owner, cooperation).await?;
        let mut z = zeros(layout.inequalities, owner, cooperation).await?;
        for (row, &kind) in layout.rows.iter().enumerate() {
            if let Row::Inequality(index) = kind {
                s[index] = original.b[row].max(1.0);
                z[index] = 1.0 / s[index];
                if !s[index].is_finite()
                    || !z[index].is_finite()
                    || s[index] <= 0.0
                    || z[index] <= 0.0
                {
                    return Ok(None);
                }
            }
            cooperation.completed(1).await?;
        }
        let y = zeros(layout.original_equalities, owner, cooperation).await?;
        let mut state = Self { s, z, y, mu: 0.0 };
        let Some(mean) = state.mean_complementarity(cooperation).await? else {
            return Ok(None);
        };
        state.mu = 0.1 * mean;
        Ok(Some(state))
    }
    pub async fn mean_complementarity(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<f64>> {
        if self.s.len() != self.z.len() {
            return Ok(None);
        }
        if self.s.is_empty() {
            return Ok(Some(0.0));
        }
        let count = self.s.len() as f64;
        let mut mean = 0.0;
        for (&s, &z) in self.s.iter().zip(&self.z) {
            if !s.is_finite() || !z.is_finite() || s <= 0.0 || z <= 0.0 {
                return Ok(None);
            }
            let product = s * z;
            if !product.is_finite() {
                return Ok(None);
            }
            mean += product / count;
            if !mean.is_finite() {
                return Ok(None);
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(mean))
    }
    pub async fn residuals(
        &self,
        layout: &Layout,
        original: &Linearization,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Residuals>> {
        let mut equality = zeros(layout.original_equalities, owner, cooperation).await?;
        let mut primal = zeros(layout.inequalities, owner, cooperation).await?;
        let mut centering = zeros(layout.inequalities, owner, cooperation).await?;
        for (row, &kind) in layout.rows.iter().enumerate() {
            match kind {
                Row::Equality(index) => equality[index] = -original.b[row],
                Row::Inequality(index) => {
                    primal[index] = self.s[index] - original.b[row];
                    centering[index] = self.s[index] * self.z[index] - self.mu;
                    if !primal[index].is_finite() || !centering[index].is_finite() {
                        return Ok(None);
                    }
                }
                Row::Fixed => {}
            }
            cooperation.completed(1).await?;
        }
        Ok(Some(Residuals {
            equality,
            primal,
            centering,
        }))
    }
}
pub(super) struct Residuals {
    pub equality: Vec<f64>,
    pub primal: Vec<f64>,
    pub centering: Vec<f64>,
}

pub(super) async fn zeros(
    length: usize,
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = workspace::capacity(length, owner)?;
    for _ in 0..length {
        values.push(0.0);
        cooperation.completed(1).await?;
    }
    Ok(values)
}

pub(super) use super::super::bound_geometry::interior_start;
