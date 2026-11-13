//! Borrowed linear geometry for the same SQP/KKT owner. Original user
//! Geometry remains owned; an internal model may borrow a validated MIP prefix.
use super::{activity, invalid, workspace, ConstraintValues, Geometry, Linearization, Point};
use crate::builtins::math::optim::{
    common::ScalarCooperation,
    intlinprog::feasibility::{Domain, OriginalLinear},
};
use crate::BuiltinResult;
use clarabel::{algebra::CscMatrix, solver::SupportedConeT};
use runmat_builtins::BuiltinCatalogIdentity;
use std::ops::Range;

enum Source<'a> {
    Owned(&'a CscMatrix<f64>),
    Original {
        source: OriginalLinear<'a>,
        rows: Range<usize>,
    },
}
pub(super) struct MatrixView<'a> {
    pub m: usize,
    pub n: usize,
    source: Source<'a>,
}
pub(super) enum Column<'a> {
    Owned {
        rows: &'a [usize],
        values: &'a [f64],
        next: usize,
    },
    Original {
        source: &'a OriginalLinear<'a>,
        rows: Range<usize>,
        column: usize,
        next: usize,
    },
}
impl Iterator for Column<'_> {
    type Item = (usize, f64);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Owned { rows, values, next } => {
                let row = *rows.get(*next)?;
                let value = values[*next];
                *next += 1;
                Some((row, value))
            }
            Self::Original {
                source,
                rows,
                column,
                next,
            } => {
                let (row, value) = source.entry(*column, *next)?;
                *next += 1;
                if row >= rows.end {
                    return None;
                }
                Some((row - rows.start, value))
            }
        }
    }
}
impl<'a> MatrixView<'a> {
    pub(super) fn owned(matrix: &'a CscMatrix<f64>) -> Self {
        Self {
            m: matrix.m,
            n: matrix.n,
            source: Source::Owned(matrix),
        }
    }
    fn original(source: OriginalLinear<'a>, rows: Range<usize>) -> Self {
        Self {
            m: rows.len(),
            n: source.variables(),
            source: Source::Original { source, rows },
        }
    }
    pub(super) fn column(&self, column: usize) -> Column<'_> {
        if self.m == 0 {
            return Column::Owned {
                rows: &[],
                values: &[],
                next: 0,
            };
        }
        match &self.source {
            Source::Owned(matrix) => {
                let range = matrix.colptr[column]..matrix.colptr[column + 1];
                Column::Owned {
                    rows: &matrix.rowval[range.clone()],
                    values: &matrix.nzval[range],
                    next: 0,
                }
            }
            Source::Original { source, rows } => {
                // entry() observes only the validated sorted original
                // prefix. At most usize::BITS steps find the first selected
                // row; next() never performs an uncharged linear skip.
                let (mut lower, mut upper) = (0usize, usize::MAX);
                if rows.start == 0 {
                    upper = 0;
                }
                while lower < upper {
                    let middle = lower + (upper - lower) / 2;
                    if source
                        .entry(column, middle)
                        .is_some_and(|(row, _)| row < rows.start)
                    {
                        lower = middle + 1;
                    } else {
                        upper = middle;
                    }
                }
                Column::Original {
                    source,
                    rows: rows.clone(),
                    column,
                    next: lower,
                }
            }
        }
    }
}
pub(super) struct LinearView<'a> {
    pub matrix: MatrixView<'a>,
    pub rhs: &'a [f64],
}
pub(in crate::builtins::math::optim::fmincon) struct GeometryView<'a> {
    pub variables: usize,
    pub(super) inequality: LinearView<'a>,
    pub(super) equality: LinearView<'a>,
    pub lower: &'a [f64],
    pub upper: &'a [f64],
    owned: Option<&'a Geometry>,
}
impl<'a> GeometryView<'a> {
    pub(super) fn owned(geometry: &'a Geometry) -> Self {
        Self {
            variables: geometry.variables,
            inequality: LinearView {
                matrix: MatrixView::owned(&geometry.inequality.matrix),
                rhs: &geometry.inequality.rhs,
            },
            equality: LinearView {
                matrix: MatrixView::owned(&geometry.equality.matrix),
                rhs: &geometry.equality.rhs,
            },
            lower: &geometry.lower,
            upper: &geometry.upper,
            owned: Some(geometry),
        }
    }
    pub(in crate::builtins::math::optim::fmincon) async fn original(
        domain: &'a Domain,
        lower: &'a [f64],
        upper: &'a [f64],
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let (original_lower, original_upper) = domain.bounds();
        if lower.len() != original_lower.len() || upper.len() != original_upper.len() {
            return Err(invalid(owner, "borrowed SQP variable extent changed"));
        }
        for ((&lower, &upper), (&a, &b)) in lower
            .iter()
            .zip(upper)
            .zip(original_lower.iter().zip(original_upper))
        {
            if !lower.is_finite() || !upper.is_finite() || lower > upper || lower < a || upper > b {
                return Err(invalid(
                    owner,
                    "borrowed SQP bounds differ from the original model domain",
                ));
            }
            cooperation.completed(1).await?;
        }
        let equality = domain.original_linear();
        let inequality = domain.original_linear();
        let split = equality.equalities();
        let rows = inequality.rows();
        let equality_rhs = &equality.rhs()[..split];
        let inequality_rhs = &inequality.rhs()[split..];
        Ok(Self {
            variables: lower.len(),
            equality: LinearView {
                matrix: MatrixView::original(equality, 0..split),
                rhs: equality_rhs,
            },
            inequality: LinearView {
                matrix: MatrixView::original(inequality, split..rows),
                rhs: inequality_rhs,
            },
            lower,
            upper,
            owned: None,
        })
    }
    pub(in crate::builtins::math::optim::fmincon) fn require_owned(
        &self,
        owner: BuiltinCatalogIdentity,
    ) -> BuiltinResult<&Geometry> {
        self.owned.ok_or_else(|| {
            invalid(
                owner,
                "selected dynamic domain requires its original owned geometry",
            )
        })
    }
}

impl GeometryView<'_> {
    pub async fn violation(
        &self,
        x: &[f64],
        point: &Point,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<(f64, f64)>> {
        if x.len() != self.variables {
            return Err(invalid(owner, "original geometry variable extent changed"));
        }
        let mut maximum = 0.0f64;
        let mut sum = 0.0;
        for (matrix, rhs, equality) in [
            (&self.equality.matrix, self.equality.rhs, true),
            (&self.inequality.matrix, self.inequality.rhs, false),
        ] {
            let Some(values) = activity(matrix, x, owner, cooperation).await? else {
                return Ok(None);
            };
            for (&value, &rhs) in values.iter().zip(rhs) {
                let residual = value - rhs;
                let violation = if equality {
                    residual.abs()
                } else {
                    residual.max(0.0)
                };
                if !violation.is_finite() {
                    return Ok(None);
                }
                maximum = maximum.max(violation);
                sum += violation;
                cooperation.completed(1).await?;
            }
        }
        for (values, equality) in [(&point.equality, true), (&point.inequality, false)] {
            for &value in values {
                let violation = if equality {
                    value.abs()
                } else {
                    value.max(0.0)
                };
                maximum = maximum.max(violation);
                sum += violation;
                cooperation.completed(1).await?;
            }
        }
        for variable in 0..x.len() {
            let violation = (self.lower[variable] - x[variable]).max(0.0)
                + (x[variable] - self.upper[variable]).max(0.0);
            maximum = maximum.max(violation);
            sum += violation;
            cooperation.completed(1).await?;
        }
        Ok((maximum.is_finite() && sum.is_finite()).then_some((maximum, sum)))
    }
    pub async fn linearize(
        &self,
        x: &[f64],
        point: &Point,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Linearization> {
        let values = self
            .constraint_values(x, point, owner, cooperation)
            .await?
            .ok_or_else(|| invalid(owner, "accepted constraint activity or residual overflow"))?;
        self.linearize_with_values(x, point, values, owner, cooperation)
            .await
    }
    pub async fn linearize_with_values(
        &self,
        x: &[f64],
        point: &Point,
        values: ConstraintValues,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Linearization> {
        let neq = self.equality.rhs.len();
        let nineq = self.inequality.rhs.len();
        let ConstraintValues {
            rhs: b,
            roles,
            equalities: eq,
            lower_rows,
            upper_rows,
        } = values;
        let inequality_jacobian = point
            .inequality_jacobian
            .as_ref()
            .ok_or_else(|| invalid(owner, "missing evaluated inequality Jacobian"))?;
        let equality_jacobian = point
            .equality_jacobian
            .as_ref()
            .ok_or_else(|| invalid(owner, "missing evaluated equality Jacobian"))?;
        let equality_jacobian = MatrixView::owned(equality_jacobian);
        let inequality_jacobian = MatrixView::owned(inequality_jacobian);
        let mut offsets = workspace::capacity(
            x.len()
                .checked_add(1)
                .ok_or_else(|| invalid(owner, "column extent overflow"))?,
            owner,
        )?;
        let mut rows = Vec::new();
        let mut values = Vec::new();
        offsets.push(0);
        for variable in 0..x.len() {
            for (matrix, offset) in [
                (&self.equality.matrix, 0),
                (&equality_jacobian, neq),
                (&self.inequality.matrix, eq),
                (&inequality_jacobian, eq + nineq),
            ] {
                for (row, value) in matrix.column(variable) {
                    workspace::push(&mut rows, offset + row, owner)?;
                    workspace::push(&mut values, value, owner)?;
                    cooperation.completed(1).await?;
                }
            }
            if let Some(row) = lower_rows[variable] {
                workspace::push(&mut rows, row, owner)?;
                workspace::push(&mut values, -1.0, owner)?;
            }
            if let Some(row) = upper_rows[variable] {
                workspace::push(&mut rows, row, owner)?;
                workspace::push(&mut values, 1.0, owner)?;
            }
            offsets.push(rows.len());
            cooperation.completed(1).await?;
        }
        let mut cones = Vec::new();
        if eq > 0 {
            cones.push(SupportedConeT::ZeroConeT(eq));
        }
        if b.len() > eq {
            cones.push(SupportedConeT::NonnegativeConeT(b.len() - eq));
        }
        let a = CscMatrix::new(b.len(), x.len(), offsets, rows, values);
        Ok(Linearization {
            semi_infinite: super::super::semi_infinite::stamp(point),
            a,
            b,
            roles,
            equalities: eq,
            cones,
        })
    }
}
