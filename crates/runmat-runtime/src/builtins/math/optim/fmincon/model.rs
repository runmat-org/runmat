//! Original constraint identities survive sparse SQP linearization. A solver
//! certificate for this model is only a statement about the local subproblem.
use super::{evaluation::Point, workspace};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::{build_runtime_error, BuiltinResult};
use clarabel::{algebra::CscMatrix, solver::SupportedConeT};
use runmat_builtins::BuiltinCatalogIdentity;
mod borrowed;
mod values;
pub(super) use borrowed::GeometryView;
pub(super) use values::ConstraintValues;

pub(super) struct LinearConstraints {
    pub matrix: CscMatrix<f64>,
    pub rhs: Vec<f64>,
}
pub(super) struct Geometry {
    pub variables: usize,
    pub inequality: LinearConstraints,
    pub equality: LinearConstraints,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Role {
    EqualityLinear(usize),
    EqualityNonlinear(usize),
    InequalityLinear(usize),
    InequalityNonlinear(usize),
    Lower(usize),
    Upper(usize),
}
pub(super) struct Linearization {
    pub semi_infinite: Option<super::semi_infinite::GenerationStamp>,
    pub a: CscMatrix<f64>,
    /// b=-c(x), so original/local violation is obtained from -b at p=0.
    pub b: Vec<f64>,
    pub roles: Vec<Role>,
    pub equalities: usize,
    pub cones: Vec<SupportedConeT<f64>>,
}

fn invalid(owner: BuiltinCatalogIdentity, detail: &str) -> crate::RuntimeError {
    build_runtime_error(format!("{}: {detail}", owner.name))
        .with_builtin(owner.name)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
async fn activity(
    matrix: &borrowed::MatrixView<'_>,
    x: &[f64],
    owner: BuiltinCatalogIdentity,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Vec<f64>>> {
    let mut result = workspace::capacity(matrix.m, owner)?;
    for _ in 0..matrix.m {
        result.push(0.0);
        cooperation.completed(1).await?;
    }
    for column in 0..matrix.n {
        for (row, value) in matrix.column(column) {
            result[row] += value * x[column];
            if !result[row].is_finite() {
                return Ok(None);
            }
            cooperation.completed(1).await?;
        }
        cooperation.completed(1).await?;
    }
    Ok(Some(result))
}

impl Geometry {
    pub fn view(&self) -> GeometryView<'_> {
        GeometryView::owned(self)
    }
    pub async fn violation(
        &self,
        x: &[f64],
        point: &Point,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<(f64, f64)>> {
        self.view().violation(x, point, owner, cooperation).await
    }
    pub async fn linearize(
        &self,
        x: &[f64],
        point: &Point,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Linearization> {
        self.view().linearize(x, point, owner, cooperation).await
    }
    pub async fn linearize_with_values(
        &self,
        x: &[f64],
        point: &Point,
        values: ConstraintValues,
        owner: BuiltinCatalogIdentity,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Linearization> {
        self.view()
            .linearize_with_values(x, point, values, owner, cooperation)
            .await
    }
}

impl Linearization {
    pub async fn violation(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<(f64, f64)> {
        let mut max = 0.0f64;
        let mut sum = 0.0;
        for (row, &rhs) in self.b.iter().enumerate() {
            let value = if row < self.equalities {
                rhs.abs()
            } else {
                (-rhs).max(0.0)
            };
            max = max.max(value);
            sum += value;
            cooperation.completed(1).await?;
        }
        Ok((max, sum))
    }
}
