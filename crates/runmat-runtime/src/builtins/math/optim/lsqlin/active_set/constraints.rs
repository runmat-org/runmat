//! Original external row ordering: equalities, inequalities, then finite bounds.
use super::*;
pub(super) struct Constraints {
    pub matrix: Matrix,
    pub rhs: Vec<f64>,
    pub roles: Vec<conic::LinearRowRole>,
    pub equalities: usize,
    pub inequalities: usize,
}
impl Constraints {
    pub(super) async fn new(
        a: Matrix,
        b: Vec<f64>,
        aeq: Matrix,
        beq: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        c: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let n = lower.len();
        if upper.len() != n || a.n != n || aeq.n != n || b.len() != a.m || beq.len() != aeq.m {
            return Err(invalid("active-set constraint geometry is inconsistent"));
        }
        let mut count = b
            .len()
            .checked_add(beq.len())
            .ok_or_else(|| invalid("active-set row count overflow"))?;
        let base = count;
        for (&lower, &upper) in lower.iter().zip(&upper) {
            count = count
                .checked_add(usize::from(lower.is_finite()) + usize::from(upper.is_finite()))
                .ok_or_else(|| invalid("active-set bound row count overflow"))?;
            c.completed(1).await?;
        }
        let nnz = a
            .nzval
            .len()
            .checked_add(aeq.nzval.len())
            .and_then(|x| x.checked_add(count - base))
            .ok_or_else(|| invalid("active-set constraint storage overflow"))?;
        let mut rhs = capacity(count)?;
        let mut roles = capacity(count)?;
        let equalities = beq.len();
        let inequalities = b.len();
        for (row, value) in beq.into_iter().enumerate() {
            rhs.push(value);
            roles.push(conic::LinearRowRole::Equality(row));
            c.completed(1).await?;
        }
        for (row, value) in b.into_iter().enumerate() {
            rhs.push(value);
            roles.push(conic::LinearRowRole::Inequality(row));
            c.completed(1).await?;
        }
        for (column, (&lower, &upper)) in lower.iter().zip(&upper).enumerate() {
            if lower.is_finite() {
                rhs.push(-lower);
                roles.push(conic::LinearRowRole::Lower(column));
            }
            if upper.is_finite() {
                rhs.push(upper);
                roles.push(conic::LinearRowRole::Upper(column));
            }
            c.completed(1).await?;
        }
        let mut pointers = capacity(
            n.checked_add(1)
                .ok_or_else(|| invalid("active-set pointer overflow"))?,
        )?;
        let mut indices = capacity(nnz)?;
        let mut values = capacity(nnz)?;
        let mut bound = base;
        pointers.push(0);
        for column in 0..n {
            for entry in aeq.colptr[column]..aeq.colptr[column + 1] {
                indices.push(aeq.rowval[entry]);
                values.push(aeq.nzval[entry]);
                c.completed(1).await?;
            }
            for entry in a.colptr[column]..a.colptr[column + 1] {
                indices.push(equalities + a.rowval[entry]);
                values.push(a.nzval[entry]);
                c.completed(1).await?;
            }
            if lower[column].is_finite() {
                indices.push(bound);
                values.push(-1.);
                bound += 1;
            }
            if upper[column].is_finite() {
                indices.push(bound);
                values.push(1.);
                bound += 1;
            }
            pointers.push(indices.len());
            c.completed(1).await?;
        }
        if bound != count {
            return Err(invalid("active-set bound role layout changed"));
        }
        Ok(Self {
            matrix: Matrix::new(count, n, pointers, indices, values),
            rhs,
            roles,
            equalities,
            inequalities,
        })
    }
}
