//! Vetted sparse LLT proposes a factor; outward bounds alone authorize SPD.
//! If R=P-LL' and ||R||inf ||L^-1||inf ||L^-1||1 < 1, then P is SPD.
//! Triangular comparison recurrences bound inverse norms without an inverse.
use super::*;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::{cholesky, SupernodalThreshold};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }
    fn minus(self, rhs: Self) -> Self {
        if rhs.lo == 0.0 && rhs.hi == 0.0 {
            return self;
        }
        Self {
            lo: (self.lo - rhs.hi).next_down(),
            hi: (self.hi - rhs.lo).next_up(),
        }
    }
    fn product(a: f64, b: f64) -> Self {
        if a == 0.0 || b == 0.0 {
            return Self::point(0.0);
        }
        let value = a * b;
        Self {
            lo: value.next_down(),
            hi: value.next_up(),
        }
    }
    fn magnitude(self) -> Option<f64> {
        (self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi)
            .then(|| self.lo.abs().max(self.hi.abs()))
    }
}
fn add(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        a
    } else {
        (a + b).next_up()
    }
}
fn mul(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        (a * b).next_up()
    }
}

pub(super) async fn certify(
    p: &Matrix,
    cooperation: &mut ScalarCooperation,
    owner: runmat_builtins::BuiltinCatalogIdentity,
) -> BuiltinResult<bool> {
    cooperation.completed(0).await?;
    let a = SymbolicSparseColMatRef::new_checked(p.n, p.n, &p.colptr, None, &p.rowval);
    let symbolic = cholesky::factorize_symbolic_cholesky(
        a,
        faer::Side::Upper,
        cholesky::SymmetricOrdering::Amd,
        cholesky::CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
            ..Default::default()
        },
    )
    .map_err(|source| factor_error(source, owner))?;
    let mut values = vec![0.0; symbolic.len_val()];
    let mut memory = MemBuffer::try_new(
        symbolic.factorize_numeric_llt_scratch::<f64>(faer::Par::Seq, Default::default()),
    )
    .map_err(|source| factor_error(source, owner))?;
    let result = symbolic.factorize_numeric_llt(
        &mut values,
        SparseColMatRef::new(a, &p.nzval),
        faer::Side::Upper,
        Default::default(),
        faer::Par::Seq,
        MemStack::new(&mut memory),
        Default::default(),
    );
    match result {
        Ok(_) => {}
        Err(faer::linalg::solvers::LltError::NonPositivePivot { .. }) => return Ok(false),
    }
    drop(memory);
    if values.iter().any(|value| !value.is_finite()) {
        return Ok(false);
    }
    cooperation.completed(p.nzval.len() as u64 + 1).await?;
    let cholesky::SymbolicCholeskyRaw::Simplicial(factor) = symbolic.raw() else {
        return Ok(false);
    };
    let ptr = factor.col_ptr();
    let row = factor.row_idx();
    let n = p.n;
    let mut diagonal = vec![0.0; n];
    for col in 0..n {
        let Some(entry) = (ptr[col]..ptr[col + 1]).find(|&i| row[i] == col) else {
            return Ok(false);
        };
        diagonal[col] = values[entry];
        if !(diagonal[col] > 0.0 && diagonal[col].is_finite()) {
            return Ok(false);
        }
        cooperation.completed(1).await?;
    }
    let mut forward = vec![1.0; n];
    let mut norm_inf = 0.0_f64;
    for col in 0..n {
        forward[col] = (forward[col] / diagonal[col]).next_up();
        norm_inf = norm_inf.max(forward[col]);
        for entry in ptr[col]..ptr[col + 1] {
            let i = row[entry];
            if i > col {
                forward[i] = add(forward[i], mul(values[entry].abs(), forward[col]));
            }
            if entry % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    let mut backward = vec![1.0; n];
    let mut norm_one = 0.0_f64;
    for col in (0..n).rev() {
        for entry in ptr[col]..ptr[col + 1] {
            let i = row[entry];
            if i > col {
                backward[col] = add(backward[col], mul(values[entry].abs(), backward[i]));
            }
            if entry % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        backward[col] = (backward[col] / diagonal[col]).next_up();
        norm_one = norm_one.max(backward[col]);
        cooperation.completed(1).await?;
    }
    if !norm_inf.is_finite() || !norm_one.is_finite() {
        return Ok(false);
    }
    drop(forward);
    drop(backward);
    // Original P is permuted using precisely the symbolic factor's typed
    // inverse permutation. Norms and positive definiteness are invariant.
    let inverse = symbolic.perm().map(|p| p.arrays().1);
    let mut residual = vec![BTreeMap::<usize, Interval>::new(); n];
    for col in 0..n {
        for entry in p.colptr[col]..p.colptr[col + 1] {
            let old_row = p.rowval[entry];
            let i = inverse.map_or(old_row, |v| v[old_row]);
            let j = inverse.map_or(col, |v| v[col]);
            residual[i.min(j)].insert(i.max(j), Interval::point(p.nzval[entry]));
            if entry % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    let mut work = 0;
    for col in 0..n {
        for left in ptr[col]..ptr[col + 1] {
            for right in left..ptr[col + 1] {
                let i = row[left];
                let j = row[right];
                let cell = residual[i.min(j)]
                    .entry(i.max(j))
                    .or_insert(Interval::point(0.0));
                *cell = cell.minus(Interval::product(values[left], values[right]));
                work += 1;
                if work == 1024 {
                    cooperation.completed(1024).await?;
                    work = 0;
                }
            }
        }
        cooperation.completed(1).await?;
    }
    let mut row_sum = vec![0.0; n];
    for (i, entries) in residual.into_iter().enumerate() {
        for (j, value) in entries {
            let Some(magnitude) = value.magnitude() else {
                return Ok(false);
            };
            row_sum[i] = add(row_sum[i], magnitude);
            if i != j {
                row_sum[j] = add(row_sum[j], magnitude);
            }
            work += 1;
            if work == 1024 {
                cooperation.completed(1024).await?;
                work = 0;
            }
        }
    }
    cooperation.completed(work).await?;
    let residual = row_sum.into_iter().fold(0.0_f64, f64::max);
    let bound = mul(mul(residual, norm_inf), norm_one);
    Ok(bound.is_finite() && bound < 1.0)
}
fn factor_error(
    source: impl std::error::Error + Send + Sync + 'static,
    owner: runmat_builtins::BuiltinCatalogIdentity,
) -> RuntimeError {
    build_runtime_error(format!(
        "{}: PSD certificate factor allocation failed",
        owner.name
    ))
    .with_builtin(owner.name)
    .with_source(source)
    .build()
}
