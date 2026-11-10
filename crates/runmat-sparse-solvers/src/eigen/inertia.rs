//! Inertia of a shifted pencil `K − σM` by a multifrontal `LBLᵀ` that keeps no factor.
//!
//! Sylvester's law of inertia: the number of negative pivots of any congruence `K − σM =
//! L B Lᵀ` equals the number of eigenvalues of `K x = λ M x` below `σ` when `M` is symmetric
//! positive definite. A multifrontal elimination processes one supernode at a time: it assembles
//! the supernode's dense front from the matrix entries and the update matrices of its children,
//! factors the front's fully summed block with Bunch–Kaufman pivoting (inside that block, as the
//! retained-factor Sturm check does), counts the pivot signs, and passes the Schur complement up
//! to the parent. Haynsworth's inertia additivity, `In(F) = In(F₁₁) + In(F / F₁₁)`, applied front
//! by front gives the inertia of the whole matrix.
//!
//! No factor column is kept after its front is processed, so the resident peak is the largest
//! set of pending update matrices plus one front, not the factor. [`PencilInertia::analyze`]
//! fixes the ordering, the supernodes and the processing order, and quotes that peak exactly
//! before any numeric work ([`PencilInertia::resident_bytes`]); [`pencil_inertia_resident_bytes`]
//! quotes it from a pattern alone for resource planning. Children are processed in the order
//! that minimizes the peak (Liu's rule: decreasing subtree peak minus update size).
//!
//! Threading: the dense kernels (Bunch–Kaufman, the block solve and the Schur update) use the
//! caller's admitted pool through faer's Rayon parallelism. The processing order is fixed by
//! the analysis, so the counts do not depend on the thread count.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::lblt as bunch_kaufman;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::linalg::matmul::triangular::BlockStructure;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymbolicCholeskyRaw, SymmetricOrdering,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::{Accum, Conj, MatMut, Side};

use crate::control::SolveControl;
use crate::csr::CsrRef;
use crate::direct::{Inertia, SparseInput};
use crate::error::{SolverError, SolverResult};
use crate::ordering::{Ordering, Permutation};
use crate::par;

const WORD: u64 = std::mem::size_of::<usize>() as u64;
const SCALAR: u64 = std::mem::size_of::<f64>() as u64;
/// Bunch–Kaufman blocking columns the dense scratch reserves per fully summed row.
const DENSE_BLOCK_COLUMNS: u64 = 64;

/// Supernodal elimination schedule of one symmetric pattern.
#[derive(Debug, Clone)]
struct Schedule {
    n: usize,
    permutation: Permutation,
    /// First permuted column of each supernode (`len = supernodes + 1`).
    begin: Vec<usize>,
    /// Pattern rows below each supernode's diagonal block, permuted numbering.
    pattern_ptr: Vec<usize>,
    pattern_idx: Vec<usize>,
    /// Children of each supernode in processing order.
    child_ptr: Vec<usize>,
    child_idx: Vec<usize>,
    /// Supernodes in processing order (children before parents).
    order: Vec<usize>,
    /// Largest number of `f64` values held at once by pending updates, the current front and
    /// the dense kernels' scratch.
    peak_values: u64,
    /// Floating point operations of one numeric pass.
    flops: f64,
}

impl Schedule {
    fn width(&self, s: usize) -> usize {
        self.begin[s + 1] - self.begin[s]
    }
    fn pattern(&self, s: usize) -> &[usize] {
        &self.pattern_idx[self.pattern_ptr[s]..self.pattern_ptr[s + 1]]
    }
    fn children(&self, s: usize) -> &[usize] {
        &self.child_idx[self.child_ptr[s]..self.child_ptr[s + 1]]
    }
    fn structure_bytes(&self) -> u64 {
        let words = 2 * self.n
            + self.begin.len()
            + self.pattern_ptr.len()
            + self.pattern_idx.len()
            + self.child_ptr.len()
            + self.child_idx.len()
            + self.order.len()
            // Front position map.
            + self.n;
        words as u64 * WORD
    }
}

/// `f64` values node `s` holds at its peak beyond pending updates of other subtrees: its front,
/// then either its children's updates (assembly), the block solve copy and its scratch, or its
/// own update (extraction).
fn node_peak_values(front: u64, width: u64, rows: u64, children_updates: u64) -> u64 {
    let square = front * front;
    let solve = 2 * width * rows + width * DENSE_BLOCK_COLUMNS;
    square + children_updates.max(solve).max(rows * rows)
}

/// Union pattern of `K` and `M` (both full symmetric storage) as CSR arrays.
fn union_pattern(k: CsrRef<'_, f64>, m: CsrRef<'_, f64>) -> SolverResult<(Vec<usize>, Vec<usize>)> {
    let n = k.nrows();
    if k.ncols() != n || m.nrows() != n || m.ncols() != n {
        return Err(SolverError::Dimension(
            "pencil inertia requires square operators of one dimension".into(),
        ));
    }
    let mut row_ptr = Vec::with_capacity(n + 1);
    row_ptr.push(0);
    let mut col_idx = Vec::with_capacity(k.nnz().max(m.nnz()));
    for row in 0..n {
        let (a, _) = k.row(row);
        let (b, _) = m.row(row);
        let (mut i, mut j) = (0, 0);
        while i < a.len() || j < b.len() {
            let next = match (a.get(i), b.get(j)) {
                (Some(&x), Some(&y)) if x == y => {
                    i += 1;
                    j += 1;
                    x
                }
                (Some(&x), Some(&y)) if x < y => {
                    i += 1;
                    x
                }
                (Some(_), Some(&y)) => {
                    j += 1;
                    y
                }
                (Some(&x), None) => {
                    i += 1;
                    x
                }
                (None, Some(&y)) => {
                    j += 1;
                    y
                }
                (None, None) => unreachable!(),
            };
            col_idx.push(next);
        }
        row_ptr.push(col_idx.len());
    }
    Ok((row_ptr, col_idx))
}

fn analyze_schedule(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    ordering: &Ordering,
) -> SolverResult<Schedule> {
    let zeros = vec![0.0f64; col_idx.len()];
    let pattern = CsrRef::try_new(n, n, row_ptr, col_idx, &zeros)?;
    let permutation = crate::ordering::order(pattern, ordering)?;
    let (columns, _) = SparseInput::Csr(pattern).as_columns();
    let perm = faer::perm::PermRef::new_checked(&permutation.perm, &permutation.inverse, n);
    let symbolic = factorize_symbolic_cholesky(
        columns.faer_symbolic(),
        Side::Lower,
        SymmetricOrdering::Custom(perm),
        CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SUPERNODAL,
            ..Default::default()
        },
    )?;
    drop(zeros);
    let SymbolicCholeskyRaw::Supernodal(supernodal) = symbolic.raw() else {
        return Err(SolverError::Breakdown(
            "pencil inertia requires a supernodal analysis".into(),
        ));
    };
    let count = supernodal.n_supernodes();
    let mut begin: Vec<usize> = supernodal.supernode_begin()[..count].to_vec();
    begin.push(supernodal.supernode_end()[count - 1]);
    let mut pattern_ptr: Vec<usize> = supernodal.col_ptr_for_row_idx().to_vec();
    let mut pattern_idx: Vec<usize> = supernodal.row_idx().to_vec();
    pattern_ptr.truncate(count + 1);
    pattern_idx.truncate(pattern_ptr[count]);
    drop(symbolic);
    // Supernode of each permuted column, then each supernode's parent.
    let mut owner = vec![0usize; n];
    for s in 0..count {
        owner[begin[s]..begin[s + 1]].fill(s);
    }
    let parent: Vec<Option<usize>> = (0..count)
        .map(|s| {
            pattern_idx[pattern_ptr[s]..pattern_ptr[s + 1]]
                .first()
                .map(|&row| owner[row])
        })
        .collect();
    drop(owner);
    // Subtree peaks bottom-up (a parent has a larger index than each child).
    let update = |s: usize| {
        let rows = (pattern_ptr[s + 1] - pattern_ptr[s]) as u64;
        rows * rows
    };
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); count];
    for s in 0..count {
        if let Some(p) = parent[s] {
            children[p].push(s);
        }
    }
    let mut subtree = vec![0u64; count];
    let mut flops = 0.0f64;
    let liu = |list: &mut Vec<usize>, subtree: &[u64]| {
        list.sort_by(|&a, &b| {
            let ka = subtree[a].saturating_sub(update(a));
            let kb = subtree[b].saturating_sub(update(b));
            kb.cmp(&ka).then(a.cmp(&b))
        });
    };
    for s in 0..count {
        let mut list = std::mem::take(&mut children[s]);
        liu(&mut list, &subtree);
        let mut held = 0u64;
        let mut peak = 0u64;
        for &c in &list {
            peak = peak.max(held + subtree[c]);
            held += update(c);
        }
        let width = (begin[s + 1] - begin[s]) as u64;
        let rows = (pattern_ptr[s + 1] - pattern_ptr[s]) as u64;
        let front = width + rows;
        peak = peak.max(node_peak_values(front, width, rows, held));
        subtree[s] = peak;
        children[s] = list;
        let (p, m) = (width as f64, rows as f64);
        // Bunch–Kaufman of the block, its solve against the coupling block, and the
        // lower-triangular Schur update.
        flops += p * p * p / 3.0 + 2.0 * p * p * m + p * m * m;
    }
    let mut roots: Vec<usize> = (0..count).filter(|&s| parent[s].is_none()).collect();
    liu(&mut roots, &subtree);
    let mut peak_values = 0u64;
    let mut held = 0u64;
    for &r in &roots {
        peak_values = peak_values.max(held + subtree[r]);
        held += update(r);
    }
    // Depth-first postorder following the chosen child order.
    let mut order = Vec::with_capacity(count);
    let mut stack: Vec<(usize, usize)> = roots.iter().rev().map(|&r| (r, 0)).collect();
    while let Some((s, next)) = stack.pop() {
        if next < children[s].len() {
            stack.push((s, next + 1));
            stack.push((children[s][next], 0));
        } else {
            order.push(s);
        }
    }
    let mut child_ptr = Vec::with_capacity(count + 1);
    child_ptr.push(0);
    let mut child_idx = Vec::with_capacity(count);
    for list in &children {
        child_idx.extend_from_slice(list);
        child_ptr.push(child_idx.len());
    }
    Ok(Schedule {
        n,
        permutation,
        begin,
        pattern_ptr,
        pattern_idx,
        child_ptr,
        child_idx,
        order,
        peak_values,
        flops,
    })
}

/// Lower triangle of `P A Pᵀ` in compressed columns for `A = K` and `A = M` on their union
/// pattern.
#[derive(Debug, Clone)]
struct PermutedLower {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    k: Vec<f64>,
    m: Vec<f64>,
}

impl PermutedLower {
    fn entries_of(row_ptr: &[usize], col_idx: &[usize], inverse: &[usize]) -> usize {
        let n = inverse.len();
        (0..n)
            .map(|row| {
                col_idx[row_ptr[row]..row_ptr[row + 1]]
                    .iter()
                    .filter(|&&col| inverse[row] >= inverse[col])
                    .count()
            })
            .sum()
    }

    fn bytes(n: usize, entries: usize) -> u64 {
        (n as u64 + 1) * WORD + entries as u64 * (WORD + 2 * SCALAR)
    }

    fn build(
        k: CsrRef<'_, f64>,
        m: CsrRef<'_, f64>,
        row_ptr: &[usize],
        col_idx: &[usize],
        inverse: &[usize],
    ) -> Self {
        let n = inverse.len();
        let mut col_ptr = vec![0usize; n + 1];
        for row in 0..n {
            for &col in &col_idx[row_ptr[row]..row_ptr[row + 1]] {
                let (i, j) = (inverse[row], inverse[col]);
                if i >= j {
                    col_ptr[j + 1] += 1;
                }
            }
        }
        for j in 0..n {
            col_ptr[j + 1] += col_ptr[j];
        }
        let entries = col_ptr[n];
        let mut next = col_ptr[..n].to_vec();
        let mut row_idx = vec![0usize; entries];
        let mut kv = vec![0.0; entries];
        let mut mv = vec![0.0; entries];
        // Rows ascending give each permuted column its rows in ascending original order; the
        // front assembly places entries by position, so in-column order does not matter.
        for row in 0..n {
            let (kc, kx) = k.row(row);
            let (mc, mx) = m.row(row);
            let (mut a, mut b) = (0, 0);
            for &col in &col_idx[row_ptr[row]..row_ptr[row + 1]] {
                let mut kval = 0.0;
                let mut mval = 0.0;
                while a < kc.len() && kc[a] < col {
                    a += 1;
                }
                if a < kc.len() && kc[a] == col {
                    kval = kx[a];
                }
                while b < mc.len() && mc[b] < col {
                    b += 1;
                }
                if b < mc.len() && mc[b] == col {
                    mval = mx[b];
                }
                let (i, j) = (inverse[row], inverse[col]);
                if i >= j {
                    let slot = next[j];
                    next[j] += 1;
                    row_idx[slot] = i;
                    kv[slot] = kval;
                    mv[slot] = mval;
                }
            }
        }
        Self {
            col_ptr,
            row_idx,
            k: kv,
            m: mv,
        }
    }
}

/// Resident bytes and work of a [`PencilInertia`] on the union pattern given as CSR arrays,
/// for resource planning before values exist.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PencilInertiaEstimate {
    /// Peak bytes of one inertia count: schedule, permuted operators, and the dense fronts,
    /// pending updates and kernel scratch.
    pub resident_bytes: u64,
    /// Floating point operations of one count.
    pub flops: f64,
}

/// Quotes [`PencilInertia::resident_bytes`] for the union pattern of `K` and `M` given as CSR
/// arrays (full symmetric storage), without values.
pub fn pencil_inertia_estimate(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    ordering: &Ordering,
) -> SolverResult<PencilInertiaEstimate> {
    let schedule = analyze_schedule(n, row_ptr, col_idx, ordering)?;
    let entries = PermutedLower::entries_of(row_ptr, col_idx, &schedule.permutation.inverse);
    Ok(PencilInertiaEstimate {
        resident_bytes: schedule
            .structure_bytes()
            .saturating_add(PermutedLower::bytes(n, entries))
            .saturating_add(schedule.peak_values.saturating_mul(SCALAR)),
        flops: schedule.flops,
    })
}

/// Counts eigenvalues of `K x = λ M x` below any shift without retaining a factor.
#[derive(Debug, Clone)]
pub struct PencilInertia {
    schedule: Schedule,
    lower: PermutedLower,
}

impl PencilInertia {
    /// Orders the union pattern of `K` and `M`, analyzes its supernodes and fixes the
    /// processing order. Both operators are full symmetric storage.
    pub fn analyze(
        k: CsrRef<'_, f64>,
        m: CsrRef<'_, f64>,
        ordering: &Ordering,
        control: SolveControl<'_>,
    ) -> SolverResult<Self> {
        control.check()?;
        let (row_ptr, col_idx) = union_pattern(k, m)?;
        let schedule = analyze_schedule(k.nrows(), &row_ptr, &col_idx, ordering)?;
        control.check()?;
        let lower = PermutedLower::build(k, m, &row_ptr, &col_idx, &schedule.permutation.inverse);
        Ok(Self { schedule, lower })
    }

    /// Peak resident bytes of [`Self::count`], including this analysis.
    pub fn resident_bytes(&self) -> u64 {
        self.schedule
            .structure_bytes()
            .saturating_add(PermutedLower::bytes(
                self.schedule.n,
                self.lower.row_idx.len(),
            ))
            .saturating_add(self.schedule.peak_values.saturating_mul(SCALAR))
    }

    pub fn flops(&self) -> f64 {
        self.schedule.flops
    }

    pub fn dimension(&self) -> usize {
        self.schedule.n
    }

    /// Inertia of `K − shift·M`. A zero or non-finite pivot is reported as
    /// [`SolverError::Singular`]: the shift is (numerically) an eigenvalue.
    pub fn count(&self, shift: f64, control: SolveControl<'_>) -> SolverResult<Inertia> {
        let schedule = &self.schedule;
        let lower = &self.lower;
        let parallel = par::faer_par();
        let mut position = vec![0usize; schedule.n];
        let supernodes = schedule.begin.len() - 1;
        let mut pending: Vec<Option<Vec<f64>>> = vec![None; supernodes];
        let mut inertia = Inertia::default();
        for (step, &s) in schedule.order.iter().enumerate() {
            if step % 256 == 0 {
                control.check()?;
            }
            let (start, end) = (schedule.begin[s], schedule.begin[s + 1]);
            let width = schedule.width(s);
            let pattern = schedule.pattern(s);
            let rows = pattern.len();
            let size = width + rows;
            for (local, global) in (start..end).chain(pattern.iter().copied()).enumerate() {
                position[global] = local;
            }
            let mut front = vec![0.0f64; size * size];
            for column in start..end {
                let local_column = column - start;
                for entry in lower.col_ptr[column]..lower.col_ptr[column + 1] {
                    let row = position[lower.row_idx[entry]];
                    front[local_column * size + row] += lower.k[entry] - shift * lower.m[entry];
                }
            }
            for &child in schedule.children(s) {
                let update = pending[child]
                    .take()
                    .expect("children are processed before their parent");
                let child_pattern = schedule.pattern(child);
                let child_rows = child_pattern.len();
                for (jj, &gj) in child_pattern.iter().enumerate() {
                    let column = position[gj];
                    let source = &update[jj * child_rows..(jj + 1) * child_rows];
                    let target = &mut front[column * size..(column + 1) * size];
                    for ii in jj..child_rows {
                        target[position[child_pattern[ii]]] += source[ii];
                    }
                }
            }
            let matrix = MatMut::from_column_major_slice_mut(&mut front, size, size);
            let (block, _, coupling, trailing) = matrix.split_at_mut(width, width);
            let mut block = block;
            let mut subdiag = vec![0.0f64; width];
            let mut perm = vec![0usize; width];
            let mut perm_inv = vec![0usize; width];
            {
                let mut buffer = MemBuffer::try_new(
                    bunch_kaufman::factor::cholesky_in_place_scratch::<usize, f64>(
                        width,
                        parallel,
                        Default::default(),
                    ),
                )
                .map_err(|_| SolverError::OutOfMemory {
                    bytes: (width as u64) * DENSE_BLOCK_COLUMNS * SCALAR,
                })?;
                let (_, permutation) = bunch_kaufman::factor::cholesky_in_place(
                    block.rb_mut(),
                    faer::diag::DiagMut::from_slice_mut(&mut subdiag),
                    &mut perm,
                    &mut perm_inv,
                    parallel,
                    MemStack::new(&mut buffer),
                    Default::default(),
                );
                let diagonal: Vec<f64> = (0..width).map(|j| block[(j, j)]).collect();
                classify_block_diagonal(&diagonal, &subdiag, &mut inertia).map_err(|pivot| {
                    SolverError::Singular {
                        pivot: start + pivot,
                    }
                })?;
                if rows > 0 {
                    // Z = F₁₁⁻¹ F₂₁ᵀ, then F₂₂ ← F₂₂ − F₂₁ Z (lower triangle).
                    let mut z = coupling.rb().transpose().to_owned();
                    let mut solve = MemBuffer::try_new(
                        bunch_kaufman::solve::solve_in_place_scratch::<usize, f64>(
                            width, rows, parallel,
                        ),
                    )
                    .map_err(|_| SolverError::OutOfMemory {
                        bytes: (width as u64) * (rows as u64) * SCALAR,
                    })?;
                    bunch_kaufman::solve::solve_in_place_with_conj(
                        block.rb(),
                        block.rb().diagonal(),
                        faer::diag::DiagRef::from_slice(&subdiag),
                        Conj::No,
                        permutation,
                        z.as_mut(),
                        parallel,
                        MemStack::new(&mut solve),
                    );
                    drop(solve);
                    faer::linalg::matmul::triangular::matmul_with_conj(
                        trailing,
                        BlockStructure::TriangularLower,
                        Accum::Add,
                        coupling.rb(),
                        BlockStructure::Rectangular,
                        Conj::No,
                        z.as_ref(),
                        BlockStructure::Rectangular,
                        Conj::No,
                        -1.0,
                        parallel,
                    );
                }
            }
            if rows > 0 {
                let mut update = vec![0.0f64; rows * rows];
                for j in 0..rows {
                    let source = &front[(width + j) * size + width..(width + j + 1) * size];
                    update[j * rows + j..(j + 1) * rows].copy_from_slice(&source[j..]);
                }
                pending[s] = Some(update);
            }
        }
        Ok(inertia)
    }
}

/// Adds the signs of a Bunch–Kaufman block diagonal (`diagonal`, with `subdiag[j] ≠ 0` joining
/// rows `j, j+1` into a 2×2 block) to `inertia`. Returns the offending row of a zero or
/// non-finite pivot.
fn classify_block_diagonal(
    diagonal: &[f64],
    subdiag: &[f64],
    inertia: &mut Inertia,
) -> Result<(), usize> {
    let n = diagonal.len();
    let mut j = 0;
    while j < n {
        if j + 1 < n && subdiag[j] != 0.0 {
            let (a, c, b) = (diagonal[j], diagonal[j + 1], subdiag[j]);
            let determinant = a * c - b * b;
            if !determinant.is_finite() || determinant == 0.0 {
                return Err(j);
            }
            if determinant < 0.0 {
                inertia.positive += 1;
                inertia.negative += 1;
            } else if a + c > 0.0 {
                inertia.positive += 2;
            } else {
                inertia.negative += 2;
            }
            j += 2;
        } else {
            let d = diagonal[j];
            if !d.is_finite() || d == 0.0 {
                return Err(j);
            }
            if d > 0.0 {
                inertia.positive += 1;
            } else {
                inertia.negative += 1;
            }
            j += 1;
        }
    }
    Ok(())
}
