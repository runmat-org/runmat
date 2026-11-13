//! Level scheduling for sparse triangular solves.
//!
//! Row `i` of a triangular solve depends on the rows named by its off-diagonal entries. Rows
//! whose dependencies all lie in earlier levels are independent and may be solved concurrently.
//! Each row's dot product is still accumulated in its stored entry order, so a scheduled solve is
//! bitwise identical to the sequential one for every thread count.

use crate::par;
use crate::scalar::Scalar;

/// Rows grouped by dependency level (`rows[level_ptr[l]..level_ptr[l + 1]]` is level `l`).
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct LevelSchedule {
    rows: Vec<usize>,
    level_ptr: Vec<usize>,
    /// Whether most rows lie in levels wide enough to fan out. Otherwise the solve keeps the
    /// natural row order, which is a valid order with far better locality.
    wide: bool,
}

/// Minimum rows in a level before it fans out over the pool. A level is one fork-join; narrow
/// levels (structured orderings give thousands of short rows per level) are faster serially.
const PARALLEL_LEVEL_ROWS: usize = 32 * LEVEL_ROWS_PER_TASK;
/// Rows per task inside a wide level.
const LEVEL_ROWS_PER_TASK: usize = 1024;

impl LevelSchedule {
    /// `dependencies(i)` yields the rows row `i` reads. For a forward (lower) solve they are all
    /// `< i`; for a backward (upper) solve all `> i`.
    pub(crate) fn build<'a>(
        n: usize,
        forward: bool,
        dependencies: impl Fn(usize) -> &'a [usize],
    ) -> Self {
        let mut level = vec![0usize; n];
        let mut depth = 0usize;
        let mut visit = |i: usize, level: &mut Vec<usize>| {
            let mut l = 0usize;
            for &j in dependencies(i) {
                l = l.max(level[j] + 1);
            }
            level[i] = l;
            depth = depth.max(l + 1);
        };
        if forward {
            for i in 0..n {
                visit(i, &mut level);
            }
        } else {
            for i in (0..n).rev() {
                visit(i, &mut level);
            }
        }
        let mut level_ptr = vec![0usize; depth + 1];
        for &l in &level {
            level_ptr[l + 1] += 1;
        }
        for l in 0..depth {
            level_ptr[l + 1] += level_ptr[l];
        }
        let mut next = level_ptr.clone();
        let mut rows = vec![0usize; n];
        for (i, &l) in level.iter().enumerate() {
            rows[next[l]] = i;
            next[l] += 1;
        }
        let wide_rows: usize = level_ptr
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|&rows| rows >= PARALLEL_LEVEL_ROWS)
            .sum();
        Self {
            rows,
            level_ptr,
            wide: 2 * wide_rows >= n.max(1),
        }
    }

    /// Whether the scheduled (level-parallel) solve is worth using.
    pub(crate) fn is_wide(&self) -> bool {
        self.wide
    }

    pub(crate) fn levels(&self) -> usize {
        self.level_ptr.len().saturating_sub(1)
    }

    /// Solves level by level: `row(i, z)` returns the new `z[i]`, reading only entries of `z`
    /// from earlier levels.
    pub(crate) fn solve<T: Scalar>(
        &self,
        z: &mut [T],
        row: impl Fn(usize, Earlier<'_, T>) -> T + Sync,
    ) {
        let shared = SharedSlice(z.as_mut_ptr(), z.len());
        for level in 0..self.levels() {
            let rows = &self.rows[self.level_ptr[level]..self.level_ptr[level + 1]];
            let kernel = |&i: &usize| {
                let (ptr, len) = shared.parts();
                let value = row(
                    i,
                    Earlier {
                        ptr,
                        len,
                        lifetime: std::marker::PhantomData,
                    },
                );
                // SAFETY: rows of one level are distinct and each writes only its own entry; no
                // row of this level reads it (its readers lie in later levels).
                unsafe { ptr.add(i).write(value) };
            };
            if rows.len() >= PARALLEL_LEVEL_ROWS && par::threads() > 1 {
                let tasks: Vec<&[usize]> = rows.chunks(LEVEL_ROWS_PER_TASK).collect();
                runmat_cpu_execution::ordered_map(&tasks, |task| task.iter().for_each(&kernel));
            } else {
                rows.iter().for_each(kernel);
            }
        }
    }
}

/// Read access to the entries of a level-scheduled solve that earlier levels completed.
#[derive(Clone, Copy)]
pub(crate) struct Earlier<'a, T> {
    ptr: *mut T,
    len: usize,
    lifetime: std::marker::PhantomData<&'a [T]>,
}

impl<T: Copy> Earlier<'_, T> {
    #[inline(always)]
    pub(crate) fn get(&self, index: usize) -> T {
        assert!(index < self.len);
        // SAFETY: in bounds; the entry belongs to an earlier level, so no thread writes it now.
        unsafe { self.ptr.add(index).read() }
    }
}

#[derive(Clone, Copy)]
struct SharedSlice<T>(*mut T, usize);

impl<T> SharedSlice<T> {
    /// Accessed through a method so closures capture the whole (Sync) wrapper.
    fn parts(&self) -> (*mut T, usize) {
        (self.0, self.1)
    }
}
// SAFETY: access is partitioned by level as documented on `LevelSchedule::solve`.
unsafe impl<T: Send> Send for SharedSlice<T> {}
unsafe impl<T: Send> Sync for SharedSlice<T> {}

#[cfg(test)]
mod tests {
    use crate::operator::Preconditioner;
    use crate::par;
    use crate::precond::{Ilu0, Ilut, IlutOptions};
    use crate::testutil::poisson3d;

    #[test]
    fn wide_levels_run_in_parallel_and_match_the_sequential_solve_bitwise() {
        // 40 000 independent tridiagonal chains of length 4: four levels of 40 000 rows each.
        let (chains, length) = (40_000, 4);
        let n = chains * length;
        let mut t = crate::TripletBuilder::new(n, n);
        for chain in 0..chains {
            for k in 0..length {
                let i = chain * length + k;
                t.push(i, i, 4.0 + (i % 7) as f64 * 0.1);
                if k > 0 {
                    t.push(i, i - 1, -1.0 - (i % 3) as f64 * 0.1);
                    t.push(i - 1, i, -0.5);
                }
            }
        }
        let a: crate::CsrMatrix<f64> = t.build();
        let ilu = Ilu0::new(a.as_ref(), false).unwrap();
        assert_eq!(ilu.lower.levels(), length);
        let r = par::seeded_vector::<f64>(n, 3);
        let mut serial = vec![0.0; n];
        ilu.apply(&r, &mut serial).unwrap();
        let mut threaded = vec![0.0; n];
        par::with_threads(4, || ilu.apply(&r, &mut threaded).unwrap());
        assert!(serial
            .iter()
            .zip(&threaded)
            .all(|(p, q)| p.to_bits() == q.to_bits()));
    }

    #[test]
    fn scheduled_triangular_solves_match_the_sequential_solve_bitwise() {
        let a = poisson3d::<f64>(24, 22, 20).with_shifted_diagonal(0.3);
        let n = a.nrows();
        let r = par::seeded_vector::<f64>(n, 17);
        let ilu = Ilu0::new(a.as_ref(), false).unwrap();
        assert!(ilu.lower.levels() > 1 && ilu.lower.levels() < n);
        let ilut = Ilut::new(a.as_ref(), IlutOptions::default()).unwrap();
        for preconditioner in [&ilu as &dyn Preconditioner<f64>, &ilut] {
            let mut serial = vec![0.0; n];
            preconditioner.apply(&r, &mut serial).unwrap();
            for threads in [2, 4] {
                let mut threaded = vec![0.0; n];
                par::with_threads(threads, || preconditioner.apply(&r, &mut threaded).unwrap());
                assert!(serial
                    .iter()
                    .zip(&threaded)
                    .all(|(p, q)| p.to_bits() == q.to_bits()));
            }
        }
    }
}
