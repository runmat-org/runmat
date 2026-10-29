//! Row-distributed solves over a value-free communicator.
//!
//! Each rank owns a contiguous range of global rows. Rank boundaries are multiples of
//! [`crate::par::REDUCTION_BLOCK`], so every global reduction block lies on one rank: a rank
//! computes the block partials of its rows, the communicator gathers all partials in rank (and
//! therefore global block) order, and every rank folds them identically. Reductions are thus
//! bitwise equal to the single-rank solve for any rank and thread layout, and the replicated
//! recurrence scalars (Krylov coefficients, stopping decisions) agree on every rank.
//!
//! The crate stays network-free: transports (the runner's rank collectives and channels, or the
//! in-memory [`LoopbackCommunicator`] used in tests) implement [`Communicator`].

use crate::csr::CsrRef;
use crate::error::{SolverError, SolverResult};
use crate::operator::LinearOperator;
use crate::par::REDUCTION_BLOCK;
use crate::scalar::Scalar;

/// Value-free collective transport between the ranks of one solve.
pub trait Communicator: Sync {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    /// Every rank's `local`, concatenated in rank order (identical on all ranks).
    fn all_gather(&self, local: &[f64]) -> SolverResult<Vec<Vec<f64>>>;
    /// Sends each `(destination, values)` and returns what every source rank sent to this one
    /// (one entry per source rank, in rank order; empty when nothing was sent).
    fn exchange(&self, outgoing: Vec<(usize, Vec<f64>)>) -> SolverResult<Vec<Vec<f64>>>;
}

/// Contiguous row ownership: rank `r` owns rows `starts[r]..starts[r + 1]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowPartition {
    starts: Vec<usize>,
}

impl RowPartition {
    /// `starts` begins at 0, ends at the global row count, is nondecreasing, and every interior
    /// boundary is a multiple of the reduction block (so reductions stay layout-invariant).
    pub fn new(starts: Vec<usize>) -> SolverResult<Self> {
        let valid = starts.len() >= 2
            && starts[0] == 0
            && starts.windows(2).all(|w| w[0] <= w[1])
            && starts[1..starts.len() - 1]
                .iter()
                .all(|s| s % REDUCTION_BLOCK == 0);
        if !valid {
            return Err(SolverError::Dimension(
                "row partition must be nondecreasing with block-aligned rank boundaries".into(),
            ));
        }
        Ok(Self { starts })
    }

    /// `ranks` nearly equal, block-aligned parts of `n` rows.
    pub fn balanced(n: usize, ranks: usize) -> Self {
        let blocks = n.div_ceil(REDUCTION_BLOCK);
        let mut starts = (0..ranks)
            .map(|r| (blocks * r / ranks * REDUCTION_BLOCK).min(n))
            .collect::<Vec<_>>();
        starts.push(n);
        Self { starts }
    }

    pub fn ranks(&self) -> usize {
        self.starts.len() - 1
    }
    pub fn rows(&self, rank: usize) -> std::ops::Range<usize> {
        self.starts[rank]..self.starts[rank + 1]
    }
    pub fn global_rows(&self) -> usize {
        *self.starts.last().unwrap()
    }
    fn owner(&self, row: usize) -> usize {
        self.starts.partition_point(|&s| s <= row) - 1
    }
}

/// The owned rows of a row-distributed CSR matrix with its halo (ghost) exchange plan.
pub struct DistributedCsr<'c, T> {
    comm: &'c dyn Communicator,
    /// Owned rows with local column numbering (owned columns, then ghosts); each row keeps the
    /// global matrix's entry order, so its columns need not be sorted.
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
    extended: usize,
    owned: usize,
    /// Local indices to send to each neighbor (`sends[rank]`).
    sends: Vec<Vec<usize>>,
    /// Number of ghost values received from each rank, in ghost order.
    receives: Vec<usize>,
}

impl<'c, T: Scalar> DistributedCsr<'c, T> {
    /// Collective: every rank passes its owned rows (`partition.rows(rank)` of the global matrix,
    /// with global column indices). Ghost columns are numbered after the owned ones, grouped by
    /// owner rank in increasing global order, so each row keeps its entry order.
    pub fn new(
        comm: &'c dyn Communicator,
        partition: &RowPartition,
        rows: CsrRef<'_, T>,
    ) -> SolverResult<Self> {
        let rank = comm.rank();
        let owned_range = partition.rows(rank);
        let owned = owned_range.len();
        if rows.nrows() != owned
            || rows.ncols() != partition.global_rows()
            || partition.ranks() != comm.size()
        {
            return Err(SolverError::Dimension(
                "distributed rows do not match the partition".into(),
            ));
        }
        // Ghost columns needed from each owner rank (sorted, unique).
        let mut needed: Vec<Vec<usize>> = vec![Vec::new(); comm.size()];
        for &col in rows.col_idx() {
            if !owned_range.contains(&col) {
                needed[partition.owner(col)].push(col);
            }
        }
        for list in &mut needed {
            list.sort_unstable();
            list.dedup();
        }
        let mut ghost_index = std::collections::HashMap::new();
        let mut next = owned;
        for list in &needed {
            for &col in list {
                ghost_index.insert(col, next);
                next += 1;
            }
        }
        // Tell each owner which of its rows we need; learn which of ours others need.
        let requests = needed
            .iter()
            .enumerate()
            .filter(|(r, list)| *r != rank && !list.is_empty())
            .map(|(r, list)| (r, list.iter().map(|&c| c as f64).collect()))
            .collect();
        let asked = comm.exchange(requests)?;
        let sends = asked
            .iter()
            .map(|list| {
                list.iter()
                    .map(|&g| g as usize - owned_range.start)
                    .collect()
            })
            .collect();
        let receives = needed.iter().map(Vec::len).collect();
        let col_idx = rows
            .col_idx()
            .iter()
            .map(|&col| {
                if owned_range.contains(&col) {
                    col - owned_range.start
                } else {
                    ghost_index[&col]
                }
            })
            .collect();
        Ok(Self {
            comm,
            row_ptr: rows.row_ptr().to_vec(),
            col_idx,
            values: rows.values().to_vec(),
            extended: next,
            owned,
            sends,
            receives,
        })
    }
}

impl<T: Scalar> LinearOperator<T> for DistributedCsr<'_, T> {
    fn nrows(&self) -> usize {
        self.owned
    }
    fn ncols(&self) -> usize {
        self.owned
    }
    /// `y = A x` on the owned rows: exchanges the halo, then the local row kernel (each row in
    /// its stored entry order, as in the single-rank product).
    fn apply(&self, x: &[T], y: &mut [T]) -> SolverResult<()> {
        let encode = |values: &mut Vec<f64>, value: T| {
            values.push(value.re());
            if T::IS_COMPLEX {
                values.push(value.im());
            }
        };
        let outgoing = self
            .sends
            .iter()
            .enumerate()
            .filter(|(_, list)| !list.is_empty())
            .map(|(r, list)| {
                let mut values = Vec::with_capacity(list.len() * 2);
                list.iter().for_each(|&i| encode(&mut values, x[i]));
                (r, values)
            })
            .collect();
        let incoming = self.comm.exchange(outgoing)?;
        let mut extended = Vec::with_capacity(self.extended);
        extended.extend_from_slice(x);
        let width = if T::IS_COMPLEX { 2 } else { 1 };
        for (values, &count) in incoming.iter().zip(&self.receives) {
            if values.len() != count * width {
                return Err(SolverError::Operator("halo exchange length".into()));
            }
            extended.extend(
                values
                    .chunks(width)
                    .map(|v| T::from_parts(v[0], if width == 2 { v[1] } else { 0.0 })),
            );
        }
        // The single-rank row kernel: each row summed sequentially in its stored entry order
        // (rows of one rank may be split over the admitted pool; rows are independent).
        crate::par::for_each_chunk_mut(y, |start, chunk| {
            for (offset, value) in chunk.iter_mut().enumerate() {
                let row = start + offset;
                let mut sum = T::zero();
                for entry in self.row_ptr[row]..self.row_ptr[row + 1] {
                    sum += self.values[entry] * extended[self.col_idx[entry]];
                }
                *value = sum;
            }
        });
        Ok(())
    }
}

/// In-memory communicator for `size` ranks running on threads of one process (tests and
/// single-machine development).
pub struct LoopbackCommunicator {
    rank: usize,
    shared: std::sync::Arc<LoopbackShared>,
}

struct LoopbackShared {
    size: usize,
    barrier: std::sync::Barrier,
    gather: std::sync::Mutex<Vec<Vec<f64>>>,
    mailboxes: std::sync::Mutex<Vec<Vec<Vec<f64>>>>,
}

impl LoopbackCommunicator {
    /// One communicator per rank.
    pub fn group(size: usize) -> Vec<Self> {
        let shared = std::sync::Arc::new(LoopbackShared {
            size,
            barrier: std::sync::Barrier::new(size),
            gather: std::sync::Mutex::new(vec![Vec::new(); size]),
            mailboxes: std::sync::Mutex::new(vec![vec![Vec::new(); size]; size]),
        });
        (0..size)
            .map(|rank| Self {
                rank,
                shared: shared.clone(),
            })
            .collect()
    }
}

impl Communicator for LoopbackCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }
    fn size(&self) -> usize {
        self.shared.size
    }
    fn all_gather(&self, local: &[f64]) -> SolverResult<Vec<Vec<f64>>> {
        self.shared.gather.lock().unwrap()[self.rank] = local.to_vec();
        self.shared.barrier.wait();
        let all = self.shared.gather.lock().unwrap().clone();
        self.shared.barrier.wait();
        Ok(all)
    }
    fn exchange(&self, outgoing: Vec<(usize, Vec<f64>)>) -> SolverResult<Vec<Vec<f64>>> {
        {
            let mut boxes = self.shared.mailboxes.lock().unwrap();
            for (destination, values) in outgoing {
                boxes[destination][self.rank] = values;
            }
        }
        self.shared.barrier.wait();
        let received = std::mem::replace(
            &mut self.shared.mailboxes.lock().unwrap()[self.rank],
            vec![Vec::new(); self.shared.size],
        );
        self.shared.barrier.wait();
        Ok(received)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::SolveControl;
    use crate::krylov::{
        gmres, gmres_distributed, pcg, pcg_distributed, GmresOptions, GmresWorkspace, KrylovOptions,
    };
    use crate::operator::Preconditioner;
    use crate::testutil::poisson3d;
    use crate::{par, CsrMatrix};

    struct Diagonal(Vec<f64>);
    impl Preconditioner<f64> for Diagonal {
        fn apply(&self, r: &[f64], z: &mut [f64]) -> SolverResult<()> {
            for ((z, r), d) in z.iter_mut().zip(r).zip(&self.0) {
                *z = r / d;
            }
            Ok(())
        }
    }

    /// Owned rows `range` of `a` with global column indices.
    fn rows(
        a: &CsrMatrix<f64>,
        range: std::ops::Range<usize>,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let a = a.as_ref();
        let (start, end) = (a.row_ptr()[range.start], a.row_ptr()[range.end]);
        let row_ptr = a.row_ptr()[range.start..=range.end]
            .iter()
            .map(|p| p - start)
            .collect();
        (
            row_ptr,
            a.col_idx()[start..end].to_vec(),
            a.values()[start..end].to_vec(),
        )
    }

    /// Solves on `ranks` loopback ranks; returns the concatenated solution and iteration count.
    fn distributed(
        a: &CsrMatrix<f64>,
        b: &[f64],
        ranks: usize,
        use_gmres: bool,
    ) -> (Vec<f64>, usize) {
        let n = a.nrows();
        let partition = RowPartition::balanced(n, ranks);
        let diagonal = a.as_ref().diagonal();
        let comms = LoopbackCommunicator::group(ranks);
        let results: Vec<(Vec<f64>, usize)> = std::thread::scope(|scope| {
            let handles: Vec<_> = comms
                .iter()
                .map(|comm| {
                    let partition = &partition;
                    let diagonal = &diagonal;
                    scope.spawn(move || {
                        let range = partition.rows(comm.rank());
                        let (row_ptr, col_idx, values) = rows(a, range.clone());
                        let local =
                            CsrRef::new_unchecked(range.len(), n, &row_ptr, &col_idx, &values);
                        let operator = DistributedCsr::new(comm, partition, local).unwrap();
                        let jacobi = Diagonal(diagonal[range.clone()].to_vec());
                        let mut x = vec![0.0; range.len()];
                        let options = KrylovOptions {
                            rtol: 1e-10,
                            max_iterations: 400,
                            ..Default::default()
                        };
                        let report = if use_gmres {
                            gmres_distributed(
                                comm,
                                &mut GmresWorkspace::default(),
                                &operator,
                                Some(&jacobi),
                                &b[range.clone()],
                                &mut x,
                                &options,
                                &GmresOptions {
                                    restart: 20,
                                    ..Default::default()
                                },
                                SolveControl::NONE,
                            )
                        } else {
                            pcg_distributed(
                                comm,
                                &operator,
                                Some(&jacobi),
                                &b[range.clone()],
                                &mut x,
                                &options,
                                SolveControl::NONE,
                            )
                        }
                        .unwrap();
                        assert!(report.converged(), "{report:?}");
                        (x, report.iterations)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let iterations = results[0].1;
        assert!(results.iter().all(|r| r.1 == iterations));
        (results.into_iter().flat_map(|r| r.0).collect(), iterations)
    }

    #[test]
    fn distributed_solves_are_bitwise_the_single_rank_solve() {
        let a = poisson3d::<f64>(32, 32, 32);
        let n = a.nrows();
        let b = par::seeded_vector::<f64>(n, 21);
        let jacobi = Diagonal(a.as_ref().diagonal());
        let options = KrylovOptions {
            rtol: 1e-10,
            max_iterations: 400,
            ..Default::default()
        };
        for use_gmres in [false, true] {
            let mut reference = vec![0.0; n];
            let report = if use_gmres {
                gmres(
                    &a,
                    Some(&jacobi),
                    &b,
                    &mut reference,
                    &options,
                    &GmresOptions {
                        restart: 20,
                        ..Default::default()
                    },
                    SolveControl::NONE,
                )
            } else {
                pcg(
                    &a,
                    Some(&jacobi),
                    &b,
                    &mut reference,
                    &options,
                    SolveControl::NONE,
                )
            }
            .unwrap();
            for ranks in [1, 2, 4] {
                let (x, iterations) = distributed(&a, &b, ranks, use_gmres);
                assert_eq!(
                    iterations, report.iterations,
                    "gmres={use_gmres} ranks={ranks}"
                );
                assert!(
                    x.iter()
                        .zip(&reference)
                        .all(|(p, q)| p.to_bits() == q.to_bits()),
                    "gmres={use_gmres} ranks={ranks}"
                );
            }
        }
    }

    #[test]
    fn partitions_keep_reduction_blocks_on_one_rank() {
        assert!(RowPartition::new(vec![0, 100, 200]).is_err());
        let p = RowPartition::balanced(10 * REDUCTION_BLOCK + 7, 3);
        assert_eq!(p.ranks(), 3);
        assert_eq!(p.global_rows(), 10 * REDUCTION_BLOCK + 7);
        assert!(RowPartition::new(
            (0..=3)
                .map(|r| p.rows(r.min(2)).start)
                .chain([p.global_rows()])
                .take(4)
                .collect()
        )
        .is_ok());
    }
}
