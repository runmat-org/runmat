//! Shared-memory parallel element assembly into a symbolic CSR pattern.
//!
//! One [`ElementDofMap`] describes which global degrees of freedom every element touches. The
//! symbolic pattern is derived once from that map ([`ElementDofMap::symbolic_pattern`]) and
//! reused for every numeric fill (load case, frequency, time step or Newton iteration).
//!
//! Numeric fill ([`assemble_elements`]) runs in bounded element batches:
//! 1. element kernels run in parallel and write dense local outputs into a reused batch buffer;
//! 2. an optional ordered visitor sees every element in canonical element order;
//! 3. rows are partitioned into disjoint ranges and each range scatters the batch in element order.
//!
//! Every global entry therefore receives its contributions in increasing element order (and, inside
//! an element, increasing local row/column order) regardless of the number of threads, the batch
//! size or the row partition. The assembled values are bitwise identical to a serial element loop
//! that accumulates `value[row][column] += local[i][j]` element by element, so results never depend
//! on the admitted thread count.

use std::ops::AddAssign;

use super::{SparsePattern, SparsePatternError};
use crate::planning::resource::LocalExecutionPlan;

/// Marks a local degree of freedom that does not enter the global system (e.g. eliminated).
pub const SKIPPED_DOF: u32 = u32::MAX;

/// Upper bound for one batch of dense element outputs.
const BATCH_BUDGET_BYTES: usize = 96 * 1024 * 1024;
const MIN_BATCH_ELEMENTS: usize = 256;
const MAX_BATCH_ELEMENTS: usize = 1 << 16;
/// Elements per parallel compute task inside one batch.
const COMPUTE_CHUNK_ELEMENTS: usize = 32;
/// Row partitions per thread for scatter load balance (does not affect results).
const PARTITIONS_PER_THREAD: usize = 4;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ElementAssemblyError<E> {
    #[error("element {element} failed: {error:?}")]
    Element { element: usize, error: E },
    #[error("element {element} references degree of freedom {dof} outside 0..{dof_count}")]
    DofOutOfRange {
        element: usize,
        dof: usize,
        dof_count: usize,
    },
    #[error("degree-of-freedom count {0} exceeds the 32-bit assembly index")]
    DofIndexOverflow(usize),
    #[error("assembly entry ({row}, {column}) is outside the symbolic pattern")]
    OutsidePattern { row: usize, column: usize },
    #[error("symbolic pattern differs from the element map: {0}")]
    PatternMismatch(&'static str),
    #[error(transparent)]
    Pattern(#[from] SparsePatternError),
    #[error("assembly storage allocation failed for {0} entries")]
    Allocation(usize),
    #[error("assembly was cancelled")]
    Cancelled,
    #[error("local CPU execution failed: {0}")]
    Execution(String),
}

impl<E> ElementAssemblyError<E> {
    /// Maps the element-kernel error type, keeping every structural assembly failure.
    pub fn map_element<F>(self, map: impl FnOnce(E) -> F) -> ElementAssemblyError<F> {
        match self {
            Self::Element { element, error } => ElementAssemblyError::Element {
                element,
                error: map(error),
            },
            Self::DofOutOfRange {
                element,
                dof,
                dof_count,
            } => ElementAssemblyError::DofOutOfRange {
                element,
                dof,
                dof_count,
            },
            Self::DofIndexOverflow(count) => ElementAssemblyError::DofIndexOverflow(count),
            Self::OutsidePattern { row, column } => {
                ElementAssemblyError::OutsidePattern { row, column }
            }
            Self::PatternMismatch(reason) => ElementAssemblyError::PatternMismatch(reason),
            Self::Pattern(error) => ElementAssemblyError::Pattern(error),
            Self::Allocation(count) => ElementAssemblyError::Allocation(count),
            Self::Cancelled => ElementAssemblyError::Cancelled,
            Self::Execution(reason) => ElementAssemblyError::Execution(reason),
        }
    }
}

/// Runs `work` under the admitted local plan. An already installed admitted pool is the
/// authority (kernels install the pool once for a whole study step); otherwise the plan's
/// thread count is installed for the duration of `work`.
pub fn within_local_plan<R: Send>(
    plan: LocalExecutionPlan,
    work: impl FnOnce() -> R + Send,
) -> Result<R, runmat_cpu_execution::CpuExecutionError> {
    let threads = usize::from(plan.thread_count().get());
    if runmat_cpu_execution::current_threads() > 1 || threads == 1 {
        Ok(work())
    } else {
        runmat_cpu_execution::with_cpu_pool(threads, work)
    }
}

/// `work(0..count)` in canonical order on the installed admitted pool (serial without one).
/// Each item is computed independently, so results never depend on the thread count.
pub fn ordered_map_range<R: Send>(count: usize, work: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    const CHUNK: usize = 1024;
    if count <= CHUNK || runmat_cpu_execution::current_threads() == 1 {
        return (0..count).map(work).collect();
    }
    let ranges = (0..count.div_ceil(CHUNK))
        .map(|chunk| chunk * CHUNK..((chunk + 1) * CHUNK).min(count))
        .collect::<Vec<_>>();
    let chunks = runmat_cpu_execution::ordered_map(&ranges, |range| {
        range.clone().map(&work).collect::<Vec<_>>()
    });
    let mut results = Vec::with_capacity(count);
    for chunk in chunks {
        results.extend(chunk);
    }
    results
}

/// Element -> global DOF incidence, in canonical element order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementDofMap {
    dof_count: usize,
    offsets: Vec<usize>,
    dofs: Vec<u32>,
}

impl ElementDofMap {
    /// Builds the map from flat offsets (`element_count + 1`) and DOFs (`SKIPPED_DOF` allowed).
    pub fn new(
        dof_count: usize,
        offsets: Vec<usize>,
        dofs: Vec<u32>,
    ) -> Result<Self, ElementAssemblyError<std::convert::Infallible>> {
        if dof_count >= SKIPPED_DOF as usize {
            return Err(ElementAssemblyError::DofIndexOverflow(dof_count));
        }
        if offsets.first() != Some(&0)
            || offsets.last() != Some(&dofs.len())
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(ElementAssemblyError::PatternMismatch(
                "element offsets must start at zero, increase and cover all DOFs",
            ));
        }
        for element in 0..offsets.len() - 1 {
            for &dof in &dofs[offsets[element]..offsets[element + 1]] {
                if dof != SKIPPED_DOF && dof as usize >= dof_count {
                    return Err(ElementAssemblyError::DofOutOfRange {
                        element,
                        dof: dof as usize,
                        dof_count,
                    });
                }
            }
        }
        Ok(Self {
            dof_count,
            offsets,
            dofs,
        })
    }

    /// Builds the map in parallel. `element_dofs(e, out)` appends the global DOFs of element `e`
    /// (`None` = skipped local DOF). Element order is preserved.
    pub fn from_elements<E: Send>(
        dof_count: usize,
        element_count: usize,
        plan: LocalExecutionPlan,
        element_dofs: impl Fn(usize, &mut Vec<Option<usize>>) -> Result<(), E> + Sync + Send,
    ) -> Result<Self, ElementAssemblyError<E>> {
        if dof_count >= SKIPPED_DOF as usize {
            return Err(ElementAssemblyError::DofIndexOverflow(dof_count));
        }
        const CHUNK: usize = 4096;
        let chunk_count = element_count.div_ceil(CHUNK);
        let ranges = (0..chunk_count)
            .map(|chunk| chunk * CHUNK..((chunk + 1) * CHUNK).min(element_count))
            .collect::<Vec<_>>();
        let chunks = within_local_plan(plan, || {
            runmat_cpu_execution::ordered_map(&ranges, |range| {
                let mut lengths = Vec::with_capacity(range.len());
                let mut dofs = Vec::new();
                let mut scratch = Vec::new();
                for element in range.clone() {
                    scratch.clear();
                    element_dofs(element, &mut scratch)
                        .map_err(|error| ElementAssemblyError::Element { element, error })?;
                    lengths.push(scratch.len());
                    for dof in &scratch {
                        dofs.push(match *dof {
                            None => SKIPPED_DOF,
                            Some(dof) if dof < dof_count => dof as u32,
                            Some(dof) => {
                                return Err(ElementAssemblyError::DofOutOfRange {
                                    element,
                                    dof,
                                    dof_count,
                                })
                            }
                        });
                    }
                }
                Ok((lengths, dofs))
            })
        })
        .map_err(|error| ElementAssemblyError::Execution(error.to_string()))?;
        let mut offsets = Vec::with_capacity(element_count + 1);
        offsets.push(0);
        let total = chunks
            .iter()
            .map(|chunk| chunk.as_ref().map_or(0, |(_, dofs)| dofs.len()))
            .sum::<usize>();
        let mut dofs = Vec::new();
        dofs.try_reserve_exact(total)
            .map_err(|_| ElementAssemblyError::Allocation(total))?;
        for chunk in chunks {
            let (lengths, chunk_dofs) = chunk?;
            for length in lengths {
                offsets.push(offsets.last().copied().unwrap_or(0) + length);
            }
            dofs.extend_from_slice(&chunk_dofs);
        }
        Ok(Self {
            dof_count,
            offsets,
            dofs,
        })
    }

    pub fn dof_count(&self) -> usize {
        self.dof_count
    }

    pub fn element_count(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn element(&self, element: usize) -> &[u32] {
        &self.dofs[self.offsets[element]..self.offsets[element + 1]]
    }

    /// Resident bytes of the incidence arrays.
    pub fn resident_bytes(&self) -> usize {
        self.offsets.len() * std::mem::size_of::<usize>() + self.dofs.len() * 4
    }

    /// Square symbolic pattern: row `r` holds every DOF sharing an element with `r`.
    /// Rows are built in parallel; the result does not depend on the thread count.
    pub fn symbolic_pattern(
        &self,
        plan: LocalExecutionPlan,
    ) -> Result<SparsePattern, ElementAssemblyError<std::convert::Infallible>> {
        let _phase = crate::progress::phase_timer("assembly.symbolic_pattern");
        let n = self.dof_count;
        if self.element_count() >= u32::MAX as usize {
            return Err(ElementAssemblyError::DofIndexOverflow(self.element_count()));
        }
        // DOF -> incident elements (counting sort keeps element order inside each row).
        let mut incidence_offsets = vec![0_usize; n + 1];
        // An element counts once per distinct DOF (duplicate local DOFs share one incidence).
        let first_use = |dofs: &[u32], local: usize| {
            dofs[local] != SKIPPED_DOF && !dofs[..local].contains(&dofs[local])
        };
        for element in 0..self.element_count() {
            let dofs = self.element(element);
            for local in 0..dofs.len() {
                if first_use(dofs, local) {
                    incidence_offsets[dofs[local] as usize + 1] += 1;
                }
            }
        }
        for row in 0..n {
            incidence_offsets[row + 1] += incidence_offsets[row];
        }
        let incidence_total = incidence_offsets[n];
        let mut incidence = Vec::new();
        incidence
            .try_reserve_exact(incidence_total)
            .map_err(|_| ElementAssemblyError::Allocation(incidence_total))?;
        incidence.resize(incidence_total, 0_u32);
        let mut cursor = incidence_offsets[..n].to_vec();
        for element in 0..self.element_count() {
            let dofs = self.element(element);
            for local in 0..dofs.len() {
                if first_use(dofs, local) {
                    let dof = dofs[local] as usize;
                    incidence[cursor[dof]] = element as u32;
                    cursor[dof] += 1;
                }
            }
        }
        drop(cursor);
        let incidence_rows =
            |row: usize| -> &[u32] { &incidence[incidence_offsets[row]..incidence_offsets[row + 1]] };
        const ROW_BLOCK: usize = 2048;
        let blocks = (0..n.div_ceil(ROW_BLOCK))
            .map(|block| block * ROW_BLOCK..((block + 1) * ROW_BLOCK).min(n))
            .collect::<Vec<_>>();
        let rows = within_local_plan(plan, || {
            runmat_cpu_execution::ordered_map(&blocks, |rows| {
                let mut lengths = Vec::with_capacity(rows.len());
                let mut columns = Vec::new();
                let mut scratch = Vec::<u32>::new();
                for row in rows.clone() {
                    scratch.clear();
                    for &element in incidence_rows(row) {
                        scratch.extend(
                            self.element(element as usize)
                                .iter()
                                .copied()
                                .filter(|&dof| dof != SKIPPED_DOF),
                        );
                    }
                    scratch.sort_unstable();
                    scratch.dedup();
                    lengths.push(scratch.len());
                    columns.extend(scratch.iter().map(|&column| u64::from(column)));
                }
                (lengths, columns)
            })
        })
        .map_err(|error| ElementAssemblyError::Execution(error.to_string()))?;
        drop(incidence);
        let total = rows.iter().map(|(_, columns)| columns.len()).sum::<usize>();
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0_u64);
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(total)
            .map_err(|_| ElementAssemblyError::Allocation(total))?;
        for (lengths, block_columns) in rows {
            for length in lengths {
                offsets.push(offsets.last().copied().unwrap_or(0) + length as u64);
            }
            columns.extend_from_slice(&block_columns);
        }
        Ok(SparsePattern::try_new(n as u64, n as u64, offsets, columns)?)
    }
}

/// Number of dense outputs each element kernel writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ElementOutputShape {
    /// Square `n x n` row-major local matrices, one per assembled operator.
    pub matrices: usize,
    /// Local vectors of length `n`, one per assembled global vector.
    pub vectors: usize,
    /// Per-element scalars reduced in canonical element order.
    pub scalars: usize,
}

impl ElementOutputShape {
    fn values_per_element(self, n: usize) -> usize {
        self.matrices * n * n + self.vectors * n + self.scalars
    }
}

/// Dense local outputs of one element. Buffers are zeroed before the kernel runs.
pub struct ElementOutputs<'a, T> {
    n: usize,
    shape: ElementOutputShape,
    values: &'a mut [T],
}

impl<T> ElementOutputs<'_, T> {
    pub fn size(&self) -> usize {
        self.n
    }
    /// Row-major `n x n` local matrix for operator `index`.
    pub fn matrix(&mut self, index: usize) -> &mut [T] {
        assert!(index < self.shape.matrices);
        let nn = self.n * self.n;
        &mut self.values[index * nn..(index + 1) * nn]
    }
    pub fn vector(&mut self, index: usize) -> &mut [T] {
        assert!(index < self.shape.vectors);
        let start = self.shape.matrices * self.n * self.n + index * self.n;
        &mut self.values[start..start + self.n]
    }
    pub fn scalar(&mut self, index: usize) -> &mut T {
        assert!(index < self.shape.scalars);
        let start = self.shape.matrices * self.n * self.n + self.shape.vectors * self.n;
        &mut self.values[start + index]
    }
}

/// Read-only view of one element's computed outputs for the ordered visitor.
pub struct ElementView<'a, T> {
    pub element: usize,
    pub dofs: &'a [u32],
    n: usize,
    shape: ElementOutputShape,
    values: &'a [T],
}

impl<T> ElementView<'_, T> {
    pub fn matrix(&self, index: usize) -> &[T] {
        let nn = self.n * self.n;
        &self.values[index * nn..(index + 1) * nn]
    }
    pub fn vector(&self, index: usize) -> &[T] {
        let start = self.shape.matrices * self.n * self.n + index * self.n;
        &self.values[start..start + self.n]
    }
    pub fn scalar(&self, index: usize) -> &T {
        let start = self.shape.matrices * self.n * self.n + self.shape.vectors * self.n;
        &self.values[start + index]
    }
}

/// Assembled numeric values on the symbolic pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct AssembledElements<T> {
    /// One value array per matrix, aligned with the pattern's column indices.
    pub matrices: Vec<Vec<T>>,
    pub vectors: Vec<Vec<T>>,
    pub scalars: Vec<T>,
}

/// Visitor that does nothing; pass to [`assemble_elements`] when no ordered capture is needed.
pub fn no_visit<T, E>(_: &ElementView<'_, T>) -> Result<(), E> {
    Ok(())
}

struct ComputeTask<'a, T, E> {
    elements: std::ops::Range<usize>,
    values: &'a mut [T],
    order: &'a mut [u32],
    failure: Option<(usize, E)>,
}

struct ScatterTask<'a, T> {
    rows: std::ops::Range<usize>,
    entry_start: usize,
    matrices: Vec<&'a mut [T]>,
    vectors: Vec<&'a mut [T]>,
    failure: Option<(usize, usize)>,
}

/// Assembles element kernels onto `pattern` (built from `map`). See the module documentation for
/// the determinism contract.
#[allow(clippy::too_many_arguments)]
pub fn assemble_elements<T, E, K, V>(
    map: &ElementDofMap,
    pattern: &SparsePattern,
    shape: ElementOutputShape,
    plan: LocalExecutionPlan,
    kernel: K,
    mut visit: V,
) -> Result<AssembledElements<T>, ElementAssemblyError<E>>
where
    T: Copy + Default + AddAssign + PartialEq + Send + Sync,
    E: Send,
    K: Fn(usize, &[u32], &mut ElementOutputs<'_, T>) -> Result<(), E> + Sync + Send,
    V: FnMut(&ElementView<'_, T>) -> Result<(), E> + Send,
{
    let _phase = crate::progress::phase_timer("assembly.numeric_fill");
    let n_dofs = map.dof_count();
    if pattern.rows() != n_dofs || pattern.columns() != n_dofs {
        return Err(ElementAssemblyError::PatternMismatch(
            "pattern shape differs from the element map",
        ));
    }
    let nnz = pattern.column_indices().len();
    let allocate = |len: usize| -> Result<Vec<T>, ElementAssemblyError<E>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| ElementAssemblyError::Allocation(len))?;
        values.resize(len, T::default());
        Ok(values)
    };
    let mut matrices = (0..shape.matrices)
        .map(|_| allocate(nnz))
        .collect::<Result<Vec<_>, _>>()?;
    let mut vectors = (0..shape.vectors)
        .map(|_| allocate(n_dofs))
        .collect::<Result<Vec<_>, _>>()?;
    let mut scalars = vec![T::default(); shape.scalars];
    let element_count = map.element_count();
    if element_count == 0 {
        return Ok(AssembledElements {
            matrices,
            vectors,
            scalars,
        });
    }
    let cancellation = crate::progress::cancellation_token();
    let row_offsets = pattern.row_offsets();
    let columns = pattern.column_indices();

    within_local_plan(plan, || {
        let threads = runmat_cpu_execution::current_threads();
        // Row partitions balanced by stored entries (results are partition independent).
        let partitions = threads * PARTITIONS_PER_THREAD;
        let mut boundaries = Vec::with_capacity(partitions + 1);
        boundaries.push(0_usize);
        for part in 1..partitions {
            let target = (nnz as u128 * part as u128 / partitions as u128) as u64;
            let row = row_offsets.partition_point(|&offset| offset < target).min(n_dofs);
            let row = row.max(*boundaries.last().unwrap());
            boundaries.push(row);
        }
        boundaries.push(n_dofs);
        boundaries.dedup();
        let part_of = |row: usize| boundaries.partition_point(|&start| start <= row) - 1;
        let part_count = boundaries.len() - 1;

        let max_values = (0..element_count)
            .map(|element| shape.values_per_element(map.element(element).len()))
            .max()
            .unwrap_or(0)
            .max(1);
        let batch_elements = (BATCH_BUDGET_BYTES / (max_values * std::mem::size_of::<T>()))
            .clamp(MIN_BATCH_ELEMENTS, MAX_BATCH_ELEMENTS);
        let mut buffer: Vec<T> = Vec::new();
        let mut order: Vec<u32> = Vec::new();
        let mut value_offsets: Vec<usize> = Vec::with_capacity(batch_elements + 1);
        let mut lists: Vec<Vec<(u32, u32)>> = vec![Vec::new(); part_count];

        let mut batch_start = 0;
        while batch_start < element_count {
            if cancellation.is_cancelled() {
                return Err(ElementAssemblyError::Cancelled);
            }
            let batch_end = (batch_start + batch_elements).min(element_count);
            value_offsets.clear();
            value_offsets.push(0);
            let mut order_len = 0;
            for element in batch_start..batch_end {
                let n = map.element(element).len();
                value_offsets.push(value_offsets.last().unwrap() + shape.values_per_element(n));
                order_len += n;
            }
            let total = *value_offsets.last().unwrap();
            buffer.clear();
            buffer
                .try_reserve(total)
                .map_err(|_| ElementAssemblyError::Allocation(total))?;
            buffer.resize(total, T::default());
            order.clear();
            order.resize(order_len, 0);

            // Phase 1: element kernels in parallel.
            let mut tasks = Vec::with_capacity((batch_end - batch_start).div_ceil(COMPUTE_CHUNK_ELEMENTS));
            {
                let mut values_rest = buffer.as_mut_slice();
                let mut order_rest = order.as_mut_slice();
                let mut start = batch_start;
                while start < batch_end {
                    let end = (start + COMPUTE_CHUNK_ELEMENTS).min(batch_end);
                    let len = value_offsets[end - batch_start] - value_offsets[start - batch_start];
                    let order_len = (start..end).map(|e| map.element(e).len()).sum::<usize>();
                    let (values, rest) = values_rest.split_at_mut(len);
                    values_rest = rest;
                    let (order_slice, rest) = order_rest.split_at_mut(order_len);
                    order_rest = rest;
                    tasks.push(ComputeTask::<T, E> {
                        elements: start..end,
                        values,
                        order: order_slice,
                        failure: None,
                    });
                    start = end;
                }
            }
            runmat_cpu_execution::for_each_mut(&mut tasks, |task| {
                let mut value_cursor = 0;
                let mut order_cursor = 0;
                for element in task.elements.clone() {
                    let dofs = map.element(element);
                    let n = dofs.len();
                    let len = shape.values_per_element(n);
                    let mut outputs = ElementOutputs {
                        n,
                        shape,
                        values: &mut task.values[value_cursor..value_cursor + len],
                    };
                    if let Err(error) = kernel(element, dofs, &mut outputs) {
                        task.failure = Some((element, error));
                        return;
                    }
                    let local_order = &mut task.order[order_cursor..order_cursor + n];
                    for (index, slot) in local_order.iter_mut().enumerate() {
                        *slot = index as u32;
                    }
                    // Stable: duplicate DOFs keep ascending local order.
                    local_order.sort_by_key(|&local| dofs[local as usize]);
                    value_cursor += len;
                    order_cursor += n;
                }
            });
            for task in &mut tasks {
                if let Some((element, error)) = task.failure.take() {
                    return Err(ElementAssemblyError::Element { element, error });
                }
            }
            drop(tasks);

            // Phase 2: ordered visitor and canonical scalar reduction.
            let mut order_starts = Vec::with_capacity(batch_end - batch_start + 1);
            order_starts.push(0_usize);
            for element in batch_start..batch_end {
                let dofs = map.element(element);
                let n = dofs.len();
                let local = element - batch_start;
                let view = ElementView {
                    element,
                    dofs,
                    n,
                    shape,
                    values: &buffer[value_offsets[local]..value_offsets[local + 1]],
                };
                visit(&view).map_err(|error| ElementAssemblyError::Element { element, error })?;
                for (index, scalar) in scalars.iter_mut().enumerate() {
                    *scalar += *view.scalar(index);
                }
                order_starts.push(order_starts.last().unwrap() + n);
            }

            // Phase 3: scatter by disjoint row partitions in element order.
            for list in &mut lists {
                list.clear();
            }
            for element in batch_start..batch_end {
                for (local_row, &dof) in map.element(element).iter().enumerate() {
                    if dof != SKIPPED_DOF {
                        lists[part_of(dof as usize)]
                            .push(((element - batch_start) as u32, local_row as u32));
                    }
                }
            }
            let mut scatter = Vec::with_capacity(part_count);
            {
                let mut matrix_rest = matrices
                    .iter_mut()
                    .map(|values| values.as_mut_slice())
                    .collect::<Vec<_>>();
                let mut vector_rest = vectors
                    .iter_mut()
                    .map(|values| values.as_mut_slice())
                    .collect::<Vec<_>>();
                for part in 0..part_count {
                    let rows = boundaries[part]..boundaries[part + 1];
                    let entry_start = row_offsets[rows.start] as usize;
                    let entry_len = row_offsets[rows.end] as usize - entry_start;
                    let mut part_matrices = Vec::with_capacity(matrix_rest.len());
                    for rest in &mut matrix_rest {
                        let (head, tail) = std::mem::take(rest).split_at_mut(entry_len);
                        *rest = tail;
                        part_matrices.push(head);
                    }
                    let mut part_vectors = Vec::with_capacity(vector_rest.len());
                    for rest in &mut vector_rest {
                        let (head, tail) = std::mem::take(rest).split_at_mut(rows.len());
                        *rest = tail;
                        part_vectors.push(head);
                    }
                    scatter.push((
                        ScatterTask {
                            rows,
                            entry_start,
                            matrices: part_matrices,
                            vectors: part_vectors,
                            failure: None,
                        },
                        &lists[part],
                    ));
                }
            }
            let buffer_ref = &buffer;
            let order_ref = &order;
            let value_offsets_ref = &value_offsets;
            let order_starts_ref = &order_starts;
            runmat_cpu_execution::for_each_mut(&mut scatter, |(task, list)| {
                for &(local_element, local_row) in list.iter() {
                    let local_element = local_element as usize;
                    let local_row = local_row as usize;
                    let dofs = map.element(batch_start + local_element);
                    let n = dofs.len();
                    let row = dofs[local_row] as usize;
                    let values = &buffer_ref
                        [value_offsets_ref[local_element]..value_offsets_ref[local_element + 1]];
                    let sorted = &order_ref
                        [order_starts_ref[local_element]..order_starts_ref[local_element + 1]];
                    let row_start = row_offsets[row] as usize;
                    let row_columns = &columns[row_start..row_offsets[row + 1] as usize];
                    let local_entry = row_start - task.entry_start;
                    if shape.matrices > 0 {
                        let mut cursor = 0;
                        for &local_column in sorted {
                            let column = dofs[local_column as usize];
                            if column == SKIPPED_DOF {
                                break;
                            }
                            while cursor < row_columns.len()
                                && row_columns[cursor] < u64::from(column)
                            {
                                cursor += 1;
                            }
                            let local = local_row * n + local_column as usize;
                            if cursor == row_columns.len()
                                || row_columns[cursor] != u64::from(column)
                            {
                                // Exact-zero contributions carry no support (as in
                                // `RealSparseAssembler::add`); anything else fails closed.
                                if (0..task.matrices.len())
                                    .all(|matrix| values[matrix * n * n + local] == T::default())
                                {
                                    continue;
                                }
                                task.failure = Some((row, column as usize));
                                return;
                            }
                            for (matrix, target) in task.matrices.iter_mut().enumerate() {
                                target[local_entry + cursor] += values[matrix * n * n + local];
                            }
                        }
                    }
                    let vector_base = shape.matrices * n * n;
                    for (vector, target) in task.vectors.iter_mut().enumerate() {
                        target[row - task.rows.start] += values[vector_base + vector * n + local_row];
                    }
                }
            });
            for (task, _) in &scatter {
                if let Some((row, column)) = task.failure {
                    return Err(ElementAssemblyError::OutsidePattern { row, column });
                }
            }
            batch_start = batch_end;
        }
        Ok(())
    })
    .map_err(|error| ElementAssemblyError::Execution(error.to_string()))??;
    Ok(AssembledElements {
        matrices,
        vectors,
        scalars,
    })
}

#[cfg(test)]
mod tests;
