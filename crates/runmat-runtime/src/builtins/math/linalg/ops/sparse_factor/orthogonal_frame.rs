//! Original-owner Gaussian Householder frame, without a retained dense Q.
use super::{SparseCompleteQrError as Error, SparseQrDecomposition};
use crate::{
    context::{host_allocation::HostAllocationError, RuntimeContext},
    execution::CpuCooperationBudget,
    BuiltinResult, RuntimeError,
};
use runmat_builtins::BuiltinCatalogIdentity;
use runmat_value::SparseTensor;

pub(crate) struct OrthogonalFrame {
    factor: SparseQrDecomposition<f64>,
    signs: Vec<f64>,
    dimension: usize,
    runtime: Option<RuntimeContext>,
    owner: BuiltinCatalogIdentity,
}

fn failure(
    error: Error,
    runtime: Option<&RuntimeContext>,
    owner: BuiltinCatalogIdentity,
) -> RuntimeError {
    let mut result = match error {
        Error::Execution(source) => crate::builtins::common::map_control_flow_with_builtin(
            source.into_runtime_error_in_context(runtime),
            owner.name,
        ),
        Error::Admission(source) => crate::context::host_allocation::host_allocation_runtime_error(
            runtime,
            "RunMat:OrthogonalFrame:Allocation",
            "original QR frame workspace",
            source,
        ),
        source => crate::build_runtime_error(format!(
            "{}: orthogonal QR frame failed: {source}",
            owner.name
        ))
        .with_builtin(owner.name)
        .with_source(source)
        .build(),
    };
    result.gpu_gather_retry = crate::GpuGatherRetry::Never;
    result
}
fn capacity<T>(
    n: usize,
    runtime: Option<&RuntimeContext>,
    owner: BuiltinCatalogIdentity,
) -> BuiltinResult<Vec<T>> {
    runmat_value::exact_capacity_vec(n).map_err(|source| {
        failure(
            Error::Admission(HostAllocationError::Numeric(source)),
            runtime,
            owner,
        )
    })
}
fn require_budget(
    runtime: Option<&RuntimeContext>,
    budget: Option<&CpuCooperationBudget>,
    owner: BuiltinCatalogIdentity,
) -> BuiltinResult<()> {
    let matches = match (runtime, budget) {
        (None, None) => true,
        (Some(runtime), Some(budget)) => budget
            .cancellation_token()
            .same_generation(&runtime.cancellation()),
        _ => false,
    };
    if !matches {
        return Err(failure(
            Error::Admission(HostAllocationError::StaleAuthority),
            runtime,
            owner,
        ));
    }
    Ok(())
}
fn capture_runtime(
    runtime: Option<&RuntimeContext>,
    budget: Option<&CpuCooperationBudget>,
    owner: BuiltinCatalogIdentity,
) -> BuiltinResult<Option<RuntimeContext>> {
    require_budget(runtime, budget, owner)?;
    Ok(runtime.map(|runtime| {
        runtime
            .clone()
            .with_cancellation_token(budget.expect("paired original budget").cancellation_token())
    }))
}
async fn tick(
    budget: &mut Option<&mut CpuCooperationBudget>,
    runtime: Option<&RuntimeContext>,
    owner: BuiltinCatalogIdentity,
) -> BuiltinResult<()> {
    if let Some(budget) = budget.as_deref_mut() {
        budget
            .checkpoint(1)
            .await
            .map_err(|e| failure(Error::Execution(e), runtime, owner))?;
    }
    Ok(())
}
#[derive(Clone, Copy)]
enum Shape {
    Square,
    Thin,
}
impl OrthogonalFrame {
    /// One original RNG turn. Explicit None retains the existing unmanaged
    /// context-free stream/math path; it never creates a context or budget.
    pub(crate) async fn random(
        dimension: usize,
        runtime: Option<&RuntimeContext>,
        owner: BuiltinCatalogIdentity,
        mut budget: Option<&mut CpuCooperationBudget>,
    ) -> BuiltinResult<Option<Self>> {
        let owned_runtime = capture_runtime(runtime, budget.as_deref(), owner)?;
        let runtime = owned_runtime.as_ref();
        let count = dimension
            .checked_mul(dimension)
            .ok_or_else(|| failure(Error::DimensionMismatch, runtime, owner))?;
        let pointer_count = dimension
            .checked_add(1)
            .filter(|_| dimension != 0)
            .ok_or_else(|| failure(Error::DimensionMismatch, runtime, owner))?;
        crate::context::host_allocation::require_unbounded_workspace(
            runtime,
            owner.name,
            "finite host-memory execution has no complete random QR frame fill bound",
        )?;
        let mut pointers = capacity(pointer_count, runtime, owner)?;
        let mut indices = capacity(count, runtime, owner)?;
        pointers.push(0usize);
        for _ in 0..dimension {
            for row in 0..dimension {
                indices.push(row);
                tick(&mut budget, runtime, owner).await?;
            }
            pointers.push(indices.len());
        }
        let mut turn =
            crate::builtins::common::random::acquire_original_turn(runtime, owner.name).await?;
        let values = match budget.as_deref_mut() {
            Some(budget) => {
                crate::builtins::common::random::generate_normal_cooperative(
                    &mut turn, count, owner.name, runtime, budget,
                )
                .await?
            }
            None => crate::builtins::common::random::generate_normal(&mut turn, count, owner.name)?,
        };
        drop(turn);
        let input = {
            let _entry = runtime.map(RuntimeContext::enter);
            SparseTensor::new(dimension, dimension, pointers, indices, values)
                .map_err(|detail| failure(Error::InvalidTopology(detail), runtime, owner))?
        };
        Self::factor_owned(&input, owned_runtime, owner, budget, Shape::Square).await
    }

    async fn factor_owned(
        input: &SparseTensor,
        owned_runtime: Option<RuntimeContext>,
        owner: BuiltinCatalogIdentity,
        mut budget: Option<&mut CpuCooperationBudget>,
        shape: Shape,
    ) -> BuiltinResult<Option<Self>> {
        let runtime = owned_runtime.as_ref();
        crate::context::host_allocation::require_unbounded_workspace(
            runtime,
            owner.name,
            "finite host-memory execution has no complete orthogonal frame fill bound",
        )?;
        if input.rows == 0
            || input.cols == 0
            || match shape {
                Shape::Square => input.rows != input.cols,
                Shape::Thin => input.rows < input.cols,
            }
            || input.is_complex()
            || input.as_f64_slice().is_none()
        {
            return Err(failure(Error::DimensionMismatch, runtime, owner));
        }
        let factor = SparseQrDecomposition::<f64>::factor_optional(input, false, runtime)
            .await
            .map_err(|e| failure(e, runtime, owner))?;
        let mut signs = capacity(input.cols, runtime, owner)?;
        for column in 0..input.cols {
            let (rows, values) = factor
                .r_column_optional(column, runtime)
                .map_err(|e| failure(e, runtime, owner))?;
            let diagonal = rows.binary_search(&column).ok().map(|i| values[i]);
            let Some(value) = diagonal.filter(|v| v.is_finite() && *v != 0.0) else {
                return Ok(None);
            };
            signs.push(if value < 0.0 { -1.0 } else { 1.0 });
            tick(&mut budget, runtime, owner).await?;
        }
        factor
            .check_owner_optional(runtime)
            .map_err(|e| failure(Error::Admission(e), runtime, owner))?;
        Ok(Some(Self {
            factor,
            signs,
            dimension: input.rows,
            runtime: owned_runtime,
            owner,
        }))
    }
    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    /// Apply the retained Haar frame to one caller-owned vector. N+1 polls
    /// can transform their sum/simplex ray with one Householder application.
    pub(crate) async fn transform_into(
        &self,
        output: &mut [f64],
        mut budget: Option<&mut CpuCooperationBudget>,
    ) -> BuiltinResult<()> {
        let runtime = self.runtime.as_ref();
        require_budget(runtime, budget.as_deref(), self.owner)?;
        if output.len() != self.dimension() {
            return Err(failure(Error::DimensionMismatch, runtime, self.owner));
        }
        self.factor
            .check_owner_optional(runtime)
            .map_err(|e| failure(Error::Admission(e), runtime, self.owner))?;
        for value in output.iter() {
            if !value.is_finite() {
                return Err(failure(Error::NonFiniteInput, runtime, self.owner));
            }
            tick(&mut budget, runtime, self.owner).await?;
        }
        for (value, &sign) in output.iter_mut().zip(&self.signs) {
            *value *= sign;
            tick(&mut budget, runtime, self.owner).await?;
        }
        self.factor
            .apply_q_optional(output, 1, false, runtime)
            .await
            .map_err(|e| failure(e, runtime, self.owner))?;
        self.factor
            .check_owner_optional(runtime)
            .map_err(|e| failure(Error::Admission(e), runtime, self.owner))?;
        Ok(())
    }
}

/// Thin orthonormal embedding of an actual sparse basis. The complement of
/// the span is not exposed as a sampler direction; callers supply zero tail.
pub(crate) struct OrthogonalSpan(OrthogonalFrame);
impl OrthogonalSpan {
    pub(crate) async fn factor(
        input: &SparseTensor,
        runtime: Option<&RuntimeContext>,
        owner: BuiltinCatalogIdentity,
        budget: Option<&mut CpuCooperationBudget>,
    ) -> BuiltinResult<Option<Self>> {
        let runtime = capture_runtime(runtime, budget.as_deref(), owner)?;
        OrthogonalFrame::factor_owned(input, runtime, owner, budget, Shape::Thin)
            .await
            .map(|frame| frame.map(Self))
    }
    pub(crate) fn dimension(&self) -> usize {
        self.0.dimension()
    }
    pub(crate) fn rank(&self) -> usize {
        self.0.signs.len()
    }
    pub(crate) async fn embed_into(
        &self,
        output: &mut [f64],
        mut budget: Option<&mut CpuCooperationBudget>,
    ) -> BuiltinResult<()> {
        let runtime = self.0.runtime.as_ref();
        require_budget(runtime, budget.as_deref(), self.0.owner)?;
        if output.len() != self.dimension() {
            return Err(failure(Error::DimensionMismatch, runtime, self.0.owner));
        }
        for &value in &output[self.rank()..] {
            if value != 0.0 {
                return Err(failure(Error::DimensionMismatch, runtime, self.0.owner));
            }
            tick(&mut budget, runtime, self.0.owner).await?;
        }
        self.0.transform_into(output, budget).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Deterministic matrix/basis adapters for this numerical fixture only.
    impl OrthogonalFrame {
        /// Unpivoted QR with R-diagonal sign correction. Independent Gaussian
        /// input therefore gives a Haar-distributed orthogonal frame.
        async fn factor(
            input: &SparseTensor,
            runtime: Option<&RuntimeContext>,
            owner: BuiltinCatalogIdentity,
            budget: Option<&mut CpuCooperationBudget>,
        ) -> BuiltinResult<Option<Self>> {
            let owned_runtime = capture_runtime(runtime, budget.as_deref(), owner)?;
            Self::factor_owned(input, owned_runtime, owner, budget, Shape::Square).await
        }
        /// Apply the same immutable Q to one caller-owned vector; no explicit Q
        /// matrix, input mirror, replacement runtime or scheduling budget.
        async fn direction_into(
            &self,
            column: usize,
            output: &mut [f64],
            mut budget: Option<&mut CpuCooperationBudget>,
        ) -> BuiltinResult<()> {
            let runtime = self.runtime.as_ref();
            require_budget(runtime, budget.as_deref(), self.owner)?;
            if column >= self.dimension() || output.len() != self.dimension() {
                return Err(failure(Error::DimensionMismatch, runtime, self.owner));
            }
            self.factor
                .check_owner_optional(runtime)
                .map_err(|e| failure(Error::Admission(e), runtime, self.owner))?;
            for value in output.iter_mut() {
                *value = 0.0;
                tick(&mut budget, runtime, self.owner).await?;
            }
            output[column] = 1.0;
            self.transform_into(output, budget).await
        }
    }
    #[test]
    fn explicit_none_and_original_context_share_householder_frame_math() {
        futures::executor::block_on(async {
            let cancellation = runmat_execution::cancellation::CancellationSource::new();
            let runtime = RuntimeContext::with_cancellation(
                std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
                cancellation.token(),
            );
            let mut budget = runtime.cpu_cooperation();
            let matrix = SparseTensor::new(
                2,
                2,
                vec![0, 2, 4],
                vec![0, 1, 0, 1],
                vec![1.0, 2.0, 3.0, 4.0],
            )
            .unwrap();
            let owner = BuiltinCatalogIdentity { name: "qr" };
            let foreign = crate::user_functions::test_support::exact_fixture_runtime();
            let mut foreign_budget = foreign.cpu_cooperation();
            assert!(
                OrthogonalFrame::random(2, Some(&runtime), owner, Some(&mut foreign_budget))
                    .await
                    .is_err()
            );
            let absent = OrthogonalFrame::factor(&matrix, None, owner, None)
                .await
                .unwrap()
                .unwrap();
            let original =
                OrthogonalFrame::factor(&matrix, Some(&runtime), owner, Some(&mut budget))
                    .await
                    .unwrap()
                    .unwrap();
            let mut first = [0.0; 2];
            let mut second = [0.0; 2];
            absent.direction_into(0, &mut first, None).await.unwrap();
            absent.direction_into(1, &mut second, None).await.unwrap();
            assert!((first[0].hypot(first[1]) - 1.0).abs() < 1e-12);
            assert!((second[0].hypot(second[1]) - 1.0).abs() < 1e-12);
            assert!((first[0] * second[0] + first[1] * second[1]).abs() < 1e-12);
            let mut repeated = [0.0; 2];
            original
                .direction_into(0, &mut repeated, Some(&mut budget))
                .await
                .unwrap();
            assert_eq!(first, repeated);
            original
                .direction_into(1, &mut repeated, Some(&mut budget))
                .await
                .unwrap();
            assert_eq!(second, repeated);
            let mut sum = [1.0; 2];
            original
                .transform_into(&mut sum, Some(&mut budget))
                .await
                .unwrap();
            for axis in 0..2 {
                assert!((sum[axis] - first[axis] - second[axis]).abs() < 1e-12);
            }

            let plane = SparseTensor::new(
                3,
                2,
                vec![0, 2, 4],
                vec![0, 2, 1, 2],
                vec![1.0, 1.0, 1.0, 1.0],
            )
            .unwrap();
            let span = OrthogonalSpan::factor(&plane, Some(&runtime), owner, Some(&mut budget))
                .await
                .unwrap()
                .unwrap();
            assert_eq!((span.dimension(), span.rank()), (3, 2));
            let mut embedded = [1.0, 0.0, 0.0];
            span.embed_into(&mut embedded, Some(&mut budget))
                .await
                .unwrap();
            assert!((embedded[2] - embedded[0] - embedded[1]).abs() < 1e-12);
            assert!((embedded.iter().fold(0.0_f64, |norm, &x| norm.hypot(x)) - 1.0).abs() < 1e-12);
            let mut outside = [1.0, 0.0, 1.0];
            assert!(span
                .embed_into(&mut outside, Some(&mut budget))
                .await
                .is_err());
            assert_eq!(outside, [1.0, 0.0, 1.0]);
            let replacement = runmat_execution::cancellation::CancellationSource::new();
            let _scope = runtime.scoped_cancellation_token(replacement.token());
            let mut rebound_budget = runtime.cpu_cooperation();
            repeated = [123.0; 2];
            assert!(original
                .direction_into(0, &mut repeated, Some(&mut rebound_budget))
                .await
                .is_err());
            assert_eq!(repeated, [123.0; 2]);
            original
                .direction_into(0, &mut repeated, Some(&mut budget))
                .await
                .unwrap();
            assert_eq!(first, repeated);
            cancellation.cancel(runmat_execution::CancellationReason::User);
            assert!(original
                .direction_into(0, &mut repeated, Some(&mut budget))
                .await
                .is_err());
        });
    }
}
