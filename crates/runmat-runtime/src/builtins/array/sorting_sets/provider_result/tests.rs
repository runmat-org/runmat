use super::*;
use futures::executor::block_on;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{
    AccelDownloadFuture, AccelProvider, HostNumericDataView, HostNumericTensorView, HostTensorView,
    SortComparison, SortOrder,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy)]
enum Outcome {
    None,
    Some,
    Failed,
    UnsupportedAfterAttempt,
    LanguageError,
}
struct MockProvider {
    inner: InProcessProvider,
    outcome: Outcome,
    calls: AtomicUsize,
    reads: AtomicUsize,
}
impl AccelProvider for MockProvider {
    fn device_info(&self) -> String {
        "sorting outcome fixture".into()
    }
    fn upload(&self, host: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        self.inner.upload(host)
    }
    fn download<'a>(&'a self, handle: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.download(handle)
    }
    fn sort_dim<'a>(
        &'a self,
        _: &'a GpuTensorHandle,
        _: usize,
        _: SortOrder,
        _: SortComparison,
    ) -> ProviderExecutionFuture<'a, Option<SortResult>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.outcome {
                Outcome::None => Ok(None),
                Outcome::Some => Ok(Some(SortResult {
                    values: host(vec![1.0, 2.0], vec![2, 1]),
                    indices: host(vec![2.0, 1.0], vec![2, 1]),
                })),
                Outcome::Failed => Err(ProviderExecutionError::from_source(std::io::Error::other(
                    "execution failed",
                ))
                .after_effect()),
                Outcome::UnsupportedAfterAttempt => Err(ProviderExecutionError::from(
                    runmat_accelerate_api::unsupported_provider_operation(
                        "attempted then unsupported",
                    ),
                )
                .with_unproven_effects()),
                Outcome::LanguageError => Err(ProviderExecutionError::from_source(
                    crate::runtime_error::semantic_error(
                        "RunMat:Fixture:Cancelled",
                        "cancelled during provider execution",
                    ),
                )
                .after_effect()),
            }
        })
    }
}
fn host(data: Vec<f64>, shape: Vec<usize>) -> HostTensorOwned {
    HostTensorOwned {
        data,
        shape,
        storage: GpuTensorStorage::Real,
    }
}
fn error(message: String) -> RuntimeError {
    crate::build_runtime_error(message)
        .with_identifier("RunMat:sort:Internal")
        .build()
}
fn fixture(outcome: Outcome) -> (MockProvider, GpuTensorHandle) {
    let inner = InProcessProvider::new();
    let input = inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[2.0, 1.0]),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    (
        MockProvider {
            inner,
            outcome,
            calls: AtomicUsize::new(0),
            reads: AtomicUsize::new(0),
        },
        input,
    )
}

#[test]
fn shared_optional_provider_outcome_allows_only_none_to_select_host() {
    for outcome in [
        Outcome::None,
        Outcome::Some,
        Outcome::Failed,
        Outcome::UnsupportedAfterAttempt,
        Outcome::LanguageError,
    ] {
        let (provider, input) = fixture(outcome);
        let result = block_on(attempt(
            provider.sort_dim(&input, 0, SortOrder::Ascend, SortComparison::Auto),
            error,
        ));
        match result {
            Ok(None) => {
                assert!(matches!(outcome, Outcome::None));
                block_on(provider.download(&input)).unwrap();
            }
            Ok(Some(result)) => {
                assert!(matches!(outcome, Outcome::Some));
                let (values, indices) =
                    sort(result, &input, SortGeometry::Dimension(0), error).unwrap();
                assert_eq!(values.numeric_dtype(), NumericDType::F32);
                assert_eq!(values.materialize_f64(), vec![1.0, 2.0]);
                assert_eq!(indices.numeric_dtype(), NumericDType::F64);
                assert_eq!(indices.materialize_f64(), vec![2.0, 1.0]);
            }
            Err(failure) => {
                assert!(!matches!(outcome, Outcome::None | Outcome::Some));
                assert_eq!(failure.gpu_gather_retry, crate::GpuGatherRetry::Never);
                assert!(!runmat_accelerate_api::is_unsupported_provider_error(
                    &failure
                ));
                if matches!(outcome, Outcome::LanguageError) {
                    assert_eq!(failure.identifier(), Some("RunMat:Fixture:Cancelled"));
                }
            }
        }
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.reads.load(Ordering::SeqCst),
            usize::from(matches!(outcome, Outcome::None))
        );
    }
}

#[test]
fn malformed_completed_outputs_are_terminal_and_permutation_validation_is_linear_in_place() {
    let (_, input) = fixture(Outcome::Some);
    for result in [
        SortResult {
            values: host(vec![1.0, 2.0], vec![1, 2]),
            indices: host(vec![2.0, 1.0], vec![2, 1]),
        },
        SortResult {
            values: host(vec![1.0, 2.0], vec![2, 1]),
            indices: host(vec![1.0, 1.0], vec![2, 1]),
        },
        SortResult {
            values: host(vec![1.0 + 2.0f64.powi(-30), 2.0], vec![2, 1]),
            indices: host(vec![2.0, 1.0], vec![2, 1]),
        },
    ] {
        let failure = sort(result, &input, SortGeometry::Dimension(0), error).unwrap_err();
        assert_eq!(failure.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(!runmat_accelerate_api::is_unsupported_provider_error(
            &failure
        ));
    }
    let mut indices = [2.0, 2.0, 1.0, 1.0];
    permutation(&mut indices, &[2, 2], 1, error).unwrap();
    assert_eq!(indices, [2.0, 2.0, 1.0, 1.0]);
    assert_eq!(cardinality(&[0, usize::MAX, usize::MAX], error).unwrap(), 0);
    assert!(cardinality(&[usize::MAX, 2], error).is_err());
    let provider = InProcessProvider::new();
    let invalid_unique = UniqueResult {
        values: host(vec![1.0, 2.0], vec![2, 1]),
        ia: host(vec![1.0, 1.0], vec![2, 1]),
        ic: host(vec![1.0, 2.0], vec![2, 1]),
    };
    let failure = unique(invalid_unique, &input, false, error).unwrap_err();
    assert_eq!(failure.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let logical = provider
        .upload_logical_raw(&runmat_accelerate_api::HostLogicalTensorView {
            data: &[0, 1],
            shape: &[2, 1],
        })
        .unwrap();
    for invalid_value in [-1.0, 2.0, f64::INFINITY, f64::NAN] {
        let failure = unique(
            UniqueResult {
                values: host(vec![0.0, invalid_value], vec![2, 1]),
                ia: host(vec![1.0, 2.0], vec![2, 1]),
                ic: host(vec![1.0, 2.0], vec![2, 1]),
            },
            &logical,
            false,
            error,
        )
        .unwrap_err();
        assert_eq!(failure.gpu_gather_retry, crate::GpuGatherRetry::Never);
    }
    assert!(unique(
        UniqueResult {
            values: host(vec![0.0, 1.0], vec![2, 1]),
            ia: host(vec![1.0, 2.0], vec![2, 1]),
            ic: host(vec![1.0, 2.0], vec![2, 1]),
        },
        &logical,
        false,
        error,
    )
    .is_ok());
}

#[test]
fn provider_set_outputs_use_the_same_orientation_policy_as_host_results() {
    let inner = InProcessProvider::new();
    let a = inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 2.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let b = inner
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[2.0, 3.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let result = UnionResult {
        values: host(vec![1.0, 2.0, 3.0], vec![1, 3]),
        ia: host(vec![1.0, 2.0], vec![2, 1]),
        ib: host(vec![2.0], vec![1, 1]),
    };
    let (values, _, _) = union(result, &a, &b, false, error).unwrap();
    assert_eq!(values.shape(), &[1, 3]);
    assert_eq!(values.numeric_dtype(), NumericDType::F32);
    let result = SetdiffResult {
        values: host(vec![1.0], vec![1, 1]),
        ia: host(vec![1.0], vec![1, 1]),
    };
    assert_eq!(
        setdiff(result, &a, false, error).unwrap().0.shape(),
        &[1, 1]
    );
    let invalid = UnionResult {
        values: host(vec![1.0, 2.0, 3.0], vec![3, 1]),
        ia: host(vec![1.0, 2.0], vec![2, 1]),
        ib: host(vec![2.0], vec![1, 1]),
    };
    assert_eq!(
        union(invalid, &a, &b, false, error)
            .unwrap_err()
            .gpu_gather_retry,
        crate::GpuGatherRetry::Never
    );
}
