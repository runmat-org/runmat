use super::*;
use crate::test_support::TestAllocationOwner;
use crate::*;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, Waker};

fn request(input: &GpuTensorHandle) -> ProviderSpectralRequest<'_> {
    ProviderSpectralRequest {
        input,
        input_len: 16,
        input_complex: false,
        window: &[1.0; 4],
        nfft: 8,
        frame_count: 4,
        frame_mode: ProviderSpectralFrameMode::ColumnSliding {
            hop: 4,
            input_rows: 8,
            frames_per_column: 2,
        },
        range: ProviderSpectralRange::Onesided,
        denominator: 4.0,
    }
}

#[test]
fn zero_frames_and_overflow_reject_without_arithmetic_panic() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![8, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut request = request(&input.handle);
    for mode in [
        ProviderSpectralFrameMode::Sliding { hop: 4 },
        ProviderSpectralFrameMode::ColumnSliding {
            hop: 4,
            input_rows: 8,
            frames_per_column: 2,
        },
        ProviderSpectralFrameMode::FoldedColumns { input_rows: 8 },
    ] {
        request.frame_mode = mode;
        request.frame_count = 0;
        assert!(validate_uniform_spectral_request(&request).is_err());
        request.frame_count = usize::MAX;
        assert!(validate_uniform_spectral_request(&request).is_err());
    }
    request.frame_count = 1;
    request.nfft = usize::MAX / 2 + 1;
    assert!(validate_uniform_spectral_request(&request).is_err());
    assert_eq!(input.probe.retirements(), 0);
}

#[test]
fn column_frames_cannot_read_across_column_boundary() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![8, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut request = request(&input.handle);
    assert!(validate_uniform_spectral_request(&request).is_ok());
    // Flattened coverage is only 10 <= 16, but frame two crosses row 8.
    request.frame_count = 2;
    request.frame_mode = ProviderSpectralFrameMode::ColumnSliding {
        hop: 6,
        input_rows: 8,
        frames_per_column: 2,
    };
    assert!(validate_uniform_spectral_request(&request).is_err());
    request.frame_count = 1;
    assert!(validate_uniform_spectral_request(&request).is_ok());
}

#[test]
fn folded_windows_cannot_read_samples_from_the_next_column() {
    let owner = TestAllocationOwner::new();
    let input = owner
        .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut request = request(&input.handle);
    request.input_len = 4;
    request.nfft = 2;
    request.frame_count = 2;
    request.window = &[1.; 3];
    request.frame_mode = ProviderSpectralFrameMode::FoldedColumns { input_rows: 2 };
    assert!(validate_uniform_spectral_request(&request).is_err());
    request.window = &[1.; 2];
    assert!(validate_uniform_spectral_request(&request).is_ok());
    // A shorter transform folds within each column, and remains supported.
    request.nfft = 1;
    assert!(validate_uniform_spectral_request(&request).is_ok());
}

#[test]
fn request_metadata_must_match_actual_typed_view() {
    let owner = TestAllocationOwner::new();
    for storage in [GpuTensorStorage::Real, GpuTensorStorage::ComplexInterleaved] {
        let input = owner
            .allocate_zeroed(vec![8, 2], NumericElementType::F32, storage)
            .unwrap();
        let mut request = request(&input.handle);
        request.input_complex = storage == GpuTensorStorage::ComplexInterleaved;
        assert!(validate_uniform_spectral_request(&request).is_ok());
        request.input_complex = !request.input_complex;
        assert!(validate_uniform_spectral_request(&request).is_err());
        request.input_complex = !request.input_complex;
        request.input_len = 15;
        assert!(validate_uniform_spectral_request(&request).is_err());
    }
}

struct NoWorkProvider {
    owner: TestAllocationOwner,
    calls: AtomicUsize,
}
impl AccelProvider for NoWorkProvider {
    fn device_id(&self) -> u32 {
        self.owner.device_id()
    }
    fn device_info(&self) -> String {
        "spectral validation fixture".into()
    }
    fn upload(&self, _: &HostTensorView) -> anyhow::Result<GpuTensorHandle> {
        panic!("validation must not upload")
    }
    fn download<'a>(&'a self, _: &'a GpuTensorHandle) -> AccelDownloadFuture<'a> {
        panic!("validation must not download")
    }
    fn uniform_spectral_estimate<'a>(
        &'a self,
        _: &'a ProviderSpectralRequest<'a>,
    ) -> ProviderExecutionFuture<'a, ProviderSpectralResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { panic!("invalid input must not reach hook") })
    }
}

#[test]
fn public_preflight_has_zero_provider_effects_and_retains_borrowed_owner() {
    let provider = Box::leak(Box::new(NoWorkProvider {
        owner: TestAllocationOwner::new(),
        calls: AtomicUsize::new(0),
    }));
    let _selected = ThreadProviderGuard::set(Some(provider));
    let input = provider
        .owner
        .allocate_zeroed(vec![8, 2], NumericElementType::F64, GpuTensorStorage::Real)
        .unwrap();
    let mut request = request(&input.handle);
    request.frame_count = 0;
    let mut future = Box::pin(uniform_spectral_estimate(request));
    let mut context = Context::from_waker(Waker::noop());
    assert!(matches!(
        future.as_mut().poll(&mut context),
        Poll::Ready(Err(_))
    ));
    drop(future);
    input.probe.mark_unavailable();
    let mut future = Box::pin(uniform_spectral_estimate(self::request(&input.handle)));
    assert!(matches!(
        future.as_mut().poll(&mut context),
        Poll::Ready(Err(_))
    ));
    drop(future);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(input.probe.retirements(), 0);
    drop(input.handle);
    assert_eq!(input.probe.retirements(), 1);
}
