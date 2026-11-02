use super::{prepare, PreparationError};

#[test]
fn typed_calls_preserve_comments_and_single_precision() {
    let source = "alias Double = f64; fn f(x: Double) -> Double { return exp /* call */ (log(x)); } fn single(x: f32) -> f32 { return exp(x); }";
    let prepared = prepare(source, None).unwrap();
    assert!(prepared
        .source()
        .contains("runmat_math64_exp /* call */ (runmat_math64_log(x))"));
    assert!(prepared.source().contains("return exp(x);"));
    assert!(prepared.requires_f64_math());
    assert_eq!(
        prepare(prepared.source(), None).unwrap().source(),
        prepared.source()
    );
    assert!(prepare(prepared.source(), None)
        .unwrap()
        .requires_f64_math());
}

#[test]
fn typed_vector_wrappers_cover_every_width() {
    for width in 2..=4 {
        let source =
            format!("fn f(x: vec{width}<f64>) -> vec{width}<f64> {{ return exp(log(x)); }}");
        let prepared = prepare(&source, None).unwrap();
        assert!(prepared.source().contains(&format!(
            "runmat_math64_exp_vec{width}(runmat_math64_log_vec{width}(x))"
        )));
        assert!(prepared
            .source()
            .contains(&format!("runmat_math64_log(a[{}])", width - 1)));
    }
}

#[test]
fn canonical_ieee_helpers_are_shared_by_parsed_declaration() {
    let source = crate::wgsl::ieee_words::inject("fn f(x: f64) -> f64 { return exp(x); }");
    let prepared = prepare(&source, None).unwrap();
    assert_eq!(prepared.source().matches("fn runmat_f64_high(").count(), 1);
    assert_eq!(prepared.source().matches("fn runmat_f64_low(").count(), 1);
    assert_eq!(
        prepared
            .source()
            .matches("fn runmat_f64_from_words(")
            .count(),
        1
    );
}

#[test]
fn incompatible_reserved_declaration_is_a_structured_error() {
    let source =
        "fn runmat_f64_low(x: f64) -> u32 { return 0u; } fn f(x: f64) -> f64 { return exp(x); }";
    assert!(matches!(
        prepare(source, None),
        Err(PreparationError::ReservedHelperCollision(_))
    ));
}

#[test]
fn permitted_intrinsics_and_single_precision_source_are_unchanged() {
    for source in [
        "fn f(x: f64) -> f64 { return sqrt(x) + floor(x); }",
        "fn f(x: f32) -> f32 { return exp(log(x)); }",
    ] {
        assert_eq!(prepare(source, None).unwrap().source(), source);
    }
}

#[test]
fn malformed_shader_is_not_deferred_to_device_creation() {
    let error = match prepare("not WGSL", None) {
        Err(error) => error,
        Ok(_) => panic!("malformed shader must fail before device creation"),
    };
    assert!(matches!(&error, PreparationError::Parse(_)));
    let mut cause = Some(&error as &dyn std::error::Error);
    let mut original_parse_error = false;
    while let Some(error) = cause {
        original_parse_error |= error
            .downcast_ref::<naga::front::wgsl::ParseError>()
            .is_some();
        cause = error.source();
    }
    assert!(
        original_parse_error,
        "original typed Naga cause survives legalization"
    );
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    {
        let Err(error) = crate::backend::wgpu::shader_compilation::spirv::compile("not WGSL", None)
        else {
            panic!("malformed SPIR-V preparation source rejects");
        };
        assert!(error.chain().any(|cause| cause
            .downcast_ref::<naga::front::wgsl::ParseError>()
            .is_some()));
    }
}

#[test]
fn every_restricted_intrinsic_and_shape_is_legalized() {
    use super::intrinsic::{Intrinsic, Shape};
    let operations = [
        Intrinsic::Radians,
        Intrinsic::Degrees,
        Intrinsic::Sin,
        Intrinsic::Cos,
        Intrinsic::Tan,
        Intrinsic::Asin,
        Intrinsic::Acos,
        Intrinsic::Atan,
        Intrinsic::Atan2,
        Intrinsic::Sinh,
        Intrinsic::Cosh,
        Intrinsic::Tanh,
        Intrinsic::Asinh,
        Intrinsic::Acosh,
        Intrinsic::Atanh,
        Intrinsic::Exp,
        Intrinsic::Exp2,
        Intrinsic::Log,
        Intrinsic::Log2,
        Intrinsic::Pow,
    ];
    let mut failures = Vec::new();
    for intrinsic in operations {
        for shape in [Shape::Scalar, Shape::Vec2, Shape::Vec3, Shape::Vec4] {
            let ty = shape
                .vector_width()
                .map_or_else(|| "f64".to_owned(), |width| format!("vec{width}<f64>"));
            let arguments = if intrinsic.arguments() == 2 {
                "x, x"
            } else {
                "x"
            };
            let source = format!(
                "fn f(x: {ty}) -> {ty} {{ return {}({arguments}); }}",
                intrinsic.spelling()
            );
            let prepared = match prepare(&source, None) {
                Ok(prepared) => prepared,
                Err(error) => {
                    failures.push(format!("{intrinsic:?}/{shape:?}: {error}"));
                    continue;
                }
            };
            assert!(prepared
                .source()
                .contains(&format!("return {}({arguments});", intrinsic.helper(shape))));
            assert_eq!(
                prepare(prepared.source(), None).unwrap().source(),
                prepared.source()
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn boxed_validation_error_retains_exact_funding_and_original_cause() {
    use runmat_execution::resource::{
        ProviderHostMetadataAdmission, ProviderHostMetadataError, RetainedHostAllocationClaim,
        RetainedHostViewGrant,
    };
    use std::error::Error as _;
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct Admission {
        live: Arc<AtomicU64>,
        peak: AtomicU64,
        limit: AtomicU64,
    }
    #[derive(Debug)]
    struct Funding {
        live: Arc<AtomicU64>,
        bytes: u64,
    }
    impl Drop for Funding {
        fn drop(&mut self) {
            self.live.fetch_sub(self.bytes, Ordering::SeqCst);
        }
    }
    impl ProviderHostMetadataAdmission for Admission {
        fn claim_view(
            &self,
            _: u64,
        ) -> Result<Box<dyn RetainedHostViewGrant>, ProviderHostMetadataError> {
            Err(ProviderHostMetadataError::OwnerFailure)
        }
        fn claim_record(
            &self,
            bytes: u64,
        ) -> Result<Arc<dyn RetainedHostAllocationClaim>, ProviderHostMetadataError> {
            let previous = self
                .live
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |live| {
                    live.checked_add(bytes)
                        .filter(|&next| next <= self.limit.load(Ordering::SeqCst))
                })
                .map_err(|_| ProviderHostMetadataError::ExceedsCapacity)?;
            self.peak.fetch_max(previous + bytes, Ordering::SeqCst);
            Ok(Arc::new(Funding {
                live: self.live.clone(),
                bytes,
            }))
        }
    }
    let bytes = std::mem::size_of::<naga::WithSpan<naga::valid::ValidationError>>() as u64;
    let owner = Admission {
        live: Arc::new(AtomicU64::new(0)),
        peak: AtomicU64::new(0),
        limit: AtomicU64::new(bytes - 1),
    };
    let invalid = "@compute @workgroup_size(0) fn main() {}";
    assert!(matches!(
        prepare(invalid, Some(&owner)),
        Err(PreparationError::Buffer(
            crate::wgsl::host_buffers::Error::Metadata(ProviderHostMetadataError::ExceedsCapacity)
        ))
    ));
    assert_eq!(owner.live.load(Ordering::SeqCst), 0);
    owner.limit.store(bytes, Ordering::SeqCst);
    let Err(error) = prepare(invalid, Some(&owner)) else {
        panic!("invalid workgroup size must fail validation");
    };
    assert!(matches!(&error, PreparationError::Validate(_)));
    assert_eq!(owner.live.load(Ordering::SeqCst), bytes);
    assert!(error
        .source()
        .and_then(|cause| cause.source())
        .and_then(|cause| cause.downcast_ref::<naga::WithSpan<naga::valid::ValidationError>>())
        .is_some());
    drop(error);
    assert_eq!(owner.live.load(Ordering::SeqCst), 0);
    let valid = "fn f() -> f32 { return 1.0; }";
    let prepared = prepare(valid, Some(&owner)).unwrap();
    // Successful validation releases the unused failure reservation. Only the
    // independently charged returned source remains; no phantom box is held.
    assert_eq!(owner.live.load(Ordering::SeqCst), valid.len() as u64);
    assert_eq!(owner.peak.load(Ordering::SeqCst), bytes);
    drop(prepared);
    assert_eq!(owner.live.load(Ordering::SeqCst), 0);
}
