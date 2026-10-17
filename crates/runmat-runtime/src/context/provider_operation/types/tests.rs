use super::{conditional_output_owner_valid, ConditionalUploadPublication};
use crate::builtins::common::gpu_helpers;
use runmat_accelerate::simple_provider::InProcessProvider;
use runmat_accelerate_api::{same_provider_instance, AccelProvider, ThreadProviderGuard};
use runmat_value::Tensor;

struct ForeignFacade(&'static InProcessProvider);

impl AccelProvider for ForeignFacade {
    fn upload(
        &self,
        host: &runmat_accelerate_api::HostTensorView,
    ) -> anyhow::Result<runmat_accelerate_api::GpuTensorHandle> {
        self.0.upload(host)
    }

    fn download<'a>(
        &'a self,
        handle: &'a runmat_accelerate_api::GpuTensorHandle,
    ) -> runmat_accelerate_api::AccelDownloadFuture<'a> {
        self.0.download(handle)
    }

    fn device_info(&self) -> String {
        self.0.device_info()
    }

    fn device_id(&self) -> u32 {
        self.0.device_id()
    }
}

#[test]
fn conditional_physical_publication_needs_exact_witness_and_rejection_is_terminal() {
    let mut pending = ConditionalUploadPublication::new(Some(8));
    assert!(!pending.validated());
    assert!(!pending.conditional_attempted());
    pending.begin();
    assert!(pending.conditional_attempted());
    assert!(pending.validate(8, true));
    assert!(pending.validated());
    assert!(!pending.validate(32, true));
    assert!(!pending.validate(8, true));
    assert!(!pending.validated());
    assert!(!ConditionalUploadPublication::new(None).validate(8, true));
    let mut invalid_owner = ConditionalUploadPublication::new(Some(8));
    invalid_owner.begin();
    assert!(!invalid_owner.validate(8, false));
    let mut premature = ConditionalUploadPublication::new(Some(8));
    assert!(!premature.validate(8, true));
    assert!(premature.conditional_attempted());
}

#[test]
fn same_device_foreign_facade_is_not_a_conditional_output_owner() {
    let owner: &'static InProcessProvider = Box::leak(Box::new(InProcessProvider::new()));
    let foreign: &'static ForeignFacade = Box::leak(Box::new(ForeignFacade(owner)));
    assert_eq!(owner.device_id(), foreign.device_id());
    assert!(!same_provider_instance(owner, foreign));
    let _selected_owner = ThreadProviderGuard::set(Some(owner));
    let output = gpu_helpers::upload_tensor(
        owner,
        &Tensor::new(vec![1.0], vec![1, 1]).expect("one-element result"),
    )
    .expect("owner result");
    assert!(!conditional_output_owner_valid(foreign, &output));
    let mut selected = ConditionalUploadPublication::new(Some(output.physical().requested_bytes()));
    selected.begin();
    assert!(!selected.validate(
        output.physical().requested_bytes(),
        conditional_output_owner_valid(foreign, &output),
    ));
    assert!(!selected.validated());
    assert!(output.allocation_lease().validate_live().is_ok());
}
