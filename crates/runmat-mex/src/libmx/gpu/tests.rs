use super::*;
use crate::{value_from_mx, MxApiMode};
use runmat_accelerate_api::AccelProvider;
use runmat_value::Value;
use std::sync::atomic::Ordering;

#[path = "tests/provider.rs"]
mod provider;
use provider::{FixtureAmbientProvider, FixtureCudaProvider};

#[test]
fn logical_gpu_arrays_use_owned_byte_class_through_upload_native_write_and_publication() {
    let _gate = FixtureCudaProvider::test_gate().lock().unwrap();
    let provider = FixtureCudaProvider::shared();
    provider.reset();
    let _selection = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let context = crate::host::NativeInvocationContext::enter(provider).unwrap();
    let mut api = MxApi::new(MxApiMode::InterleavedComplex);
    let host = api.allocate(MxArray::logical(vec![1, 0], vec![1, 1, 2]).unwrap());
    let gpu = api.gpu_create_from_array(host, false).unwrap();
    let duplicate = api.gpu_create_from_array(gpu, true).unwrap();
    assert!(!api.gpu_is_same(gpu, duplicate).unwrap());
    assert_eq!(api.gpu_class_id(gpu).unwrap(), MxClassId::Logical);
    let address = api.gpu_data(duplicate, true).unwrap();
    // SAFETY: genuine owned U8 backing, retained by the entered context;
    // the two-byte logical allocation has no concurrent native access.
    unsafe {
        (address as *mut u8).add(1).write(1);
    }
    let gathered = api.gpu_to_host(duplicate).unwrap();
    let MxArrayData::Logical(data) = api.arena().get(gathered).unwrap().data() else {
        panic!("logical class lost")
    };
    assert_eq!(data.as_slice(), [1, 1]);
    let published = value_from_mx(api.arena().get(duplicate).unwrap()).unwrap();
    let Value::GpuTensor(handle) = &published else {
        panic!("output is not resident")
    };
    assert!(runmat_accelerate_api::handle_is_logical(handle));
    assert_eq!(handle.physical().element_type(), NumericElementType::U8);
    assert_eq!(handle.shape(), [1, 1, 2]);
    api.arena_mut().destroy(gpu).unwrap();
    api.arena_mut().destroy(duplicate).unwrap();
    drop(published);
    assert_eq!(provider.frees(), 1);
    context.leave().unwrap();
    assert_eq!(provider.frees(), 2);
    assert_eq!(provider.live_count(), 0);
}

#[test]
fn gpu_arrays_preserve_alias_identity_and_transfer_owned_lifetimes_once() {
    let _gate = FixtureCudaProvider::test_gate().lock().unwrap();
    let provider = FixtureCudaProvider::shared();
    provider.reset();
    let _selection = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

    {
        let guard = provider.enter_native_device_context().unwrap();
        assert_eq!(provider.context_enters.load(Ordering::Acquire), 1);
        drop(guard);
        assert_eq!(provider.context_leaves.load(Ordering::Acquire), 1);
    }

    let mut api = MxApi::new(MxApiMode::InterleavedComplex);
    let host = api
        .allocate(MxArray::numeric(NumericStorage::F64(vec![1.0, 2.0]), vec![1, 2], None).unwrap());
    let gpu = api.gpu_create_from_array(host, false).unwrap();
    let alias = api.gpu_create_from_array(gpu, false).unwrap();
    let duplicate = api.gpu_create_from_array(gpu, true).unwrap();
    assert!(api.gpu_is_same(gpu, alias).unwrap());
    assert!(!api.gpu_is_same(gpu, duplicate).unwrap());

    let context = crate::host::NativeInvocationContext::enter(provider).unwrap();
    let address = api.gpu_data(gpu, true).unwrap();
    // SAFETY: the fixture provider deliberately exposes host-addressable
    // storage to exercise the native pointer and ownership contract.
    unsafe { *(address as *mut f64) = 9.0 };
    let gathered = api.gpu_to_host(alias).unwrap();
    let Value::Tensor(gathered) = value_from_mx(api.arena().get(gathered).unwrap()).unwrap() else {
        panic!("gathered GPU array must be a dense tensor");
    };
    assert_eq!(gathered.materialize_f64(), vec![9.0, 2.0]);

    api.arena_mut().destroy(duplicate).unwrap();
    assert_eq!(provider.frees(), 1);
    api.arena_mut().destroy(gpu).unwrap();
    assert_eq!(provider.frees(), 1);

    let Value::GpuTensor(published) = value_from_mx(api.arena().get(alias).unwrap()).unwrap()
    else {
        panic!("GPU output must retain its provider handle");
    };
    api.arena_mut().destroy(alias).unwrap();
    assert_eq!(provider.frees(), 1);
    assert_eq!(published.shape(), vec![1, 2]);
    drop(published);
    assert_eq!(
        provider.frees(),
        1,
        "native export still owns the allocation"
    );
    context.leave().unwrap();
    assert_eq!(provider.frees(), 2);
}

#[test]
fn non_cuda_gpu_input_transfers_explicitly_to_the_cuda_provider() {
    let _gate = FixtureCudaProvider::test_gate().lock().unwrap();
    let cuda = FixtureCudaProvider::shared();
    let ambient = FixtureAmbientProvider::shared();
    cuda.reset();
    ambient.downloads.store(0, Ordering::Release);
    let _selection = runmat_accelerate_api::ThreadProviderGuard::set(Some(ambient));
    let readback_before =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::ProviderReadback);
    let upload_before =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::ProviderUpload);

    let mut api = MxApi::new(MxApiMode::InterleavedComplex);
    let source = api.allocate(MxArray::gpu_borrowed(MxClassId::Double, ambient.handle()).unwrap());
    let transferred = api.gpu_create_from_array(source, false).unwrap();
    let MxArrayData::Gpu(transferred_gpu) = api.arena().get(transferred).unwrap().data() else {
        panic!("transferred array must remain provider-resident");
    };
    assert_eq!(transferred_gpu.lease.handle().device_id(), cuda.device_id());
    assert_ne!(
        transferred_gpu.lease.handle().device_id(),
        ambient.device_id()
    );
    assert_eq!(ambient.downloads.load(Ordering::Acquire), 1);

    let gathered = api.gpu_to_host(transferred).unwrap();
    let Value::Tensor(gathered) = value_from_mx(api.arena().get(gathered).unwrap()).unwrap() else {
        panic!("transferred array must gather as a dense tensor");
    };
    assert_eq!(gathered.materialize_f64(), vec![11.0, 13.0]);
    let readback_after =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::ProviderReadback);
    let upload_after =
        runmat_value::host_copy_metrics(runmat_value::HostCopyReason::ProviderUpload);
    assert_eq!(readback_after.operations - readback_before.operations, 2);
    assert_eq!(readback_after.bytes - readback_before.bytes, 32);
    assert_eq!(upload_after.operations - upload_before.operations, 1);
    assert_eq!(upload_after.bytes - upload_before.bytes, 16);

    api.arena_mut().destroy(source).unwrap();
    api.arena_mut().destroy(transferred).unwrap();
    assert_eq!(cuda.frees(), 1);
}

#[test]
fn complex_gpu_components_stay_on_the_provider_and_round_trip_exactly() {
    let _gate = FixtureCudaProvider::test_gate().lock().unwrap();
    let provider = FixtureCudaProvider::shared();
    provider.reset();
    let _selection = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

    let mut api = MxApi::new(MxApiMode::InterleavedComplex);
    let real = api
        .gpu_create(MxClassId::Double, vec![2, 1], false, true)
        .unwrap();
    let imaginary = api
        .gpu_create(MxClassId::Double, vec![2, 1], false, false)
        .unwrap();
    let context = crate::host::NativeInvocationContext::enter(provider).unwrap();
    let real_address = api.gpu_data(real, true).unwrap() as *mut f64;
    let imaginary_address = api.gpu_data(imaginary, true).unwrap() as *mut f64;
    // SAFETY: fixture allocations are aligned, host-addressable f64 lanes.
    unsafe {
        real_address.write(1.5);
        real_address.add(1).write(-2.0);
        imaginary_address.write(3.0);
        imaginary_address.add(1).write(4.5);
    }

    let complex = api.gpu_create_complex(real, imaginary).unwrap();
    assert!(api.arena().get(complex).unwrap().is_complex());
    let copied_real = api
        .gpu_copy_component(complex, NativeDeviceComponent::Real)
        .unwrap();
    let copied_imaginary = api
        .gpu_copy_component(complex, NativeDeviceComponent::Imaginary)
        .unwrap();
    let gathered_real = api.gpu_to_host(copied_real).unwrap();
    let gathered_imaginary = api.gpu_to_host(copied_imaginary).unwrap();
    let Value::Tensor(gathered_real) =
        value_from_mx(api.arena().get(gathered_real).unwrap()).unwrap()
    else {
        panic!("real component must gather as a tensor");
    };
    let Value::Tensor(gathered_imaginary) =
        value_from_mx(api.arena().get(gathered_imaginary).unwrap()).unwrap()
    else {
        panic!("imaginary component must gather as a tensor");
    };
    assert_eq!(gathered_real.materialize_f64(), vec![1.5, -2.0]);
    assert_eq!(gathered_imaginary.materialize_f64(), vec![3.0, 4.5]);

    for pointer in [real, imaginary, complex, copied_real, copied_imaginary] {
        api.arena_mut().destroy(pointer).unwrap();
    }
    assert_eq!(
        provider.frees(),
        3,
        "two native exports remain owned until sync"
    );
    context.leave().unwrap();
    assert_eq!(provider.frees(), 5);
    assert_eq!(provider.live_count(), 0);
}

#[test]
fn compiled_gpu_matrix_fixture_uses_provider_memory_and_publishes_without_copy() {
    let _gate = FixtureCudaProvider::test_gate().lock().unwrap();
    let provider = FixtureCudaProvider::shared();
    provider.reset();
    let _selection = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));

    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("gpu_gateway.c");
    std::fs::write(
        &source,
        r#"
#include "mex.h"
#include "gpu/mxGPUArray.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,
             const mxArray *prhs[]) {
if (nrhs != 1 || nlhs != 2) {
    mexErrMsgIdAndTxt("RunMat:test:arity",
                      "expected one input and two outputs");
}
if (mxInitGPU() != MX_GPU_SUCCESS) {
    mexErrMsgIdAndTxt("RunMat:test:gpu", "could not enter GPU context");
}
const mxGPUArray *input = mxGPUCreateFromMxArray(prhs[0]);
if (mxGPUGetDataReadOnly(input) == NULL) {
    mexErrMsgIdAndTxt("RunMat:test:gpu", "input has no device data");
}
mxGPUArray *copy = mxGPUCopyGPUArray(input);
if (mxGPUIsSame(input, copy)) {
    mexErrMsgIdAndTxt("RunMat:test:gpu", "copy retained input identity");
}
double *data = (double *)mxGPUGetData(copy);
data[0] += 5.0;
plhs[0] = mxGPUCreateMxArrayOnGPU(copy);
plhs[1] = mxGPUCreateMxArrayOnCPU(copy);
mxGPUDestroyGPUArray(input);
mxGPUDestroyGPUArray(copy);
}
"#,
    )
    .unwrap();

    let artifact = crate::MexBuild::new(&source, directory.path())
        .api(crate::MexApi::R2018a)
        .compile()
        .unwrap();
    let module = crate::MexModule::load(&artifact.module).unwrap();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[2.0, 4.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let invocation = module
        .invoke(&[Value::GpuTensor(input.clone())], 2, module.api_mode())
        .unwrap();
    let Value::GpuTensor(output) = &invocation.outputs[0] else {
        panic!("first fixture output must stay provider-resident");
    };
    assert!(!output.shares_allocation_with(&input));
    let Value::Tensor(gathered) = &invocation.outputs[1] else {
        panic!("second fixture output must be gathered host data");
    };
    assert_eq!(gathered.materialize_f64(), vec![7.0, 4.0]);
    assert_eq!(provider.context_enters.load(Ordering::Acquire), 1);
    assert_eq!(provider.context_leaves.load(Ordering::Acquire), 1);
    assert_eq!(provider.live_count(), 2);

    drop(input);
    drop(invocation);
    assert_eq!(provider.frees(), 2);
}
