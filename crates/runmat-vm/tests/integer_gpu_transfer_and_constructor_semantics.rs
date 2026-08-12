#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_integer_constructor_controls_preserve_shape_and_class() {
    execute_source(
        "a = ones(int8(-2),uint16(3),uint64(1),'uint32'); if ~isa(a,'uint32') || ~isequal(size(a),[0 3]); error('ones integer construction failed'); end; b = inf(uint8(2),uint16(3),uint32(1)); if ~isa(b,'double') || ~isequal(size(b),[2 3]); error('inf construction failed'); end; c = nan(int16(2),uint64(1)); if ~isa(c,'double') || ~isequal(size(c),[2 1]); error('nan construction failed'); end; d = rand(uint32(1),uint8(2)); if ~isa(d,'double') || ~isequal(size(d),[1 2]); error('rand construction failed'); end; p = uint64(9); q = ones('like',p); if ~isa(q,'uint64') || ~isequal(size(q),[1 1]) || q ~= uint64(1); error('ones like scalar failed'); end;",
    )
    .expect("compiled constructor integer semantics");
}

#[test]
fn compiled_constructor_extensions_reject_in_matlab_mode() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "x = inf(uint16([2;3]));",
            "RunMat:compatibility:InfColumnSizeVectorExtension",
        ),
        (
            "x = nan(uint16([2;3]));",
            "RunMat:compatibility:NanColumnSizeVectorExtension",
        ),
        (
            "x = ones(uint16([2;3]));",
            "RunMat:compatibility:OnesColumnSizeVectorExtension",
        ),
        (
            "x = rand(uint16([2;3]));",
            "RunMat:compatibility:RandColumnSizeVectorExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("column size vector must be gated");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_integer_gpuarray_intent_and_gather_roundtrip() {
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let provider = runmat_accelerate_api::provider().expect("test provider");
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    execute_source(
        "x = uint64(9007199254740993); g = gpuArray(x); if ~isgpuarray(g); error('explicit integer gpuArray intent lost'); end; y = gather(g); if ~isa(y,'uint64') || y ~= x; error('integer gather roundtrip failed'); end; if isgpuarray(y); error('gathered host value reported resident'); end;",
    )
    .expect("compiled integer gpuArray transfer semantics");
}

#[test]
fn compiled_gpudevice_facade_is_gated_in_matlab_mode() {
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let provider = runmat_accelerate_api::provider().expect("test provider");
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let error = execute_source("d = gpuDevice(uint8(1));")
        .expect_err("RunMat provider-info facade must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:GpuDeviceProviderInfoExtension")
    );
}
