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

#[test]
fn compiled_typed_empty_construction_and_observers_preserve_class_and_shape() {
    for class in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "a={class}.empty(2,0,3); if ~isa(a,'{class}') || ~isempty(a) || numel(a)~=0 || ~isequal(size(a),[2 0 3]); error('typed empty observer mismatch'); end; b=reshape(a,0,6); if ~isa(b,'{class}') || ~isequal(size(b),[0 6]); error('typed empty reshape mismatch'); end; c=cat(1,a,{class}.empty(4,0,3)); if ~isa(c,'{class}') || ~isequal(size(c),[6 0 3]); error('typed empty concatenation mismatch'); end;"
        );
        execute_source(&source).unwrap_or_else(|error| panic!("{class}: {error}"));
    }
}

#[test]
fn compiled_empty_concatenation_uses_only_nonempty_inputs_for_class_selection() {
    execute_source(
        "a=uint16.empty(0,4); b=uint8([1 2 3;4 5 6]); c=cat(1,a,b); if ~isa(c,'uint8') || ~isequal(c,b); error('empty class omission mismatch'); end; d=cat(2,int16.empty(0,2),uint8.empty(0,3)); if ~isa(d,'int16') || ~isequal(size(d),[0 5]); error('all-empty class selection mismatch'); end;",
    )
    .expect("compiled typed-empty concatenation semantics");
}
