#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_trigonometric_integer_extensions_are_gated() {
    for (source, identifier) in [
        (
            "y=sin(uint8(1));",
            "RunMat:compatibility:SinIntegerInputExtension",
        ),
        (
            "y=sind(uint8(30));",
            "RunMat:compatibility:SindIntegerInputExtension",
        ),
        (
            "y=sinh(uint8(1));",
            "RunMat:compatibility:SinhIntegerInputExtension",
        ),
        (
            "y=sinpi(uint8(1));",
            "RunMat:compatibility:SinpiIntegerInputExtension",
        ),
        (
            "y=sinc(uint8(1));",
            "RunMat:compatibility:SincNonfloatingInputExtension",
        ),
        (
            "y=sawtooth(uint8(1));",
            "RunMat:compatibility:SawtoothNondoubleInputExtension",
        ),
    ] {
        let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        let error = execute_source(source).expect_err("typed input must be gated");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_integer_signal_cast_and_size_paths_preserve_their_contracts() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source("base=uint64(9007199254740992); wide=base+uint64(1); z=sinpi([wide uint64(0)]); if ~isequal(z,[0 0]); error('sinpi wide integer identity'); end; s=single(wide); if ~isa(s,'single'); error('single class'); end; g=sign(int64([-9223372036854775808 0 7])); if ~isa(g,'int64') || ~isequal(g,int64([-1 0 1])); error('sign class or value'); end; a=reshape(uint8(1:6),[2 3]); e=size(a,[]); if ~isa(e,'double') || ~isequal(size(e),[1 0]); error('size empty dimensions'); end; if size(a,uint8(2)) ~= 3; error('size integer dimension'); end;")
        .expect("compiled exact signal, cast, sign, and size contracts");
}

#[test]
fn compiled_sparse_integer_roles_separate_documented_structure_from_extension_values() {
    {
        let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        execute_source("s=sparse(uint64([1 2]),uint64([1 2]),[3 4]); if ~isequal(full(s),[3 0;0 4]); error('sparse integer subscripts'); end;")
            .expect("documented typed integer subscripts");
        let error = execute_source("s=sparse(uint8(1),uint8(1),uint16(7));")
            .expect_err("typed integer sparse values are a RunMat extension");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:SparseIntegerExtension")
        );
    }
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source("s=sparse(uint8(1),uint8(1),uint16(7)); v=nonzeros(s); if ~isa(v,'uint16') || v ~= uint16(7); error('typed sparse value'); end;")
        .expect("RunMat typed sparse values");
}

#[test]
fn compiled_struct_preference_and_numeric_title_values_remain_exact() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source("base=uint64(9007199254740992); wide=base+uint64(1); s=setfield(struct(),'value',wide); if ~isa(s.value,'uint64') || s.value ~= wide; error('setfield integer payload'); end; setpref('integer-slice','wide',wide); p=getpref('integer-slice','wide'); if ~isa(p,'uint64') || p ~= wide; error('setpref integer payload'); end;")
        .expect("compiled structure and preference payload preservation");
}
