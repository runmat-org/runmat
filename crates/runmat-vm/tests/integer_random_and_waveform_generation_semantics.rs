#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_random_size_controls_accept_every_integer_class() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "a=rand({constructor}(2)); if ~isequal(size(a),[2 2]); error('rand size'); end; b=randn({constructor}(2)); if ~isequal(size(b),[2 2]); error('randn size'); end; c=randi(3,{constructor}(2)); if ~isequal(size(c),[2 2]); error('randi size'); end; d=randperm({constructor}(2)); if ~isequal(size(d),[1 2]); error('randperm size'); end;"
        );
        execute_source(&source).unwrap_or_else(|error| panic!("{constructor}: {error:?}"));
    }
}

#[test]
fn compiled_rng_accepts_public_integer_seed_classes_and_gates_wide_seeds() {
    {
        let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        for constructor in [
            "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        ] {
            execute_source(&format!("rng({constructor}(1));"))
                .unwrap_or_else(|error| panic!("{constructor}: {error:?}"));
        }
        let error = execute_source("rng(uint64(4294967296));")
            .expect_err("strict mode rejects seeds outside the documented domain");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:RngWideSeedExtension")
        );
    }
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source("seed=bitshift(uint64(1),53)+uint64(1); rng(seed); s=rng(); if ~isa(s.Seed,'uint64') || s.Seed~=seed; error('wide seed lost'); end;")
        .expect("compiled RunMat mode preserves wide integer seed state");
}

#[test]
fn compiled_runmat_extensions_cover_waveforms_distributions_and_exact_sampling() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "p=rectpuls(uint8([0 1]),uint8(2)); if ~isequal(p,[1 0]); error('rectpuls edge'); end; q=pulstran(uint8([0 1]),uint8(0),'rectpuls',uint8(2)); if ~isequal(q,[1 0]); error('pulstran integer form'); end; r=random('Normal',uint8(0),uint8(1),uint8(2)); if ~isequal(size(r),[2 2]); error('random integer form'); end; base=bitshift(uint64(1),53); s=randsample(uint64([base base+uint64(1)]),uint8(2)); if ~isa(s,'uint64') || numel(s)~=2; error('randsample integer form'); end;",
    )
    .expect("compiled RunMat integer extensions");
}

#[test]
fn compiled_strict_mode_rejects_waveform_and_column_size_extensions() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for source in [
        "y=rectpuls(uint8([0 1]));",
        "y=randn(uint8([2;3]));",
        "y=random('Normal',uint8(0),1);",
        "y=randsample(uint8([1 2]),1);",
    ] {
        let error = execute_source(source).expect_err("strict extension must reject");
        assert!(
            error.to_string().contains("compatibility"),
            "{source}: {error:?}"
        );
    }
}
