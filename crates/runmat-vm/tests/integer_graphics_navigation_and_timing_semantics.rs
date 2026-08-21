#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_mesh_and_patch_properties_preserve_integer_storage() {
    let _plot_guard = runmat_runtime::builtins::plotting::lock_plot_test_context();
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "b=uint64(9007199254740992); z=reshape([b+uint64(1) b+uint64(2) b+uint64(3) b+uint64(4)],[2 2]); hm=mesh(z); zr=get(hm,'ZData'); if ~isa(zr,'uint64') || ~isequal(zr,z); error('mesh integer storage mismatch'); end; f=int16([1 2 3]); v=int16([0 0;1 0;0 1]); hp=patch('Faces',f,'Vertices',v); fr=get(hp,'Faces'); vr=get(hp,'Vertices'); if ~isa(fr,'int16') || ~isequal(fr,f) || ~isa(vr,'int16') || ~isequal(vr,v); error('patch integer storage mismatch'); end;",
    )
    .expect("compiled native graphics properties");
}

#[test]
fn compiled_colormap_pan_and_pause_accept_documented_integer_controls() {
    let _plot_guard = runmat_runtime::builtins::plotting::lock_plot_test_context();
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!(
            "c=parula({constructor}(6)); if ~isa(c,'double') || ~isequal(size(c),[6 3]); error('parula class or shape mismatch'); end; p=pan(); set(p,'Enable',{constructor}(1)); if ~strcmp(get(p,'Enable'),'on'); error('pan enable mismatch'); end; set(p,'Enable',{constructor}(0)); pause({constructor}(0));"
        );
        execute_source(&source).expect("compiled documented integer controls");
    }
}

#[test]
fn compiled_graphics_and_navigation_extensions_have_stable_strict_identifiers() {
    let _matlab = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for (source, identifier) in [
        (
            "ax=gca(); h=patch(uint64(ax),[0 1 0],[0 0 1]);",
            "RunMat:compatibility:PatchIntegerAxesHandleExtension",
        ),
        (
            "f=gcf(); p=pan(uint64(f));",
            "RunMat:compatibility:PanIntegerGraphicsTargetExtension",
        ),
        (
            "opentoline('definitely-missing.m',uint16(1));",
            "RunMat:compatibility:OpentolineIntegerLineExtension",
        ),
        (
            "opentoline('definitely-missing.m',1,uint16(1));",
            "RunMat:compatibility:OpentolineIntegerColumnExtension",
        ),
    ] {
        let error = execute_source(source).expect_err("strict extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_openfig_rejects_integer_filename_roles() {
    for constructor in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let source = format!("h=openfig({constructor}(1));");
        let error = execute_source(&source).expect_err("openfig filename must be text");
        assert_eq!(
            error.identifier(),
            Some("RunMat:figurePersistence:InvalidArgument"),
            "{constructor}"
        );
    }
}

#[test]
fn compiled_timer_title_toeplitz_trace_and_tpdf_semantics_are_exact() {
    let _plot_guard = runmat_runtime::builtins::plotting::lock_plot_test_context();
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source(
        "timerVal=tic(); if ~isa(timerVal,'uint64'); error('tic token class'); end; first=toc(timerVal); pause(0.001); second=toc(); if second<first; error('toc consumed latest timer'); end; wide=bitshift(uint64(1),53)+uint64(1); h=title(wide); if ~strcmp(get(h,'String'),'9007199254740993'); error('title integer formatting'); end; T=toeplitz(uint64([wide wide+uint64(1)])); if ~isa(T,'uint64') || T(2,1)~=wide+uint64(1); error('toeplitz exact storage'); end; tr=trace(uint64([1 2;3 4])); if ~isa(tr,'double') || tr~=5; error('trace integer boundary'); end; p=tpdf(uint16(0),uint16(5)); if ~isa(p,'double') || p<=0; error('tpdf integer extension'); end;",
    )
    .expect("compiled timing, text, shape, and statistical semantics");
}
