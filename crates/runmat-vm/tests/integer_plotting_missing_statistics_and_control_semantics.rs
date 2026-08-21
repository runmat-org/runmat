#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

#[test]
fn compiled_integer_plot_sources_and_defaults_remain_exact() {
    let _plot_guard = runmat_runtime::builtins::plotting::lock_plot_test_context();
    execute_source("wide=uint64(9007199254740992)+uint64(1); x=[wide wide+uint64(1)]; y=int16([-2 3]); hs=stairs(x,y); sx=get(hs,'XData'); sy=get(hs,'YData'); if ~isa(sx,'uint64') || sx(1)~=wide || ~isa(sy,'int16'); error('stairs typed source'); end; hm=stem(x,y); mx=get(hm,'XData'); my=get(hm,'YData'); if ~isa(mx,'uint64') || mx(2)~=wide+uint64(1) || ~isa(my,'int16'); error('stem typed source'); end; opts=statset(); d=statget(opts,'TolFun',wide); if ~isa(d,'uint64') || d~=wide; error('statget typed default'); end;")
        .expect("compiled exact plotting and statistics value carriers");
}

#[test]
fn compiled_documented_integer_indicator_is_available_in_compatibility_mode() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    execute_source("a=standardizeMissing([-99 2],int16(-99)); if ~isnan(a(1)) || a(2)~=2; error('integer indicator'); end;")
        .expect("documented integer indicator");
}

#[test]
fn compiled_integer_missing_statistics_and_control_extensions_are_gated() {
    for (source, identifier) in [
        (
            "y=standardizeMissing(uint8([1 2]),uint8(1));",
            "RunMat:compatibility:StandardizeMissingIntegerDataExtension",
        ),
        (
            "o=statset('MaxIter',uint8(2));",
            "RunMat:compatibility:StatsetIntegerOptionValueExtension",
        ),
        (
            "s=tf(1,[1 1]); y=step(s,uint8(2));",
            "RunMat:compatibility:StepIntegerTimeExtension",
        ),
        (
            "i=stepinfo(uint8([0 1]),[0 1]);",
            "RunMat:compatibility:StepinfoIntegerSampledDataExtension",
        ),
    ] {
        let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
        let error = execute_source(source).expect_err("typed extension must be gated");
        assert_eq!(error.identifier(), Some(identifier), "{source}");
    }
}

#[test]
fn compiled_runmat_integer_control_extensions_use_checked_floating_boundaries() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    execute_source("a=standardizeMissing(uint8([1 2]),uint8(1)); if ~isa(a,'uint8') || ~isequal(a,uint8([1 2])); error('integer missing no-op'); end; o=statset('MaxIter',uint16(7)); if o.MaxIter~=7; error('integer statset'); end; s=tf(1,[1 1]); [y,t]=step(s,uint16(2)); if ~isa(y,'double') || ~isa(t,'double') || t(end)~=2; error('integer step time'); end; i=stepinfo(uint16([0 1 1]),uint16([0 1 2]),uint16(1)); if i.SteadyStateValue~=1; error('integer stepinfo samples'); end;")
        .expect("compiled RunMat integer control extensions");
}
