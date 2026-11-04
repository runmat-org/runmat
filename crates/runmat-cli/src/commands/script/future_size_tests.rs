use super::*;
use clap::Parser;

// Construction only: no source read, session creation, stdin read, GUI launch
// or runtime poll. These are the two measured oversized ancestor future owners,
// not a claim that every downstream poll frame is bounded.
#[test]
fn script_route_and_request_futures_keep_bounded_inline_state() {
    const MAX_ROUTE_INLINE_STATE: usize = 32 * 1024;
    // Unlike routing glue, the request retains the owned session/controller
    // and response/source/outcome lifecycle across its remaining awaits. The
    // measured matching request is 41,552 bytes after boxing the execution
    // child at its construction owner. Keep bounded headroom for this distinct
    // owner instead of requiring another allocation to fit the route budget.
    // These are inline-state regression limits, not a public stack guarantee:
    // actual GUI execution must still pass on the unchanged default stack.
    const MAX_REQUEST_INLINE_STATE: usize = 64 * 1024;
    let cli = Cli::try_parse_from(["runmat", "qualification.m"]).unwrap();
    let config = RunMatRuntimeConfig::default();
    let graphics = runmat_core::SessionGraphicsHost::Unavailable;
    for path in ["qualification.m", "qualification.fea", "-"] {
        let route = execute_script_with_graphics(
            ScriptInvocation {
                script: PathBuf::from(path),
                args: Vec::new(),
                function: None,
                emit_bytecode_path: None,
                json: false,
                fea_root: crate::cli::FeaRootArgs::default(),
                execution: ExecutionTargetArgs::default(),
            },
            &cli,
            &config,
            &graphics,
        );
        let size = std::mem::size_of_val(&route);
        assert!(
            size <= MAX_ROUTE_INLINE_STATE,
            "script route inline state: {size}"
        );
    }
    let request = execute_script_request(
        PathBuf::from("qualification.m"),
        SourceInput::Text {
            name: "qualification.m".into(),
            text: "disp(1);".into(),
        },
        &cli,
        &config,
        &graphics,
    );
    let size = std::mem::size_of_val(&request);
    assert!(
        size <= MAX_REQUEST_INLINE_STATE,
        "script request inline state: {size}"
    );
}
