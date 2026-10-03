use runmat_mir::{
    analysis::analyze_assembly, lowering::lower_assembly, MirCallArg, MirRvalue, MirStmtKind,
};
use runmat_types::CapabilityRequirement;

fn analyze(source: &str) -> runmat_mir::analysis::AnalysisStore {
    let ast = runmat_parser::parse(source).expect("parse capability program");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
        .expect("lower capability HIR");
    let mir = lower_assembly(&hir.assembly).expect("lower capability MIR");
    analyze_assembly(&mir)
}

fn source_services(source: &str) -> (bool, bool) {
    let analysis = analyze(source);
    let function = analysis.functions.first().expect("analyzed function");
    (
        function
            .capabilities
            .0
            .contains(&CapabilityRequirement::Filesystem),
        function
            .capabilities
            .0
            .contains(&CapabilityRequirement::Network),
    )
}

#[test]
fn image_reader_source_literals_reach_function_capability_summaries() {
    // These assertions require the composed typed imread catalog leaf. A
    // literal extractor alone cannot prove the function summary or manifest.
    assert_eq!(
        source_services("function y = read_local(); y = imread('pixels.png'); end"),
        (true, false)
    );
    assert_eq!(
        source_services(
            "function y = read_file_url(); y = imread('file:///images/pixels.png'); end"
        ),
        (true, false)
    );
    assert_eq!(
        source_services(
            "function y = read_http(); y = imread('https://example.test/pixels.png'); end"
        ),
        (false, true)
    );
    assert_eq!(
        source_services(
            "function y = read_alias(); p = 'https://example.test/pixels.png'; y = imread(p); end"
        ),
        (false, true)
    );
    // The call must observe p's source literal before assignment overwrites p.
    assert_eq!(
        source_services("function p = read_reused_target(); p = 'https://example.test/pixels.png'; p = imread(p); end"),
        (false, true)
    );
}

#[test]
fn dynamic_and_expanded_image_reader_sources_keep_both_services() {
    assert_eq!(
        source_services("function y = read_dynamic(path); y = imread(path); end"),
        (true, true)
    );
    let source = "function y = read_expanded(varargin); y = imread(varargin{:}); end";
    let ast = runmat_parser::parse(source).expect("parse expanded source");
    let hir =
        runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower expanded HIR");
    let mir = lower_assembly(&hir.assembly).expect("lower expanded MIR");
    let call = mir
        .bodies
        .values()
        .flat_map(|body| &body.blocks)
        .flat_map(|block| &block.statements)
        .find_map(|statement| match &statement.kind {
            MirStmtKind::Assign {
                value: MirRvalue::Call(call),
                ..
            }
            | MirStmtKind::Expr(MirRvalue::Call(call)) => Some(call),
            _ => None,
        })
        .expect("expanded image reader call");
    assert!(call
        .args
        .iter()
        .any(|arg| !matches!(arg, MirCallArg::Single(_))));
    let analysis = analyze_assembly(&mir);
    let function = analysis
        .functions
        .first()
        .expect("analyzed expanded function");
    assert!(function
        .capabilities
        .0
        .contains(&CapabilityRequirement::Filesystem));
    assert!(function
        .capabilities
        .0
        .contains(&CapabilityRequirement::Network));
}
