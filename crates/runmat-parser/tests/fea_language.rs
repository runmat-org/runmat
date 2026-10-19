use runmat_parser::fea::{format_fea, parse_fea, FeaFormatOptions};

const COMPLETE_SURFACE: &str = include_str!("fea/complete_surface.fea");

#[test]
fn complete_fea_language_fixture_parses_without_recovery() {
    let module = parse_fea(COMPLETE_SURFACE);
    assert!(module.diagnostics.is_empty(), "{:#?}", module.diagnostics);
    assert_eq!(module.items.len(), 14);
}

#[test]
fn complete_fea_language_fixture_formats_idempotently() {
    let first = format_fea(COMPLETE_SURFACE, FeaFormatOptions::default());
    assert!(first.diagnostics.is_empty(), "{:#?}", first.diagnostics);
    let second = format_fea(&first.source, FeaFormatOptions::default());
    assert!(second.diagnostics.is_empty(), "{:#?}", second.diagnostics);
    assert_eq!(second.source, first.source);
}
