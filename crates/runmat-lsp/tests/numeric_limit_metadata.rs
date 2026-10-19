use lsp_types::Position;
use runmat_lsp::core::analysis::{analyze_document_with_compat, signature_help_at, CompatMode};

#[test]
fn numeric_limit_signatures_come_from_the_canonical_catalog() {
    for (name, source) in [
        ("intmin", "x=intmin(\"like\",int64(1));"),
        ("intmax", "x=intmax(\"uint64\");"),
        ("realmin", "x=realmin(\"like\",single(1));"),
        ("realmax", "x=realmax(\"single\");"),
        ("flintmax", "x=flintmax();"),
    ] {
        let analysis = analyze_document_with_compat(source, CompatMode::RunMat);
        assert!(analysis.syntax_error.is_none(), "{source}");
        assert!(analysis.lowering_error.is_none(), "{source}");
        let position = Position::new(0, source.find(name).expect("builtin") as u32);
        let help = signature_help_at(source, &analysis, &position).expect("signature help");
        let labels = help
            .signatures
            .iter()
            .map(|signature| signature.label.as_str())
            .collect::<Vec<_>>();
        assert!(
            labels.contains(&format!("value = {name}()").as_str()),
            "{name}: {labels:?}"
        );
        assert!(
            labels.contains(&format!("value = {name}(typename)").as_str()),
            "{name}: {labels:?}"
        );
        assert!(
            labels.contains(&format!("value = {name}(\"like\", prototype)").as_str()),
            "{name}: {labels:?}"
        );
    }
}
