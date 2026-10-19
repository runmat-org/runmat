use super::*;

#[test]
fn tokenizes_minimal_study_losslessly() {
    let source = r#"// one study
geometry bracket {
    file = "./bracket.step"
    units = mm
}
study static_case {
    geometry = geometry.bracket
    output peak = max(output.stress, over = geometry.bracket.region.body)
}
"#;
    let stream = tokenize_fea(source);
    assert!(stream.diagnostics.is_empty(), "{:?}", stream.diagnostics);
    let reconstructed = stream
        .tokens
        .iter()
        .filter(|token| token.kind != FeaTokenKind::EndOfFile)
        .map(|token| stream.lexeme(*token))
        .collect::<String>();
    assert_eq!(reconstructed, source);
    assert!(stream
        .tokens
        .iter()
        .any(|token| token.kind == FeaTokenKind::Keyword(FeaKeyword::Geometry)));
}

#[test]
fn distinguishes_ranges_decimals_exponents_and_units() {
    let stream = tokenize_fea("workers = 1..64\ntol = 1e-8\ndensity = 7850 kg/m^3\n");
    assert!(stream.diagnostics.is_empty(), "{:?}", stream.diagnostics);
    let kinds = stream
        .significant_tokens()
        .map(|token| token.kind)
        .collect::<Vec<_>>();
    assert!(kinds.contains(&FeaTokenKind::Range));
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| **kind == FeaTokenKind::Number)
            .count(),
        5
    );
}

#[test]
fn reports_bad_input_without_losing_following_tokens() {
    let stream = tokenize_fea("name = \"unterminated\nnext = @\n");
    assert_eq!(stream.diagnostics.len(), 2);
    assert_eq!(
        stream.diagnostics[0].kind,
        FeaLexDiagnosticKind::UnterminatedString
    );
    assert_eq!(
        stream.diagnostics[1].kind,
        FeaLexDiagnosticKind::InvalidCharacter
    );
    assert!(stream
        .significant_tokens()
        .any(|token| stream.lexeme(token) == "next"));
}

#[test]
fn accepts_unicode_identifiers_and_both_string_quotes() {
    let stream = tokenize_fea("material álpha { name = 'steel' }\n");
    assert!(stream.diagnostics.is_empty());
    assert!(stream
        .significant_tokens()
        .any(|token| stream.lexeme(token) == "álpha"));
}
