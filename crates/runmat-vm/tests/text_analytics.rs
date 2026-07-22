use runmat_builtins::{StringArray, Value};

#[path = "support/mod.rs"]
mod test_helpers;
use test_helpers::execute_source;

fn has_object_class(vars: &[Value], class_name: &str) -> bool {
    vars.iter()
        .any(|value| matches!(value, Value::Object(object) if object.class_name == class_name))
}

fn has_num(vars: &[Value], expected: f64) -> bool {
    vars.iter()
        .any(|value| matches!(value, Value::Num(value) if (*value - expected).abs() < 1.0e-8))
}

fn has_text(vars: &[Value], expected: &str) -> bool {
    vars.iter().any(|value| match value {
        Value::String(text) => text == expected,
        Value::StringArray(StringArray { data, .. }) => data.iter().any(|text| text == expected),
        _ => false,
    })
}

#[test]
fn text_analytics_surface_executes_from_scripts() {
    let vars = execute_source(
        r#"
        docs = tokenizedDocument(["red blue red"; "blue green"]);
        bag = bagOfWords(docs);
        enc = wordEncoding(["red" "blue" "green"]);
        idx = word2ind(enc, "blue");
        idx_scalar = idx(1);
        txt = extractHTMLText("<p>Hello <b>RunMat</b></p>");
        sent = tokenizedDocument("good bad");
        compound = vaderSentimentScores(sent);
        score = compound(1);
        "#,
    )
    .expect("text analytics script");

    assert!(has_object_class(&vars, "tokenizedDocument"));
    assert!(has_object_class(&vars, "bagOfWords"));
    assert!(has_object_class(&vars, "wordEncoding"));
    assert!(has_num(&vars, 2.0));
    assert!(has_text(&vars, "Hello RunMat"));
}
