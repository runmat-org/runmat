use super::super::{dependencies, from_listings, ModuleScope};
use super::relative;
use crate::syntax::{parse_assembly_in, parse_drawing_in};

fn part(name: &str) -> String {
    format!("runmat 0.7\nstd 1\n\npart {name} {{\n}}\n")
}

fn scope(files: &[(&str, &str)]) -> ModuleScope {
    ModuleScope::from_files(files.iter().copied())
}

/// A drawing in `drawings/` whose view shows the model `model`.
fn drawing(model: &str) -> String {
    format!(
        "runmat 0.7\nstd 1\n\ndrawing l_bracket {{\n    sheet \"1\" = {{ format = a3, scale = [1, 1], projection = third_angle }}\n    view front = standard(orientation = front, sheet = \"1\", model = {model}, at = [150, 165], display = hidden_lines)\n}}\n"
    )
}

fn drawing_messages(model: &str, extra: &[(&str, &str)]) -> (Vec<String>, Option<String>) {
    let text = drawing(model);
    let bracket = part("l_bracket");
    let mut files = vec![
        ("/p/l_bracket.part", bracket.as_str()),
        ("/p/drawings/l_bracket.drawing", text.as_str()),
    ];
    files.extend_from_slice(extra);
    let parsed = parse_drawing_in("/p/drawings/l_bracket.drawing", &text, &scope(&files));
    let messages = parsed
        .diagnostics
        .iter()
        .map(|d| d.message.clone())
        .collect();
    let file = parsed
        .document
        .and_then(|d| d.views.first().map(|v| v.model.file.as_str().to_string()));
    (messages, file)
}

#[test]
fn a_drawing_in_a_subfolder_names_its_part_by_file_or_by_folder() {
    for model in [
        r#"{ part = l_bracket, from = "../l_bracket.part" }"#,
        r#"{ from = "../l_bracket.part" }"#,
        r#"{ part = l_bracket, from = ".." }"#,
    ] {
        let (messages, file) = drawing_messages(model, &[]);
        assert!(messages.is_empty(), "{model}: {messages:?}");
        assert_eq!(file.as_deref(), Some("/p/l_bracket.part"), "{model}");
    }
}

#[test]
fn a_part_name_the_file_does_not_declare_lists_the_parts_it_holds() {
    let (messages, _) = drawing_messages(r#"{ part = flange, from = "../l_bracket.part" }"#, &[]);
    assert_eq!(
        messages,
        ["no `part flange` in `../l_bracket.part`; it holds part l_bracket"]
    );
}

#[test]
fn a_file_of_another_kind_says_what_it_holds() {
    let rig = "runmat 0.7\nstd 1\n\nassembly rig {\n}\n";
    let (messages, _) = drawing_messages(
        r#"{ part = rig, from = "../rig.assembly" }"#,
        &[("/p/rig.assembly", rig)],
    );
    assert_eq!(messages, ["`../rig.assembly` holds assembly rig, not a part"]);
}

#[test]
fn a_missing_path_says_so_and_suggests_the_nearest_real_one() {
    // The agent's first try wrote the file as if it sat beside the drawing.
    let (messages, _) = drawing_messages(r#"{ part = l_bracket, from = "./l_bracket.part" }"#, &[]);
    assert_eq!(
        messages,
        ["`./l_bracket.part` is neither a design file nor a folder in the workspace; did you mean `../l_bracket.part`?"]
    );
    let (messages, _) = drawing_messages(r#"{ part = l_bracket, from = "../l_braket.part" }"#, &[]);
    assert_eq!(
        messages,
        ["`../l_braket.part` is neither a design file nor a folder in the workspace; did you mean `../l_bracket.part`?"]
    );
    let flange = part("flange");
    let (messages, _) = drawing_messages(
        r#"{ part = flange, from = "../flanges" }"#,
        &[("/p/flange/flange.part", flange.as_str())],
    );
    assert_eq!(
        messages,
        ["the folder `../flanges` holds no designs (or isn't in the workspace); did you mean `../flange`?"]
    );
    let (messages, _) = drawing_messages(r#"{ part = l_bracket, from = "../nowhere" }"#, &[]);
    assert_eq!(
        messages,
        ["the folder `../nowhere` holds no designs (or isn't in the workspace)"]
    );
}

#[test]
fn an_assembly_uses_a_part_from_its_file() {
    let text = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part bracket from \"parts/bracket.part\" as b\n    use part bracket from \"./parts\" as c\n}\n";
    let bracket = part("bracket");
    let files = [
        ("/p/rig.assembly", text),
        ("/p/parts/bracket.part", bracket.as_str()),
    ];
    let parsed = parse_assembly_in("/p/rig.assembly", text, &scope(&files));
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let doc = parsed.document.unwrap();
    for o in &doc.occurrences {
        assert_eq!(o.source.definition.file.as_str(), "/p/parts/bracket.part");
    }
    // The host lists the folder and reads the file.
    assert!(dependencies("/p/rig.assembly", text).contains(&"/p/parts/bracket.part".to_string()));
    assert_eq!(
        from_listings("/p/rig.assembly", "parts/bracket.part"),
        ["/p/parts/bracket.part"]
    );
}

#[test]
fn file_form_uses_count_toward_folder_cycles() {
    let a = "runmat 0.7\nstd 1\n\nassembly a {\n    use assembly b from \"../b/b.assembly\" as b\n}\n";
    let b = "runmat 0.7\nstd 1\n\nassembly b {\n    use assembly a from \"../a/a.assembly\" as a\n}\n";
    let s = scope(&[("/p/a/a.assembly", a), ("/p/b/b.assembly", b)]);
    assert!(!s.problems().is_empty(), "the cycle is found");
}

#[test]
fn relative_paths_read_from_the_folder() {
    assert_eq!(relative("/p/drawings", "/p/l_bracket.part"), "../l_bracket.part");
    assert_eq!(relative("/p", "/p/cabin"), "./cabin");
    assert_eq!(relative("/p", "/p"), ".");
}
