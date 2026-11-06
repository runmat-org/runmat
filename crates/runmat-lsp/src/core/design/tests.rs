use super::*;

const BRACKET: &str = include_str!("../../../../runmat-cad/tests/corpus/documents/bracket.part");
const ASSEMBLY: &str =
    include_str!("../../../../runmat-cad/tests/corpus/documents/synth0.assembly");
const DRAWING: &str = include_str!("../../../../runmat-cad/tests/corpus/documents/rig.drawing");
const CIRCUIT: &str = include_str!("../../../../runmat-circuit/tests/projects/io/io.circuit");
const LAYOUT: &str = include_str!("../../../../runmat-circuit/tests/projects/io/io.circuit.layout");
const LOCK: &str = include_str!("../../../../runmat-circuit/tests/projects/io/io.circuit.lock");

fn uri(path: &str) -> Url {
    Url::parse(&format!("file://{path}")).unwrap()
}

fn analyze(path: &str, text: &str, files: &DesignFiles) -> DesignDocumentAnalysis {
    let uri = uri(path);
    let dialect = design_dialect(&uri).expect("a design file");
    DesignDocumentAnalysis::analyze(&uri, dialect, text, files)
}

fn circuit_files() -> DesignFiles {
    DesignFiles::from([
        ("/p/io.circuit".to_string(), CIRCUIT.to_string()),
        ("/p/io.circuit.lock".to_string(), LOCK.to_string()),
        ("/p/io.circuit.layout".to_string(), LAYOUT.to_string()),
    ])
}

/// The position of the `nth` occurrence of `needle`, `into` characters in.
fn at(text: &str, needle: &str, nth: usize, into: usize) -> Position {
    let offset = text
        .match_indices(needle)
        .nth(nth)
        .unwrap_or_else(|| panic!("{needle} #{nth} not in the text"))
        .0
        + into;
    offset_to_position(text, offset)
}

fn hover_text(analysis: &DesignDocumentAnalysis, position: Position) -> String {
    match analysis.hover(&position).expect("a hover").contents {
        HoverContents::Markup(m) => m.value,
        other => panic!("{other:?}"),
    }
}

fn labels(items: &[CompletionItem]) -> Vec<&str> {
    items.iter().map(|i| i.label.as_str()).collect()
}

#[test]
fn every_design_file_type_gets_tokens_and_diagnostics() {
    let files = circuit_files();
    for (path, text) in [
        ("/p/bracket.part", BRACKET),
        ("/p/synth0.assembly", ASSEMBLY),
        ("/p/rig.drawing", DRAWING),
        ("/p/io.circuit", CIRCUIT),
        ("/p/io.circuit.layout", LAYOUT),
    ] {
        let analysis = analyze(path, text, &files);
        let lexical = DesignDocumentAnalysis::lexical_semantic_tokens(analysis.dialect(), text);
        let tokens = analysis.semantic_tokens();
        assert!(
            lexical.data.len() > 20,
            "{path}: {} lexical tokens",
            lexical.data.len()
        );
        assert!(tokens.data.len() >= lexical.data.len(), "{path}");
        assert!(
            tokens.data.iter().any(|t| t.token_type == KEYWORD),
            "{path}: no keyword tokens"
        );
        assert!(
            tokens
                .data
                .iter()
                .any(|t| t.token_modifiers_bitset & DECLARATION != 0),
            "{path}: no declared names marked"
        );
        // Tokens never overlap and stay in order.
        let (mut line, mut start, mut end) = (0u32, 0u32, 0u32);
        for t in &tokens.data {
            if t.delta_line > 0 {
                line += t.delta_line;
                start = t.delta_start;
            } else {
                start += t.delta_start;
                assert!(start >= end, "{path}: overlapping tokens on line {line}");
            }
            end = start + t.length;
        }
        assert!(
            !analysis.document_symbols().is_empty(),
            "{path}: no symbols"
        );
        let _ = analysis.diagnostics();
    }
}

#[test]
fn a_part_reports_regeneration_problems_where_they_are_written() {
    let analysis = analyze("/p/bracket.part", BRACKET, &DesignFiles::new());
    let diagnostics = analysis.diagnostics();
    let missing = at(BRACKET, "feature round_missing", 0, 0).line;
    assert!(
        diagnostics.iter().any(|d| d.range.start.line >= missing
            && d.range.start.line <= missing + 3
            && d.source.as_deref() == Some("runmat-cad")),
        "{diagnostics:#?}"
    );
    assert!(analysis.status_message().contains("error"));
}

#[test]
fn a_typed_error_shows_as_a_syntax_diagnostic() {
    let broken = BRACKET.replace("feature body = extrude(", "feature body = extrude((");
    let analysis = analyze("/p/bracket.part", &broken, &DesignFiles::new());
    assert!(analysis
        .diagnostics()
        .iter()
        .any(|d| d.code == Some(NumberOrString::String("design.syntax".into()))));
}

#[test]
fn hover_shows_resolved_values_with_units() {
    let analysis = analyze("/p/bracket.part", BRACKET, &DesignFiles::new());
    let text = hover_text(&analysis, at(BRACKET, "input.width", 0, 7));
    assert!(text.contains("input width : length = 80 mm"), "{text}");
    assert!(text.contains("80 mm (0.08 m)"), "{text}");
    let feature = hover_text(&analysis, at(BRACKET, "feature.body", 0, 9));
    assert!(feature.contains("extrude"), "{feature}");
}

#[test]
fn part_completion_offers_features_topology_and_roles() {
    let analysis = analyze("/p/bracket.part", BRACKET, &DesignFiles::new());
    let after_feature = at(BRACKET, "feature.body.face.end", 0, "feature.".len());
    let items = analysis.completion_items(&after_feature);
    for name in ["body", "cut", "h", "round"] {
        assert!(labels(&items).contains(&name), "{:?}", labels(&items));
    }
    let topology = analysis.completion_items(&at(
        BRACKET,
        "feature.body.face.end",
        0,
        "feature.body.".len(),
    ));
    assert!(
        labels(&topology).contains(&"face"),
        "{:?}",
        labels(&topology)
    );
    let roles = analysis.completion_items(&at(
        BRACKET,
        "feature.body.face.end",
        0,
        "feature.body.face.".len(),
    ));
    assert!(
        labels(&roles).contains(&"end") && labels(&roles).contains(&"side"),
        "{:?}",
        labels(&roles)
    );
    let inputs = analysis.completion_items(&at(BRACKET, "input.width", 0, "input.".len()));
    assert_eq!(labels(&inputs), vec!["depth", "thickness", "width"]);
    let entities = analysis.completion_items(&at(BRACKET, "s1.l1)", 0, "s1.".len()));
    assert!(labels(&entities).contains(&"l4"), "{:?}", labels(&entities));
    let units = analysis.completion_items(&at(BRACKET, "80 mm", 0, 3));
    assert!(labels(&units).contains(&"mm"));
}

#[test]
fn definition_references_and_rename_follow_the_names() {
    let analysis = analyze("/p/bracket.part", BRACKET, &DesignFiles::new());
    let use_site = at(BRACKET, "sketch.s1", 0, "sketch.".len());
    let definition = analysis.definition(&use_site).expect("a definition");
    assert_eq!(
        definition.range.start,
        at(BRACKET, "sketch s1 on", 0, "sketch ".len())
    );
    // `input.width`: the declaration and four uses.
    let refs = analysis.references(&at(BRACKET, "input width", 0, 6), true);
    assert_eq!(refs.len(), 5, "{refs:#?}");
    let edit = analysis
        .rename(&at(BRACKET, "input.width", 0, 7), "span")
        .unwrap()
        .expect("an edit");
    let edits = edit.changes.unwrap().into_values().next().unwrap();
    assert_eq!(edits.len(), 5);
    assert!(edits.iter().all(|e| e.new_text == "span"));
    // A sketch entity through its sketch: `s1.l1`.
    let entity = analysis.definition(&at(BRACKET, "s1.l1)", 0, 3)).unwrap();
    assert_eq!(entity.range.start, at(BRACKET, "l1 = line", 0, 0));
    assert!(analysis
        .rename(&at(BRACKET, "input.width", 0, 7), "depth")
        .is_err());
}

#[test]
fn a_use_path_goes_to_its_file() {
    let analysis = analyze("/p/io.circuit.layout", LAYOUT, &circuit_files());
    let location = analysis
        .definition(&at(LAYOUT, "\"./io.circuit\"", 0, 3))
        .expect("the circuit file");
    assert_eq!(location.uri.path(), "/p/io.circuit");
}

#[test]
fn circuit_hover_shows_parts_instances_and_pins() {
    let analysis = analyze("/p/io.circuit", CIRCUIT, &circuit_files());
    let errors: Vec<_> = analysis
        .diagnostics()
        .into_iter()
        .filter(|d| d.severity == Some(DiagnosticSeverity::ERROR))
        .collect();
    assert!(errors.is_empty(), "{errors:#?}");
    let u = hover_text(
        &analysis,
        at(CIRCUIT, "component U =", 0, "component ".len()),
    );
    assert!(u.contains("OPA4171"), "{u}");
    assert!(u.contains("Instances"), "{u}");
    let gnd = hover_text(&analysis, at(CIRCUIT, "net.GND", 0, 5));
    assert!(gnd.contains("pins"), "{gnd}");
    // `"RS.1"` names component RS.
    let rs = analysis.definition(&at(CIRCUIT, "\"RS.1\"", 0, 1)).unwrap();
    assert_eq!(
        rs.range.start,
        at(CIRCUIT, "component RS", 0, "component ".len())
    );
    let parts = analysis.completion_items(&at(CIRCUIT, "part = OPA4171", 0, "part = ".len()));
    assert!(
        labels(&parts).contains(&"OPA4171"),
        "{:?}",
        &labels(&parts)[..10.min(parts.len())]
    );
    let nets = analysis.completion_items(&at(CIRCUIT, "net.IN,", 0, "net.".len()));
    assert!(labels(&nets).contains(&"GND"));
}

#[test]
fn a_layout_names_nets_from_its_tracks() {
    let analysis = analyze("/p/io.circuit.layout", LAYOUT, &circuit_files());
    let refs = analysis.references(&at(LAYOUT, "net GND", 0, 4), false);
    assert!(refs.len() > 10, "{}", refs.len());
    let u1 = hover_text(
        &analysis,
        at(LAYOUT, "footprint U1 ", 0, "footprint ".len()),
    );
    assert!(u1.contains("OPA4171"), "{u1}");
}

#[test]
fn formatting_is_stable_on_formatted_sources() {
    let analysis = analyze("/p/bracket.part", BRACKET, &DesignFiles::new());
    let edits = analysis.formatting_edits();
    if let Some(edit) = edits.first() {
        let again = analyze("/p/bracket.part", &edit.new_text, &DesignFiles::new());
        assert!(again.formatting_edits().is_empty());
    }
}

#[test]
fn workspace_symbols_name_declarations_by_kind() {
    let analysis = analyze("/p/bracket.part", BRACKET, &DesignFiles::new());
    let all = analysis.workspace_symbols(None);
    let names: Vec<&str> = all.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"feature.body"), "{names:?}");
    assert!(names.contains(&"input.width"), "{names:?}");
    let found = analysis.workspace_symbols(Some("WIDTH"));
    assert!(
        !found.is_empty()
            && found
                .iter()
                .all(|s| s.name.to_ascii_lowercase().contains("width"))
    );
}

#[test]
fn circuit_simulation_words_and_units_color_and_complete() {
    let v = highlight::dialect_vocabulary(Dialect::Circuit);
    // Analysis and measure kinds read as functions; a measure's `spec` as a clause.
    for word in ["noise", "temperature", "monte_carlo", "tran", "output_noise", "noise_density", "rise_time"] {
        assert_eq!(v.words.get(word).map(|c| c.token_type), Some(FUNCTION), "{word}");
    }
    assert_eq!(v.words.get("spec").map(|c| c.token_type), Some(KEYWORD));
    // Every SI symbol under every prefix the circuit reader takes, and its named units.
    for unit in ["uA", "nV", "nF", "pF", "kohm", "MHz", "degC", "dB", "ppm", "mil"] {
        assert!(v.units.contains(unit), "{unit}");
    }
}

/// A design named by symbol opens its declaration in whichever file of its folder holds it,
/// and hover says where it resolved from (report 12 §3.8).
#[test]
fn designs_named_by_symbol_go_to_their_declaration_across_folders() {
    let text = "runmat 0.7\nstd 1\n\nassembly cabin {\n    use part seat_frame as seat\n    use part rivet from hardware as rivet\n}\n";
    let files = DesignFiles::from([
        ("/p/cabin/seats.part".to_string(), "runmat 0.7\nstd 1\n\npart seats {\n    output seat_frame = { body = feature.b.body }\n}\n".to_string()),
        ("/p/+hardware/rivet.part".to_string(), "runmat 0.7\nstd 1\n\npart rivet {\n}\n".to_string()),
    ]);
    let analysis = analyze("/p/cabin/cabin.assembly", text, &files);
    let seat = analysis.definition(&at(text, "seat_frame", 0, 2)).expect("a definition");
    assert_eq!(seat.uri.path(), "/p/cabin/seats.part");
    assert_eq!(seat.range.start, Position::new(4, 11), "the output's name");
    let rivet = analysis.definition(&at(text, "rivet from", 0, 1)).expect("a definition");
    assert_eq!(rivet.uri.path(), "/p/+hardware/rivet.part");
    let Some(HoverContents::Markup(hover)) = analysis.hover(&at(text, "as rivet", 0, 4)).map(|h| h.contents) else {
        panic!("a hover")
    };
    assert!(hover.value.contains("resolved from `hardware`"), "{}", hover.value);
    assert!(hover.value.contains("/p/+hardware/rivet.part"), "{}", hover.value);
}
