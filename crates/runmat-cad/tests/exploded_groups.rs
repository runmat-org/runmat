//! Exploded steps that name whole patterns and mirrors read and print back in their text form.

use runmat_cad::assembly::{AssemblyDocument, RelPath};
use runmat_cad::syntax::{self, print_assembly};

const RIG: &str = r#"assembly rig {
    use part clip from "./parts" as clip
    pattern clips = linear(seeds = [clip], direction = [0, 1, 0], spacing = 750 mm, count = 4)
    mirror clips_port = {
        seeds = [clips],
        plane = { plane = explicit, point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0] },
        hand = symmetric(point = [0 mm, 0 mm, 0 mm], normal = [0, 1, 0])
    }
    exploded open = {
        steps = [
            { id = clips_up, patterns = [[{ name = clips }]], motion = { motion = "translate", direction = { direction = "vector", vector = [0, 0, 1] }, distance = 50 mm } },
            { id = port_out, mirrors = [[{ name = clips_port }]], motion = { motion = "translate", direction = { direction = "vector", vector = [0, -1, 0] }, distance = 20 mm, spacing = 5 mm } },
            { id = mixed, occurrences = [[{ name = clip }]], patterns = [[{ name = wing }, { name = ribs }]], mirrors = [[{ name = wing }, { name = ribs_port }]], motion = { motion = "translate", direction = { direction = "vector", vector = [1, 0, 0] }, distance = 10 mm } }
        ]
    }
}
"#;

fn parse(text: &str) -> syntax::Parsed<AssemblyDocument> {
    let scope = runmat_cad::modules::ModuleScope::from_files(
        [
            ("rig.assembly", text),
            ("parts/clip.part", "part clip {\n}\n"),
        ]
        .iter()
        .copied(),
    );
    syntax::parse_assembly_in("rig.assembly", text, &scope)
}

#[test]
fn steps_naming_patterns_and_mirrors_round_trip_through_text() {
    let parsed = parse(RIG);
    assert!(parsed.is_valid(), "{:#?}", parsed.diagnostics);
    let d = parsed.document.unwrap();
    let steps = &d.exploded[0].steps;
    assert_eq!(steps[0].patterns, vec![RelPath::new(&["clips"])]);
    assert!(steps[0].occurrences.is_empty() && steps[0].mirrors.is_empty());
    assert_eq!(steps[1].mirrors, vec![RelPath::new(&["clips_port"])]);
    assert_eq!(steps[2].patterns, vec![RelPath::new(&["wing", "ribs"])]);
    assert_eq!(steps[2].mirrors[0].to_string(), "wing/ribs_port");

    let printed = print_assembly(&d, &parsed.uses).unwrap();
    assert!(
        printed.contains("patterns = [[{ name = clips }]]"),
        "{printed}"
    );
    assert!(
        printed.contains("mirrors = [[{ name = clips_port }]]"),
        "{printed}"
    );
    // A step with no occurrences of its own prints no empty `occurrences` list.
    assert!(!printed.contains("occurrences = []"), "{printed}");
    let back = parse(&printed);
    assert!(back.is_valid(), "{:#?}\n{printed}", back.diagnostics);
    assert_eq!(back.document.as_ref(), Some(&d), "{printed}");
}
