//! The design dialects on the shared grammar: generic forms, the lossless tree and the views.

use runmat_design_syntax::ast::{self, Expr, Headed, Item, Statement};
use runmat_design_syntax::tree::NodeKind;
use runmat_design_syntax::{parse_source, Dialect};

const PART: &str = r#"// A bracket, written by hand.
runmat 0.7
std 1

use material aluminum from "./materials" as al
use "./fasteners" as lib
use part skeleton from "//wing/skeleton" @ "R3"

part bracket {
    input width : length = 80 mm   // the overall width
    input thickness : length = 8 mm
    let half = input.width / 2
    property finish : text = "anodized"

    construct {
        plane base = origin.xy
        sketch profile on plane.base {
            line l1 = line(start = [0, 0] mm, end = [input.width, 0 mm])
            constrain horizontal(l1)
            constrain distance(l1.start, l1.end) = 60 mm
            solved {
                l1.end = [80, 0] mm
            }
        }
        feature body = extrude { profile = sketch.profile, distance = input.thickness }
        result = feature.body
    }

    when input.width > 100 mm {
        feature rib = rib { thickness = 2 mm }
    }

    region mount_face : face {
        select feature.body.face.end
        expect exactly_one
    }

    publish hardpoint = feature.body.face.end
    check mass_budget = part.bracket.mass <= 2 kg
    link verifies feature.body to "checks/mass.fea#check.mass_budget"
    tolerance = 10 mm ± 0.1 mm
    asymmetric = 10 mm ± [0.02 mm, 0 mm]
    density = airDensity(altitude = 10 km) as kg/m^3
    pinned = part.skeleton.publish.station_pitch @ "R3"
    ratio = 5 %
    volatile_value = volatile sensor(channel = 2)
}
"#;

#[test]
fn a_part_parses_every_generic_form_without_diagnostics_and_keeps_every_byte() {
    let syntax = parse_source(Dialect::Part, PART);
    assert!(syntax.diagnostics.is_empty(), "{:#?}", syntax.diagnostics);
    assert_eq!(syntax.syntax().text(), PART);
    let file = syntax.file();
    assert_eq!(file.version(false).as_deref(), Some("0.7"));
    assert_eq!(file.version(true).as_deref(), Some("1"));
    let items = file.items();
    assert_eq!(items.len(), 4);
    let Item::Use(material) = &items[0] else {
        panic!()
    };
    let (kind, name) = material.symbol().unwrap();
    assert_eq!(
        (kind.text().as_str(), name.text().as_str()),
        ("material", "aluminum")
    );
    assert_eq!(
        material.source(),
        Some(ast::UseSource::Path("./materials".into()))
    );
    assert_eq!(material.alias().unwrap().text(), "al");
    let Item::Use(folder) = &items[1] else {
        panic!()
    };
    assert!(folder.symbol().is_none());
    assert_eq!(folder.alias().unwrap().text(), "lib");
    let Item::Use(pinned) = &items[2] else {
        panic!()
    };
    assert_eq!(pinned.pin().as_deref(), Some("R3"));
    let Item::Declaration(part) = &items[3] else {
        panic!()
    };
    assert_eq!(part.keyword(), "part");
    assert_eq!(part.name().unwrap().text(), "bracket");
    let statements = part.block().unwrap().statements();
    let kinds: Vec<NodeKind> = statements.iter().map(|s| s.syntax().kind()).collect();
    use NodeKind::*;
    assert_eq!(
        kinds,
        vec![
            ValueStatement,
            ValueStatement,
            ValueStatement,
            ValueStatement,
            BlockStatement,
            When,
            BlockStatement,
            Publish,
            Check,
            Link,
            Assignment,
            Assignment,
            Assignment,
            Assignment,
            Assignment,
            Assignment
        ]
    );
    let Statement::Value(width) = &statements[0] else {
        panic!()
    };
    assert_eq!(width.keyword(), "input");
    assert_eq!(width.type_text().as_deref(), Some("length"));
    let Some(Expr::Quantity(quantity)) = width.value() else {
        panic!()
    };
    assert_eq!(quantity.unit(), "mm");
    let Statement::Block(construct) = &statements[4] else {
        panic!()
    };
    let inner = construct.block().unwrap().statements();
    let Statement::Block(sketch) = &inner[1] else {
        panic!()
    };
    assert_eq!(sketch.keyword(), "sketch");
    assert_eq!(sketch.name().unwrap().text(), "profile");
    assert_eq!(sketch.clauses()[0].word(), "on");
    let sketch_statements = sketch.block().unwrap().statements();
    assert!(matches!(sketch_statements[1], Statement::Constrain(_)));
    let Statement::Constrain(distance) = &sketch_statements[2] else {
        panic!()
    };
    assert!(distance.equals().is_some());
    assert!(matches!(sketch_statements[3], Statement::Solved(_)));
    let Statement::Block(region) = &statements[6] else {
        panic!()
    };
    assert_eq!(region.type_text().as_deref(), Some("face"));
    let tolerance = |index: usize| match &statements[index] {
        Statement::Assignment(assignment) => assignment.value().unwrap(),
        _ => panic!(),
    };
    assert!(matches!(tolerance(10), Expr::Toleranced(_)));
    assert!(matches!(tolerance(11), Expr::Toleranced(_)));
    let Expr::Coerce(coerce) = tolerance(12) else {
        panic!()
    };
    assert_eq!(coerce.unit(), "kg/m^3");
    let Expr::Pinned(pin) = tolerance(13) else {
        panic!()
    };
    assert_eq!(pin.revision().as_deref(), Some("R3"));
    let Expr::Quantity(percent) = tolerance(14) else {
        panic!()
    };
    assert_eq!(percent.unit(), "%");
    assert!(matches!(tolerance(15), Expr::PrefixApplication(_)));
}

#[test]
fn design_dialects_report_misplaced_and_malformed_items() {
    let syntax = parse_source(
        Dialect::Part,
        "part a {\n  constrain\n}\nassembly b {}\nrunmat 0.7\n}\n",
    );
    let messages: Vec<&str> = syntax
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect();
    assert!(
        messages.iter().any(|m| m.contains("`constrain` requires")),
        "{messages:?}"
    );
    assert!(
        messages.iter().any(|m| m.contains("does not start a part")),
        "{messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("before every other item")),
        "{messages:?}"
    );
    assert!(
        messages.iter().any(|m| m.contains("unmatched")),
        "{messages:?}"
    );
    assert_eq!(
        syntax.syntax().text(),
        "part a {\n  constrain\n}\nassembly b {}\nrunmat 0.7\n}\n"
    );
}

#[test]
fn the_fea_dialect_is_the_fea_parser() {
    let source = "study s {\n    use material.steel\n}\n";
    let syntax = parse_source(Dialect::Fea, source);
    let module = runmat_design_syntax::parse_fea(source);
    assert_eq!(syntax.tree, module.tree);
    assert_eq!(syntax.diagnostics, module.diagnostics);
}

#[test]
fn the_design_formatter_is_idempotent_and_keeps_comments() {
    let once = runmat_design_syntax::format_source(Dialect::Part, PART, Default::default());
    assert!(once.diagnostics.is_empty(), "{:#?}", once.diagnostics);
    assert!(once.source.contains("// the overall width"));
    assert!(once.source.contains("10 mm ± 0.1 mm"), "{}", once.source);
    assert!(once.source.contains("@ \"R3\""), "{}", once.source);
    assert!(once.source.contains("5 %"), "{}", once.source);
    let twice =
        runmat_design_syntax::format_source(Dialect::Part, &once.source, Default::default());
    assert_eq!(once.source, twice.source);
}

#[test]
fn design_punctuation_stays_invalid_in_fea_sources() {
    let module = runmat_design_syntax::parse_fea("study s {\n    x = 5 %\n}\n");
    assert!(module
        .tokens
        .diagnostics
        .iter()
        .any(|d| d.message.contains("invalid character `%` in FEA source")));
}

#[test]
fn a_name_the_fea_dialect_reserves_is_a_name_in_a_design() {
    let text = "runmat 0.7\nstd 1\n\nuse circuit esc\n\nlayout esc {\n    net_class signal = { track_width = 0.15 mm, clearance = 0.15 mm }\n    net_class table = { track_width = 0.2 mm }\n    net DRV_EN : signal\n}\n";
    let syntax = parse_source(Dialect::CircuitLayout, text);
    let messages: Vec<&str> = syntax
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect();
    assert!(messages.is_empty(), "{messages:?}");
    assert_eq!(syntax.syntax().text(), text);
}
