use super::*;

const STUDY: &str = r#"
import material aluminum from "./aluminum.fea"
import function impactPulse from "./impactPulse.m"

geometry bracket_geometry {
    file = "./geometry/bracket.step"
    units = mm

    region body : solid {
        select name "Bracket"
        expect exactly_one
    }
}

mesh bracket_mesh on geometry.bracket_geometry {
    tetrahedra {
        order = quadratic
        target_size = 5 mm
    }
}

model bracket(region body : region<solid>) {
    assign material.aluminum to input.body
}

study bracket_static {
    input scale : real = 1.0
    geometry = geometry.bracket_geometry
    mesh = mesh.bracket_mesh
    model = model.bracket

    analysis static {
        structural static {
            constraints = [constraint.fixed_support]
        }
        accuracy {
            relative_residual <= 1e-8
        }
    }

    output peak_stress = max(
        output.von_mises,
        over = geometry.bracket_geometry.region.body,
    )
}

sweep sizes over study.bracket_static {
    axis scale = [0.5, 1.0, 2.0]
}

execution scalable {
    resources {
        workers = 1..64
        memory_per_worker <= 32 GiB
    }
}
"#;

// witness: FEA-AUTHORING-LANGUAGE-002/grammar_and_recovery @
//   shared_infrastructure/mesh_independent/source_check;
//   boundary: Representative declarations, spans and imports only; no complete grammar or semantics
//   proof.
#[test]
fn parses_representative_language_surface() {
    let module = parse_fea(STUDY);
    assert!(module.diagnostics.is_empty(), "{:#?}", module.diagnostics);
    assert_eq!(module.items.len(), 8);
    assert_eq!(module.source(), STUDY);
    assert!(module.is_valid());
    let FeaModuleItem::Import(function_import) = &module.items[1] else {
        panic!("expected function import");
    };
    assert_eq!(
        &STUDY[function_import.source_span.start..function_import.source_span.end],
        "\"./impactPulse.m\""
    );
}

#[test]
fn parses_typed_geometry_features_and_derived_lineage() {
    let module = parse_fea(
        "geometry base {\nconstruct {\nfeature stock = box(width = 1 m, depth = 1 m, height = 1 m)\nresult = feature.stock\n}\n}\ngeometry derived from geometry.base {\nderive {\nfeature cut = cylinder(radius = 1 mm, height = 1 m, axis = global([0, 0, 1]), origin = [0, 0, 0] m)\nresult = subtract(source.result, feature.cut)\n}\n}\n",
    );
    assert!(module.diagnostics.is_empty(), "{:#?}", module.diagnostics);
    let FeaModuleItem::Declaration(derived) = &module.items[1] else {
        panic!("expected derived geometry");
    };
    let source = derived.source.as_ref().expect("typed source lineage");
    assert_eq!(source.segments[1].text, "base");
    let FeaStatementKind::Block { body, .. } = &derived.body.statements[0].kind else {
        panic!("expected derive block");
    };
    assert!(matches!(
        body.statements[0].kind,
        FeaStatementKind::Feature { .. }
    ));
}

#[test]
fn parses_typed_output_call_and_attributed_region() {
    let module = parse_fea(
        "study s {\noutput peak = max(output.stress, over = geometry.g.region.body)\n}\n",
    );
    assert!(module.diagnostics.is_empty(), "{:#?}", module.diagnostics);
    let FeaModuleItem::Declaration(declaration) = &module.items[0] else {
        panic!("expected declaration");
    };
    let FeaStatementKind::Output { value, .. } = &declaration.body.statements[0].kind else {
        panic!("expected output");
    };
    let FeaExpressionKind::Call { arguments, .. } = &value.kind else {
        panic!("expected call");
    };
    assert_eq!(arguments.len(), 2);
    assert!(matches!(arguments[1], FeaCallArgument::Named { .. }));
}

#[test]
fn parses_output_policy_block_without_reclassifying_its_value() {
    let module = parse_fea(
        "study s {\noutput history {\nvalue = sample(output.tip, over = time, at = [0 s, 1 s])\nretention = explicit_samples\nmaximum_samples = 2\n}\n}\n",
    );
    assert!(module.diagnostics.is_empty(), "{:#?}", module.diagnostics);
    let FeaModuleItem::Declaration(declaration) = &module.items[0] else {
        panic!("expected declaration");
    };
    let FeaStatementKind::OutputDefinition { body, .. } = &declaration.body.statements[0].kind
    else {
        panic!("expected output definition");
    };
    assert_eq!(body.statements.len(), 3);
    assert!(matches!(
        body.statements[0].kind,
        FeaStatementKind::Assignment { .. }
    ));
}

// witness: FEA-AUTHORING-COMPILER-002/syntax_pass @
//   shared_infrastructure/mesh_independent/source_check;
//   boundary: Malformed top-level token then valid material declaration; syntax recovery only, not
//   complete editor services.
#[test]
fn recovers_at_the_next_top_level_declaration() {
    let module = parse_fea("not_a_declaration\nmaterial steel {\nname = \"steel\"\n}\n");
    assert_eq!(module.diagnostics.len(), 1);
    assert_eq!(module.items.len(), 2);
    assert!(matches!(module.items[0], FeaModuleItem::Error(_)));
    assert!(matches!(
        module.items[1],
        FeaModuleItem::Declaration(FeaDeclaration {
            kind: FeaDeclarationKind::Material,
            ..
        })
    ));
}

#[test]
fn declaration_names_reject_reserved_keywords() {
    let module = parse_fea("material study {\n}\n");
    assert!(module
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.kind == FeaSyntaxDiagnosticKind::ExpectedIdentifier));
    assert!(matches!(module.items[0], FeaModuleItem::Error(_)));
}

#[test]
fn mesh_target_is_a_typed_geometry_reference() {
    let module =
        parse_fea("mesh bracket on geometry.bracket_geometry {\n    tetrahedra {\n    }\n}\n");
    assert!(module.diagnostics.is_empty(), "{:#?}", module.diagnostics);
    let FeaModuleItem::Declaration(declaration) = &module.items[0] else {
        panic!("expected mesh declaration");
    };
    let target = declaration.target.as_ref().expect("mesh target");
    assert_eq!(
        target
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>(),
        ["geometry", "bracket_geometry"]
    );
}

#[test]
fn imports_after_declarations_are_diagnosed_but_retained() {
    let module =
        parse_fea("material steel {\n}\nimport geometry bracket from \"./bracket.step\"\n");
    assert!(module
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.kind == FeaSyntaxDiagnosticKind::ImportAfterDeclaration));
    assert!(matches!(module.items[1], FeaModuleItem::Import(_)));
}

#[test]
fn reports_unclosed_blocks_but_keeps_the_partial_tree() {
    let module = parse_fea("study incomplete {\nanalysis static {\nkind = structural\n");
    assert!(module
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.kind == FeaSyntaxDiagnosticKind::UnclosedBlock));
    assert_eq!(module.items.len(), 1);
}

#[test]
fn same_line_collections_require_commas() {
    let module = parse_fea("study invalid {\nvalues = [1 2]\n}\n");
    assert!(module
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.message.contains("expected `,`")));
}

#[test]
fn signed_unit_exponents_preserve_quantity_arithmetic() {
    for (unit, operation) in [("K^-1", "+"), ("K^+1", "-"), ("kg*m^-3", "+")] {
        let source = format!("model m {{\nvalue = 2 {unit} {operation} 3 {unit}\n}}\n");
        let module = parse_fea(&source);
        assert!(module.diagnostics.is_empty(), "{:?}", module.diagnostics);
        let FeaModuleItem::Declaration(declaration) = &module.items[0] else {
            panic!("expected model");
        };
        let FeaStatementKind::Assignment { value, .. } = &declaration.body.statements[0].kind
        else {
            panic!("expected assignment");
        };
        let FeaExpressionKind::Binary {
            left,
            operator,
            right,
        } = &value.kind
        else {
            panic!("expected quantity arithmetic");
        };
        assert_eq!(
            *operator,
            if operation == "+" {
                FeaBinaryOperator::Add
            } else {
                FeaBinaryOperator::Subtract
            }
        );
        for operand in [left, right] {
            let FeaExpressionKind::Quantity { unit: tokens, .. } = &operand.kind else {
                panic!("expected quantity operand");
            };
            assert_eq!(&source[tokens.span.start..tokens.span.end], unit);
        }
        let formatted = format_fea(&source, FeaFormatOptions::default());
        assert!(
            formatted.diagnostics.is_empty(),
            "{:?}",
            formatted.diagnostics
        );
        assert!(formatted.source.contains(unit));
        assert_eq!(
            format_fea(&formatted.source, FeaFormatOptions::default()).source,
            formatted.source
        );
    }
}

// witness: FEA-AUTHORING-LANGUAGE-002/grammar_and_recovery @
//   shared_infrastructure/mesh_independent/source_check;
//   boundary: One formatter fixture and second formatting equality; no complete recovery or
//   adversarial grammar credit.
#[test]
fn formatter_is_canonical_and_idempotent() {
    let source = "study   bracket{\ngeometry=geometry.g\noutput peak=max(output.stress,over=geometry.g.region.body)\n}\n";
    let formatted = format_fea(source, FeaFormatOptions::default());
    assert!(
        formatted.diagnostics.is_empty(),
        "{:#?}",
        formatted.diagnostics
    );
    assert_eq!(
        formatted.source,
        "study bracket {\n    geometry = geometry.g\n    output peak = max(output.stress, over = geometry.g.region.body)\n}\n"
    );
    let second = format_fea(&formatted.source, FeaFormatOptions::default());
    assert_eq!(second.source, formatted.source);
}

#[test]
fn formatter_preserves_comments_and_multiline_calls() {
    let source = "study s { // study\noutput peak = max(\noutput.stress, // field\nover = geometry.g.region.body,\n)\n}\n";
    let formatted = format_fea(source, FeaFormatOptions::default());
    assert!(formatted.source.contains("// study"));
    assert!(formatted.source.contains("// field"));
    assert!(formatted.source.contains("    output peak = max(\n"));
    assert_eq!(
        format_fea(&formatted.source, FeaFormatOptions::default()).source,
        formatted.source
    );
}

#[test]
fn formatter_keeps_units_types_and_unary_signs_compact() {
    let source = "model m {\ninput body:region<solid>\ndensity=7850 kg/m^3\nforce=[0,-input.load,0] kN\nmetadata={label=\"x\",enabled=true}\n}\n";
    let formatted = format_fea(source, FeaFormatOptions::default());
    assert!(
        formatted.diagnostics.is_empty(),
        "{:#?}",
        formatted.diagnostics
    );
    assert!(formatted.source.contains("input body : region<solid>"));
    assert!(formatted.source.contains("density = 7850 kg/m^3"));
    assert!(formatted.source.contains("force = [0, -input.load, 0] kN"));
    assert!(formatted
        .source
        .contains("metadata = { label = \"x\", enabled = true }"));
}
