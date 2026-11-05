//! AP242 semantic PMI: written into a STEP file with named faces, checked against the AP242
//! entity definitions, and read back unchanged.

use runmat_cad::pmi::{Characteristic, DatumReference, MaterialCondition};
use runmat_cad::step::p21::parse;
use runmat_cad::value::ValueExpr;
use runmat_cad::step::pmi::{self, schema, Limits, StepDatum, StepLocation, StepPmi, StepSize, StepTolerance};

/// The shape of a kernel-written AP242 part: a product definition shape, its B-rep
/// representation in millimetres, and named faces.
const PLATE: &str = "ISO-10303-21;
HEADER;
FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }'));
ENDSEC;
DATA;
#1 = SHAPE_DEFINITION_REPRESENTATION(#2,#10);
#2 = PRODUCT_DEFINITION_SHAPE('','',#3);
#3 = PRODUCT_DEFINITION('design','',#4,#5);
#10 = ADVANCED_BREP_SHAPE_REPRESENTATION('',(#11),#20);
#11 = MANIFOLD_SOLID_BREP('',#12);
#12 = CLOSED_SHELL('',(#13,#14,#15,#16));
#13 = ADVANCED_FACE('feature.body.face.start',(#30),#31,.F.);
#14 = ADVANCED_FACE('feature.body.face.end',(#30),#31,.T.);
#15 = ADVANCED_FACE('feature.body.face.side(s.l1)',(#30),#31,.T.);
#16 = ADVANCED_FACE('feature.h.face.side',(#30),#32,.T.);
#20 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#21)) REPRESENTATION_CONTEXT('','') );
#21 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
ENDSEC;
END-ISO-10303-21;
";

fn faces(n: &[&str]) -> Vec<String> {
    n.iter().map(|s| s.to_string()).collect()
}

fn sample() -> StepPmi {
    StepPmi {
        datums: vec![
            StepDatum { label: "A".into(), faces: faces(&["feature.body.face.start"]) },
            StepDatum { label: "B".into(), faces: faces(&["feature.body.face.side(s.l1)"]) },
        ],
        tolerances: vec![
            StepTolerance {
                name: "top_flat".into(),
                characteristic: Characteristic::Flatness,
                faces: faces(&["feature.body.face.end"]),
                value: 0.05,
                diametral: false,
                modifier: MaterialCondition::Rfs,
                datums: vec![],
            },
            StepTolerance {
                name: "top_parallel".into(),
                characteristic: Characteristic::Parallelism,
                faces: faces(&["feature.body.face.end"]),
                value: 0.05,
                diametral: false,
                modifier: MaterialCondition::Rfs,
                datums: vec![DatumReference { label: "A".into(), modifier: MaterialCondition::Rfs }],
            },
            StepTolerance {
                name: "hole_position".into(),
                characteristic: Characteristic::Position,
                faces: faces(&["feature.h.face.side"]),
                value: 0.1,
                diametral: true,
                modifier: MaterialCondition::Mmc,
                datums: vec![
                    DatumReference { label: "A".into(), modifier: MaterialCondition::Rfs },
                    DatumReference { label: "B".into(), modifier: MaterialCondition::Mmc },
                ],
            },
        ],
        sizes: vec![
            StepSize {
                name: "thickness".into(),
                kind: "thickness".into(),
                faces: faces(&["feature.body.face.start", "feature.body.face.end"]),
                limits: Limits { nominal: 10.0, lower: -0.1, upper: 0.1, fit: None },
            },
            StepSize {
                name: "hole".into(),
                kind: "diameter".into(),
                faces: faces(&["feature.h.face.side"]),
                limits: Limits { nominal: 4.0, lower: 0.0, upper: 0.0, fit: Some("H7".into()) },
            },
        ],
        locations: vec![StepLocation {
            name: "hole_from_side".into(),
            from: faces(&["feature.body.face.side(s.l1)"]),
            to: faces(&["feature.h.face.side"]),
            limits: Limits { nominal: 10.0, lower: -0.05, upper: 0.05, fit: None },
        }],
    }
}

#[test]
fn pmi_round_trips_through_ap242_and_conforms_to_the_schema() {
    let pmi = sample();
    let (text, missing) = pmi::embed(PLATE, &pmi).expect("embeds");
    assert!(missing.is_empty(), "{missing:?}");
    let data = parse(&text).expect("the file still parses");
    assert_eq!(schema::check(&data), Vec::<String>::new(), "{text}");
    for entity in [
        "DATUM_FEATURE(", "DATUM(", "DATUM_SYSTEM(", "DATUM_REFERENCE_COMPARTMENT(", "GEOMETRIC_ITEM_SPECIFIC_USAGE(",
        "FLATNESS_TOLERANCE(", "PARALLELISM_TOLERANCE(", "POSITION_TOLERANCE()", "GEOMETRIC_TOLERANCE_WITH_MODIFIERS((.MAXIMUM_MATERIAL_REQUIREMENT.))",
        "TOLERANCE_ZONE_FORM('cylindrical or circular')", "DIMENSIONAL_SIZE(", "DIMENSIONAL_LOCATION(", "PLUS_MINUS_TOLERANCE(", "LIMITS_AND_FITS('H','','7'",
    ] {
        assert!(text.contains(entity), "no {entity}");
    }
    let back = pmi::read(&text).expect("reads");
    assert_eq!(back, pmi);
}

#[test]
fn the_schema_check_finds_wrong_arities_and_references() {
    let (text, _) = pmi::embed(PLATE, &sample()).unwrap();
    let bad = text.replacen("DATUM('A','',#2,.F.,'A')", "DATUM('A','',#13,.F.)", 1);
    let p = schema::check(&parse(&bad).unwrap()).join("\n");
    assert!(p.contains("DATUM takes 5 attributes"), "{p}");
    let bad = text.replacen("GEOMETRIC_ITEM_SPECIFIC_USAGE('','',", "GEOMETRIC_ITEM_SPECIFIC_USAGE('','',#13,", 1).replacen("#10,#13)", "#13)", 1);
    let p = schema::check(&parse(&bad).unwrap()).join("\n");
    assert!(!p.is_empty());
}

#[test]
fn values_follow_the_files_length_unit() {
    let metres = PLATE.replace("SI_UNIT(.MILLI.,.METRE.)", "SI_UNIT($,.METRE.)");
    let (text, _) = pmi::embed(&metres, &sample()).unwrap();
    assert!(!text.contains("LENGTH_MEASURE(0.05)"), "values are written in the file's unit (metres)");
    let back = pmi::read(&text).unwrap();
    assert!((back.tolerances[0].value - 0.05).abs() < 1e-12);
}

/// The `.part` PMI statements this lane reads and writes parse into the model.
#[test]
fn part_pmi_statements_parse() {
    let text = r#"runmat 0.7
std 1

part plate {
    sketch s on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = l1.end, end = [80 mm, 50 mm])
        l3 = line(start = l2.end, end = [0 mm, 50 mm])
        l4 = line(start = l3.end, end = l1.start)
    }
    feature body = extrude(profile = sketch.s, distance = 10 mm)
    datum A = { on = feature.body.face.start }
    tolerance top_parallel = { characteristic = parallelism, on = [feature.body.face.end], tolerance = 0.05 mm, datums = [{ label = "A" }] }
    tolerance hole_pos = { characteristic = position, on = [feature.body.face.end], tolerance = 0.1 mm, diametral = true, modifier = mmc, datums = [{ label = "A", modifier = mmc }] }
    size thickness = { on = [feature.body.face.start, feature.body.face.end], size = 10 mm ± [0.1 mm, 0.05 mm] }
    size bore = { on = [feature.body.face.end], size = 4 mm, fit = "H7" }
    location edge = { from = feature.body.face.side(s.l1), to = feature.body.face.side(s.l3), distance = 50 mm ± 0.2 mm }
}
"#;
    let parsed = runmat_cad::syntax::parse_part("/p/plate.part", text);
    let doc = parsed.document.unwrap_or_else(|| panic!("{:?}", parsed.diagnostics));
    assert!(parsed.diagnostics.is_empty(), "{:?}", parsed.diagnostics);
    let p = &doc.pmi;
    assert_eq!(p.datums.len(), 1);
    assert_eq!(p.tolerances.len(), 2, "{p:?}");
    assert_eq!(p.tolerances[1].modifier, MaterialCondition::Mmc);
    assert!(p.tolerances[1].diametral);
    assert_eq!(p.sizes.len(), 2);
    let ValueExpr::Toleranced { plus, minus, .. } = &p.sizes[0].size else { panic!("{:?}", p.sizes[0].size) };
    assert_ne!(plus, minus, "an asymmetric tolerance keeps both bounds");
    assert!(matches!(p.locations[0].distance, ValueExpr::Toleranced { .. }), "{:?}", p.locations[0].distance);
    assert_eq!(p.sizes[1].fit.as_deref(), Some("H7"));
    assert_eq!(p.locations.len(), 1, "{p:?}");
}
