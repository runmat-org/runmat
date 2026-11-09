//! STEP AP214 export: a faceted B-rep per body, millimetres.

use std::fmt::Write;

use super::mesh::{normal, Mesh};

fn step_string(s: &str) -> String {
    s.replace('\'', "''")
}

/// STEP AP214 (`AUTOMOTIVE_DESIGN`): one faceted B-rep solid per body, planar triangular faces,
/// millimetres.
pub fn step(meshes: &[Mesh], name: &str) -> String {
    let mut e: Vec<String> = Vec::new();
    let mut add = |s: String| -> usize {
        e.push(s);
        e.len()
    };
    let ctx = add("APPLICATION_CONTEXT('automotive design')".into());
    add(format!(
        "APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{ctx})"
    ));
    let pctx = add(format!("PRODUCT_CONTEXT('',#{ctx},'mechanical')"));
    let product = add(format!(
        "PRODUCT('{n}','{n}','',(#{pctx}))",
        n = step_string(name)
    ));
    let formation = add(format!("PRODUCT_DEFINITION_FORMATION('','',#{product})"));
    let pdctx = add(format!(
        "PRODUCT_DEFINITION_CONTEXT('part definition',#{ctx},'design')"
    ));
    let pd = add(format!(
        "PRODUCT_DEFINITION('design','',#{formation},#{pdctx})"
    ));
    let pds = add(format!("PRODUCT_DEFINITION_SHAPE('','',#{pd})"));
    let len = add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))".into());
    let ang = add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))".into());
    let sol = add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())".into());
    let unc = add(format!(
        "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{len},'distance_accuracy_value','confusion accuracy')"
    ));
    let gctx = add(format!(
        "(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) GLOBAL_UNIT_ASSIGNED_CONTEXT((#{len},#{ang},#{sol})) REPRESENTATION_CONTEXT('',''))"
    ));
    let origin = add("CARTESIAN_POINT('',(0.,0.,0.))".into());
    let z = add("DIRECTION('',(0.,0.,1.))".into());
    let x = add("DIRECTION('',(1.,0.,0.))".into());
    let frame = add(format!("AXIS2_PLACEMENT_3D('',#{origin},#{z},#{x})"));
    let mut solids = Vec::new();
    let num = |v: f64| {
        let s = format!("{v:.6}");
        s.trim_end_matches('0').to_owned()
    };
    for m in meshes.iter().filter(|m| !m.triangles.is_empty()) {
        let points: Vec<usize> = m
            .positions
            .iter()
            .map(|p| {
                add(format!(
                    "CARTESIAN_POINT('',({},{},{}))",
                    num(p[0] * 1e3),
                    num(p[1] * 1e3),
                    num(p[2] * 1e3)
                ))
            })
            .collect();
        let mut faces = Vec::new();
        for t in &m.triangles {
            let p = t.map(|i| m.positions[i as usize]);
            let n = normal(p[0], p[1], p[2]);
            let r = {
                let u = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
                let l = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt().max(1e-12);
                u.map(|v| v / l)
            };
            let loop_ = add(format!(
                "POLY_LOOP('',(#{},#{},#{}))",
                points[t[0] as usize], points[t[1] as usize], points[t[2] as usize]
            ));
            let bound = add(format!("FACE_OUTER_BOUND('',#{loop_},.T.)"));
            let dn = add(format!(
                "DIRECTION('',({},{},{}))",
                num(n[0]),
                num(n[1]),
                num(n[2])
            ));
            let dr = add(format!(
                "DIRECTION('',({},{},{}))",
                num(r[0]),
                num(r[1]),
                num(r[2])
            ));
            let place = add(format!(
                "AXIS2_PLACEMENT_3D('',#{},#{dn},#{dr})",
                points[t[0] as usize]
            ));
            let plane = add(format!("PLANE('',#{place})"));
            faces.push(add(format!("FACE_SURFACE('',(#{bound}),#{plane},.T.)")));
        }
        let shell = add(format!(
            "CLOSED_SHELL('',({}))",
            faces
                .iter()
                .map(|f| format!("#{f}"))
                .collect::<Vec<_>>()
                .join(",")
        ));
        solids.push(add(format!(
            "FACETED_BREP('{}',#{shell})",
            step_string(&m.name)
        )));
    }
    let rep = add(format!(
        "FACETED_BREP_SHAPE_REPRESENTATION('{}',(#{frame},{}),#{gctx})",
        step_string(name),
        solids
            .iter()
            .map(|s| format!("#{s}"))
            .collect::<Vec<_>>()
            .join(",")
    ));
    add(format!("SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})"));
    let mut out = String::from("ISO-10303-21;\nHEADER;\n");
    let _ = writeln!(
        out,
        "FILE_DESCRIPTION(('RunMat design export: faceted B-rep, millimetres'),'2;1');"
    );
    let _ = writeln!(
        out,
        "FILE_NAME('{}.step','',(''),(''),'RunMat','RunMat','');",
        step_string(name)
    );
    out.push_str("FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n");
    for (i, s) in e.iter().enumerate() {
        let _ = writeln!(out, "#{}={};", i + 1, s);
    }
    out.push_str("ENDSEC;\nEND-ISO-10303-21;\n");
    out
}
