//! The library symbols a fresh `.kicad_sch` embeds: one per rail's power symbol and one per
//! entity and unit layout, with KiCad units numbered as the schematic's units.

use super::*;

/// The power symbol a rail's ports place: a pin at the origin (the connection point), hidden,
/// named after the rail, and the bar or arrow drawn away from it.
pub(super) fn power_lib_symbol(net: &str) -> SExpr {
    let lib = power_lib(net);
    let short = lib.split_once(':').map(|x| x.1).unwrap_or(&lib).to_string();
    let ground = is_ground_rail(net);
    let pt = |x: f64, y: f64| SExpr::list("xy", vec![len(x), len(y)]);
    let shape = if ground {
        vec![
            pt(0.0, 0.0),
            pt(0.0, -1.27),
            pt(1.27, -1.27),
            pt(0.0, -2.54),
            pt(-1.27, -1.27),
            pt(0.0, -1.27),
        ]
    } else {
        vec![
            pt(0.0, 0.0),
            pt(0.0, 2.54),
            pt(-0.762, 1.27),
            pt(0.0, 2.54),
            pt(0.762, 1.27),
        ]
    };
    let value_y = if ground { -3.81 } else { 3.556 };
    SExpr::list(
        "symbol",
        vec![
            SExpr::str(&lib),
            SExpr::list("power", vec![]),
            SExpr::list(
                "pin_names",
                vec![SExpr::list("offset", vec![SExpr::num(0.0)])],
            ),
            SExpr::list("in_bom", vec![SExpr::sym("no")]),
            SExpr::list("on_board", vec![SExpr::sym("yes")]),
            SExpr::list(
                "property",
                vec![
                    SExpr::str("Reference"),
                    SExpr::str("#PWR"),
                    SExpr::list(
                        "at",
                        vec![
                            len(0.0),
                            len(if ground { -6.35 } else { 6.35 }),
                            SExpr::num(0.0),
                        ],
                    ),
                    hidden_effects(),
                ],
            ),
            SExpr::list(
                "property",
                vec![
                    SExpr::str("Value"),
                    SExpr::str(net),
                    SExpr::list("at", vec![len(0.0), len(value_y), SExpr::num(0.0)]),
                    effects(None),
                ],
            ),
            SExpr::list(
                "symbol",
                vec![
                    SExpr::str(&format!("{short}_0_1")),
                    SExpr::list(
                        "polyline",
                        vec![
                            SExpr::list("pts", shape),
                            stroke(),
                            SExpr::list(
                                "fill",
                                vec![SExpr::list("type", vec![SExpr::sym("none")])],
                            ),
                        ],
                    ),
                ],
            ),
            SExpr::list(
                "symbol",
                vec![
                    SExpr::str(&format!("{short}_1_1")),
                    SExpr::list(
                        "pin",
                        vec![
                            SExpr::sym("power_in"),
                            SExpr::sym("line"),
                            SExpr::list(
                                "at",
                                vec![
                                    len(0.0),
                                    len(0.0),
                                    SExpr::num(if ground { 270.0 } else { 90.0 }),
                                ],
                            ),
                            SExpr::list("length", vec![SExpr::num(0.0)]),
                            SExpr::sym("hide"),
                            SExpr::list("name", vec![SExpr::str(net), effects(None)]),
                            SExpr::list("number", vec![SExpr::str("1"), effects(None)]),
                        ],
                    ),
                ],
            ),
        ],
    )
}

pub(super) fn lib_symbol(lib: &str, units: &[(UnitLayout, Option<String>)]) -> SExpr {
    let short = lib.split_once(':').map(|x| x.1).unwrap_or(lib);
    let mut v = vec![
        SExpr::str(lib),
        // Names inside the body at the canvas's name offset; numbers above the pin lines.
        SExpr::list(
            "pin_names",
            vec![SExpr::list("offset", vec![mm(NAME_OFFSET)])],
        ),
        SExpr::list("in_bom", vec![SExpr::sym("yes")]),
        SExpr::list("on_board", vec![SExpr::sym("yes")]),
        SExpr::list(
            "property",
            vec![
                SExpr::str("Reference"),
                SExpr::str("U"),
                SExpr::list(
                    "at",
                    vec![SExpr::num(0.0), SExpr::num(0.0), SExpr::num(0.0)],
                ),
                effects(None),
            ],
        ),
        SExpr::list(
            "property",
            vec![
                SExpr::str("Value"),
                SExpr::str(short),
                SExpr::list(
                    "at",
                    vec![SExpr::num(0.0), SExpr::num(0.0), SExpr::num(0.0)],
                ),
                effects(None),
            ],
        ),
    ];
    if units.iter().any(|u| u.0.art.small()) {
        v.insert(2, SExpr::list("pin_numbers", vec![SExpr::sym("hide")]));
    }
    for (u, (lay, label)) in units.iter().enumerate() {
        let mut body = vec![SExpr::str(&format!("{short}_{}_1", u + 1))];
        if let Some(label) = label {
            body.push(SExpr::list("unit_name", vec![SExpr::str(label)]));
        }
        body.extend([SExpr::list(
            "rectangle",
            vec![
                SExpr::list("start", vec![mm(-lay.half_w), mm(lay.half_h)]),
                SExpr::list("end", vec![mm(lay.half_w), mm(-lay.half_h)]),
                SExpr::list(
                    "stroke",
                    vec![
                        SExpr::list("width", vec![SExpr::num(0.254)]),
                        SExpr::list("type", vec![SExpr::sym("default")]),
                    ],
                ),
                SExpr::list(
                    "fill",
                    vec![SExpr::list("type", vec![SExpr::sym("background")])],
                ),
            ],
        )]);
        for lp in &lay.pins {
            let (name, kind) = (&lp.name, lp.kind);
            let tip = lay.tip(lp.side, lp.at);
            // The pin's direction from its tip to the body.
            let angle = match lp.side {
                PinSide::Left => 0.0,
                PinSide::Right => 180.0,
                PinSide::Top => 270.0,
                PinSide::Bottom => 90.0,
            };
            for pad in &lp.pads {
                body.push(SExpr::list(
                    "pin",
                    vec![
                        SExpr::sym(crate::schematic::pin_kind(kind)),
                        SExpr::sym("line"),
                        SExpr::list("at", vec![mm(tip.x), mm(tip.y), SExpr::num(angle)]),
                        SExpr::list("length", vec![mm(lay.pin_len)]),
                        SExpr::list("name", vec![SExpr::str(name), effects(None)]),
                        SExpr::list("number", vec![SExpr::str(pad), effects(None)]),
                    ],
                ));
            }
        }
        v.push(SExpr::list("symbol", body));
    }
    SExpr::list("symbol", v)
}
