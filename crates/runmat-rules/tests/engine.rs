//! Selector, cascade, pair-rule and incremental tests over a small mutable board graph.

use runmat_rules::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct Item {
    kind: String,
    name: Option<String>,
    classes: Vec<String>,
    attrs: BTreeMap<String, AttrValue>,
    parent: Option<u32>,
    layer: Option<String>,
    net_class: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct Board {
    items: BTreeMap<u32, Item>,
}

impl Board {
    fn add(&mut self, id: u32, kind: &str, parent: Option<u32>) -> &mut Item {
        self.items.insert(
            id,
            Item {
                kind: kind.into(),
                name: None,
                classes: vec![],
                attrs: BTreeMap::new(),
                parent,
                layer: None,
                net_class: None,
            },
        );
        self.items.get_mut(&id).unwrap()
    }
}

impl Item {
    fn name(&mut self, n: &str) -> &mut Self {
        self.name = Some(n.into());
        self
    }
    fn class(&mut self, c: &str) -> &mut Self {
        self.classes.push(c.into());
        self
    }
    fn attr(&mut self, k: &str, v: AttrValue) -> &mut Self {
        self.attrs.insert(k.into(), v);
        self
    }
    fn layer(&mut self, l: &str) -> &mut Self {
        self.layer = Some(l.into());
        self
    }
    fn net_class(&mut self, c: &str) -> &mut Self {
        self.net_class = Some(c.into());
        self
    }
}

impl ItemGraph for Board {
    type Id = u32;
    fn items(&self) -> Vec<u32> {
        self.items.keys().rev().copied().collect()
    }
    fn kind(&self, id: u32) -> &str {
        &self.items[&id].kind
    }
    fn is_kind(&self, id: u32, kind: &str) -> bool {
        let k = self.kind(id);
        k == kind || (kind == "copper" && matches!(k, "pad" | "track" | "via"))
    }
    fn name(&self, id: u32) -> Option<&str> {
        self.items[&id].name.as_deref()
    }
    fn has_class(&self, id: u32, c: &str) -> bool {
        self.items[&id].classes.iter().any(|x| x == c)
    }
    fn attr(&self, id: u32, n: &str) -> Option<AttrValue> {
        self.items[&id].attrs.get(n).cloned()
    }
    fn parent(&self, id: u32) -> Option<u32> {
        self.items.get(&id).and_then(|i| i.parent)
    }
    fn children(&self, id: u32) -> Vec<u32> {
        self.items
            .iter()
            .filter(|(_, i)| i.parent == Some(id))
            .map(|(k, _)| *k)
            .collect()
    }
    fn pseudo(&self, id: u32, name: &str, args: &[PseudoArg]) -> Option<bool> {
        let it = &self.items[&id];
        match name {
            "layer" => {
                let l = it.layer.as_deref()?;
                Some(args.iter().any(|a| match a.text.as_str() {
                    "outer" => l == "F.Cu" || l == "B.Cu",
                    "inner" => l.starts_with("In"),
                    t => t == l,
                }))
            }
            "net-class" => {
                // Inherit the net class from the nearest net ancestor.
                let mut cur = Some(id);
                while let Some(c) = cur {
                    if let Some(nc) = &self.items[&c].net_class {
                        return Some(args.iter().any(|a| &a.text == nc));
                    }
                    cur = self.items[&c].parent;
                }
                Some(false)
            }
            _ => None,
        }
    }
}

fn mm(v: f64) -> AttrValue {
    AttrValue::Num(Quantity::length_mm(v))
}

fn sel(s: &str) -> Selector {
    Selector::parse(s, &BuiltinUnits).unwrap_or_else(|e| panic!("{s}: {e}"))
}

/// nets 1 (VBUS, power), 2 (SDA, signal); tracks under nets; a component with pads.
fn board() -> Board {
    let mut b = Board::default();
    b.add(1, "net", None)
        .name("VBUS")
        .class("power")
        .net_class("power");
    b.add(2, "net", None)
        .name("SDA")
        .class("signal")
        .net_class("default");
    b.add(10, "track", Some(1))
        .layer("F.Cu")
        .attr("width", mm(0.5));
    b.add(11, "track", Some(1))
        .layer("In1.Cu")
        .attr("width", mm(0.2));
    b.add(12, "track", Some(2))
        .layer("B.Cu")
        .attr("width", mm(0.15));
    b.add(20, "component", None)
        .name("U1")
        .attr("ref", AttrValue::str("U1"))
        .attr("mpn", AttrValue::str("STM32G071RBT6"))
        .attr("locked", AttrValue::Bool(true));
    b.add(21, "pad", Some(20)).name("U1.1").layer("F.Cu").attr(
        "layers",
        AttrValue::List(vec![AttrValue::str("F.Cu"), AttrValue::str("F.Mask")]),
    );
    b.add(30, "component", None)
        .name("R1")
        .attr("ref", AttrValue::str("R1"))
        .attr(
            "value",
            AttrValue::Num(Quantity::new(10e3, Dim::RESISTANCE)),
        );
    b.add(31, "pad", Some(30)).name("R1.1").layer("B.Cu");
    b
}

fn ids(b: &Board, s: &str) -> Vec<u32> {
    sel(s).select(b).into_iter().map(|(i, _)| i).collect()
}

#[test]
fn type_id_class_attribute() {
    let b = board();
    assert_eq!(ids(&b, "track"), vec![10, 11, 12]);
    assert_eq!(ids(&b, "copper"), vec![10, 11, 12, 21, 31]);
    assert_eq!(ids(&b, "#VBUS"), vec![1]);
    assert_eq!(ids(&b, "net.power"), vec![1]);
    assert_eq!(ids(&b, "*.signal"), vec![2]);
    assert_eq!(ids(&b, "component[ref^=U]"), vec![20]);
    assert_eq!(ids(&b, "component[ref$='1']"), vec![20, 30]);
    assert_eq!(ids(&b, "[mpn*=G071]"), vec![20]);
    assert_eq!(ids(&b, "[mpn*=g071 i]"), vec![20]);
    assert_eq!(ids(&b, "[locked]"), vec![20]);
    assert_eq!(ids(&b, "[locked=false]"), Vec::<u32>::new());
    assert_eq!(ids(&b, "pad[layers~=F.Mask]"), vec![21]);
    assert_eq!(ids(&b, "component[ref!=U1]"), vec![30]);
    assert_eq!(ids(&b, "#\"U1.1\""), vec![21]);
}

#[test]
fn numeric_comparisons_with_units() {
    let b = board();
    assert_eq!(ids(&b, "track[width>=0.2mm]"), vec![10, 11]);
    assert_eq!(ids(&b, "track[width>0.2mm]"), vec![10]);
    assert_eq!(ids(&b, "track[width<7mil]"), vec![12]);
    assert_eq!(ids(&b, "track[width=200um]"), vec![11]);
    assert_eq!(ids(&b, "component[value>=4.7kohm]"), vec![30]);
    // Dimension mismatch never matches.
    assert_eq!(ids(&b, "track[width>1V]"), Vec::<u32>::new());
    assert!(Selector::parse("track[width>wide]", &BuiltinUnits).is_err());
}

#[test]
fn combinators_and_pseudos() {
    let b = board();
    assert_eq!(ids(&b, "net.power > track"), vec![10, 11]);
    assert_eq!(ids(&b, "net track"), vec![10, 11, 12]);
    assert_eq!(ids(&b, "component[ref^=U] pad"), vec![21]);
    assert_eq!(ids(&b, "component > pad:layer(outer)"), vec![21, 31]);
    assert_eq!(ids(&b, "track:layer(inner)"), vec![11]);
    assert_eq!(ids(&b, "track:net-class(power)"), vec![10, 11]);
    assert_eq!(ids(&b, "copper:net-class(power):layer(outer)"), vec![10]);
    assert_eq!(ids(&b, "track:not(:layer(outer))"), vec![11]);
    assert_eq!(ids(&b, "net:has(> track:layer(B.Cu))"), vec![2]);
    assert_eq!(ids(&b, "component:has(pad:layer(F.Cu))"), vec![20]);
    assert_eq!(ids(&b, ":is(net, component)#VBUS"), vec![1]);
    // Unknown host pseudo-classes never match.
    assert_eq!(ids(&b, "track:frobnicate"), Vec::<u32>::new());
}

#[test]
fn set_operations() {
    let b = board();
    assert_eq!(ids(&b, "track, pad"), vec![10, 11, 12, 21, 31]);
    assert_eq!(ids(&b, "copper & :layer(outer)"), vec![10, 12, 21, 31]);
    assert_eq!(ids(&b, "copper - :layer(B.Cu)"), vec![10, 11, 21]);
    assert_eq!(
        ids(&b, "(track, pad) - track:net-class(power)"),
        vec![12, 21, 31]
    );
    assert_eq!(ids(&b, "copper - pad - track:layer(inner)"), vec![10, 12]);
}

#[test]
fn specificity_rules() {
    let b = board();
    let s = |t: &str, id| sel(t).matches(&b, id).unwrap();
    assert_eq!(
        s("track", 10),
        Specificity {
            ids: 0,
            classes: 0,
            types: 1
        }
    );
    assert_eq!(
        s("net.power > track", 10),
        Specificity {
            ids: 0,
            classes: 1,
            types: 2
        }
    );
    assert_eq!(
        s("#VBUS", 1),
        Specificity {
            ids: 1,
            classes: 0,
            types: 0
        }
    );
    assert_eq!(
        s(":where(#VBUS) track", 10),
        Specificity {
            ids: 0,
            classes: 0,
            types: 1
        }
    );
    assert_eq!(
        s(":is(#VBUS, net) track", 10),
        Specificity {
            ids: 1,
            classes: 0,
            types: 1
        }
    );
    // Union takes the best matching branch; intersection sums.
    assert_eq!(s("track, track:layer(outer)", 10).classes, 1);
    assert_eq!(s("track, track:layer(outer)", 11).classes, 0);
    assert_eq!(
        s("track & :layer(outer)", 10),
        Specificity {
            ids: 0,
            classes: 1,
            types: 1
        }
    );
}

#[test]
fn parse_errors_have_spans() {
    let u = BuiltinUnits;
    let e = Selector::parse("track[width", &u).unwrap_err();
    assert!(
        e.message.contains("operator") || e.message.contains("]"),
        "{e}"
    );
    let e = Selector::parse("track:not()", &u).unwrap_err();
    assert!(e.span.start >= 5, "{e}");
    assert!(Selector::parse("", &u).is_err());
    assert!(Selector::parse("track )", &u).is_err());
    assert!(Selector::parse("track[w='abc", &u).is_err());
}

#[test]
fn schema_validation() {
    let schema = Schema::new()
        .kinds(&["net", "track", "pad", "component", "copper"])
        .attr("width", AttrType::Num(Dim::LENGTH))
        .attr("ref", AttrType::Str)
        .pseudo("layer", 1, 4)
        .pseudo("net-class", 1, 8);
    assert!(sel("track[width>=0.2mm]:layer(outer)")
        .validate(&schema)
        .is_empty());
    let errs = sel("trak[width>=2V]:layr(x)").validate(&schema);
    let msgs: Vec<_> = errs.iter().map(|e| e.message.clone()).collect();
    assert_eq!(errs.len(), 3, "{msgs:?}");
    assert!(msgs[0].contains("unknown item type `trak`"));
    assert!(msgs[1].contains("dimension"));
    assert!(msgs[2].contains("unknown pseudo-class"));
    let errs = sel("component[ref>3mm]").validate(&schema);
    assert_eq!(errs.len(), 1);
    let errs = sel("track:layer()").validate(&schema);
    assert_eq!(errs.len(), 1);
}

fn widths(b: &Board) -> Cascade<f64, u32> {
    let mut c: Cascade<f64, u32> = Cascade::new();
    c.register_property("width", false, Some(0.1));
    c.register_property("clearance", true, None);
    let src = |o| SourceKey::new("board.circuit", o, o + 10);
    c.push_rule(
        Rule::new("default-width", sel("track"))
            .at(
                src(0),
                SourceOrder {
                    file_rank: 1,
                    offset: 0,
                },
            )
            .set("width", 0.15),
    );
    c.push_rule(
        Rule::new("power", sel("net.power > track"))
            .at(
                src(20),
                SourceOrder {
                    file_rank: 1,
                    offset: 20,
                },
            )
            .set("width", 0.5)
            .set("clearance", 0.3),
    );
    c.push_rule(
        Rule::new("inner-thin", sel("track:layer(inner)"))
            .at(
                src(40),
                SourceOrder {
                    file_rank: 1,
                    offset: 40,
                },
            )
            .set("width", 0.25),
    );
    c.push_rule(
        Rule::new("net-clearance", sel("net"))
            .at(
                src(60),
                SourceOrder {
                    file_rank: 1,
                    offset: 60,
                },
            )
            .set("clearance", 0.2),
    );
    c.rebuild(b);
    c
}

#[test]
fn precedence_specificity_then_priority_then_source() {
    let b = board();
    let mut c = widths(&b);
    // 11: `net.power > track` (0,1,2) beats `track:layer(inner)` (0,1,1).
    let e = c.explain(&b, 11, "width");
    let r = e.effective.clone().unwrap();
    assert_eq!(r.value, 0.5);
    match &r.origin {
        Origin::Rule {
            rule_id, source, ..
        } => {
            assert_eq!(rule_id, "power");
            assert_eq!(source.start, 20);
        }
        o => panic!("{o:?}"),
    }
    assert_eq!(e.overridden.len(), 2);
    assert_eq!(e.overridden[0].rule_id, "inner-thin");
    assert_eq!(e.overridden[0].lost_to, LostTo::LowerSpecificity);
    assert_eq!(e.overridden[1].rule_id, "default-width");
    assert!(format!("{e}").contains("rule `power`"));

    // Priority breaks a specificity tie.
    let (h1, _) = c.insert_rule(
        &b,
        Rule::new("a", sel("track:layer(B.Cu)"))
            .at(
                SourceKey::new("f", 100, 110),
                SourceOrder {
                    file_rank: 1,
                    offset: 100,
                },
            )
            .priority(5)
            .set("width", 0.33),
    );
    c.insert_rule(
        &b,
        Rule::new("b", sel("track:layer(outer)"))
            .at(
                SourceKey::new("f", 200, 210),
                SourceOrder {
                    file_rank: 1,
                    offset: 200,
                },
            )
            .set("width", 0.44),
    );
    let e = c.explain(&b, 12, "width");
    assert_eq!(e.effective.unwrap().value, 0.33);
    assert_eq!(e.overridden[0].rule_id, "b");
    assert_eq!(e.overridden[0].lost_to, LostTo::LowerPriority);

    // Equal priority: the later source wins.
    c.replace_rule(
        &b,
        h1,
        Rule::new("a", sel("track:layer(B.Cu)"))
            .at(
                SourceKey::new("f", 100, 110),
                SourceOrder {
                    file_rank: 1,
                    offset: 100,
                },
            )
            .set("width", 0.33),
    );
    let e = c.explain(&b, 12, "width");
    assert_eq!(e.effective.unwrap().value, 0.44);
    assert_eq!(e.overridden[0].lost_to, LostTo::EarlierSource);

    // A later file rank beats an earlier one regardless of offset.
    c.insert_rule(
        &b,
        Rule::new("project", sel("track:layer(outer)"))
            .at(
                SourceKey::new("g", 0, 5),
                SourceOrder {
                    file_rank: 2,
                    offset: 0,
                },
            )
            .set("width", 0.55),
    );
    assert_eq!(c.resolve(&b, 12, "width").unwrap().value, 0.55);
}

#[test]
fn inline_inheritance_default() {
    let b = board();
    let mut c = widths(&b);
    c.set_inline(
        10,
        "width",
        1.0,
        SourceKey::new("board.circuit.layout", 5, 9),
    );
    let e = c.explain(&b, 10, "width");
    assert_eq!(e.effective.as_ref().unwrap().value, 1.0);
    assert!(matches!(e.effective.unwrap().origin, Origin::Inline { .. }));
    assert!(e.overridden.iter().all(|x| x.lost_to == LostTo::Inline));
    assert_eq!(e.overridden.len(), 2);

    // Track 12 has no clearance rule of its own; it inherits from its net.
    let r = c.resolve(&b, 12, "clearance").unwrap();
    assert_eq!(r.value, 0.2);
    assert!(matches!(r.origin, Origin::Inherited { from: 2, .. }));
    // Track 10 has the power rule directly.
    assert_eq!(c.resolve(&b, 10, "clearance").unwrap().value, 0.3);
    // A pad under a component: no rules, no ancestor values, no default.
    assert!(c.resolve(&b, 21, "clearance").is_none());
    assert_eq!(c.resolve(&b, 21, "width").unwrap().origin, Origin::Default);
    let all = c.effective_all(&b, 10);
    assert_eq!(all.len(), 2);
}

#[test]
fn incremental_equals_rebuild() {
    // Random edits; after each, the incremental cascade must equal a fresh rebuild.
    let mut b = board();
    let mut c = widths(&b);
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let layers = ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"];
    let mut next_id = 100u32;
    for step in 0..400 {
        let r = next() % 6;
        let keys: Vec<u32> = b.items.keys().copied().collect();
        let (changed, removed): (Vec<u32>, Vec<u32>) = match r {
            0 => {
                let id = next_id;
                next_id += 1;
                let parent = [1u32, 2][(next() % 2) as usize];
                b.add(id, "track", Some(parent))
                    .layer(layers[(next() % 4) as usize])
                    .attr("width", mm((next() % 50) as f64 / 100.0));
                (vec![id], vec![])
            }
            1 => {
                let id = keys[(next() as usize) % keys.len()];
                if id < 100 {
                    continue;
                }
                b.items.remove(&id);
                (vec![], vec![id])
            }
            2 => {
                let id = keys[(next() as usize) % keys.len()];
                if b.items[&id].kind == "track" {
                    b.items.get_mut(&id).unwrap().layer =
                        Some(layers[(next() % 4) as usize].into());
                }
                (vec![id], vec![])
            }
            3 => {
                // Toggle a net's class: changes every child's `net.power > track` match.
                let id = [1u32, 2][(next() % 2) as usize];
                let it = b.items.get_mut(&id).unwrap();
                if it.classes.iter().any(|x| x == "power") {
                    it.classes.retain(|x| x != "power");
                } else {
                    it.classes.push("power".into());
                }
                (vec![id], vec![])
            }
            4 => {
                // Reparent a track.
                let id = keys[(next() as usize) % keys.len()];
                if b.items[&id].kind != "track" {
                    continue;
                }
                let it = b.items.get_mut(&id).unwrap();
                it.parent = Some(if it.parent == Some(1) { 2 } else { 1 });
                (vec![id], vec![])
            }
            _ => {
                // Rule edit.
                let h = c.handle_of("inner-thin").unwrap();
                let w = (next() % 9) as f64 / 10.0;
                let s = if step % 2 == 0 {
                    "track:layer(inner)"
                } else {
                    "track:layer(In2.Cu)"
                };
                c.replace_rule(
                    &b,
                    h,
                    Rule::new("inner-thin", sel(s))
                        .at(
                            SourceKey::new("board.circuit", 40, 50),
                            SourceOrder {
                                file_rank: 1,
                                offset: 40,
                            },
                        )
                        .set("width", w),
                );
                (vec![], vec![])
            }
        };
        c.items_changed(&b, &changed, &removed);
        let mut fresh = c.clone();
        fresh.rebuild(&b);
        assert_eq!(c.match_table(), fresh.match_table(), "step {step}");
        for id in b.items.keys() {
            assert_eq!(c.resolve(&b, *id, "width"), fresh.resolve(&b, *id, "width"));
            assert_eq!(
                c.resolve(&b, *id, "clearance"),
                fresh.resolve(&b, *id, "clearance")
            );
        }
    }
}

#[test]
fn has_invalidates_ancestors() {
    let mut b = board();
    let mut c: Cascade<i32, u32> = Cascade::new();
    c.push_rule(Rule::new("bottom-nets", sel("net:has(> track:layer(B.Cu))")).set("x", 1));
    c.rebuild(&b);
    assert!(c.resolve(&b, 1, "x").is_none());
    b.items.get_mut(&10).unwrap().layer = Some("B.Cu".into());
    let affected = c.items_changed(&b, &[10], &[]);
    assert!(affected.contains(&1));
    assert_eq!(c.resolve(&b, 1, "x").unwrap().value, 1);
}

#[test]
fn pair_rules() {
    let b = board();
    let mut p: PairCascade<f64, u32> = PairCascade::new();
    p.set_default("clearance", 0.1);
    p.push_rule(
        PairRule::new("power-any", sel("copper:net-class(power)"), sel("copper"))
            .at(
                SourceKey::new("r", 0, 1),
                SourceOrder {
                    file_rank: 0,
                    offset: 0,
                },
            )
            .set("clearance", 0.3),
    );
    p.push_rule(
        PairRule::new("power-pad", sel("track:net-class(power)"), sel("pad"))
            .at(
                SourceKey::new("r", 10, 11),
                SourceOrder {
                    file_rank: 0,
                    offset: 10,
                },
            )
            .set("clearance", 0.4),
    );
    p.push_rule(
        PairRule::new(
            "inner",
            sel("copper:layer(inner)"),
            sel("copper:layer(inner)"),
        )
        .at(
            SourceKey::new("r", 20, 21),
            SourceOrder {
                file_rank: 0,
                offset: 20,
            },
        )
        .set("clearance", 0.12),
    );
    p.rebuild(&b);
    // Signal track vs pad: default.
    let r = p.resolve(12, 31, "clearance").unwrap();
    assert_eq!(r.value, 0.1);
    assert!(r.rule.is_none());
    // Power track vs signal track: power-any (both orders).
    assert_eq!(p.resolve(10, 12, "clearance").unwrap().value, 0.3);
    assert_eq!(p.resolve(12, 10, "clearance").unwrap().value, 0.3);
    // Power track vs pad: both rules are (0,1,2); power-pad is later in source.
    let e = p.explain(10, 21, "clearance");
    assert_eq!(e.effective.as_ref().unwrap().value, 0.4);
    assert_eq!(e.overridden[0].winner.rule_id, "power-any");
    assert_eq!(e.overridden[0].lost_to, LostTo::EarlierSource);
    // Inner track pair with itself: inner (0,2,2) beats power-any (0,1,2).
    assert_eq!(p.resolve(11, 11, "clearance").unwrap().value, 0.12);
    assert_eq!(p.max_value_by("clearance", |v| *v), Some(0.4));
    assert!(format!("{e}").contains("power-pad"));
}

#[test]
fn determinism_across_item_orders() {
    // The same graph listed in two orders gives the same tables and explanations.
    let b = board();
    let c1 = widths(&b);
    struct Rev<'a>(&'a Board);
    impl ItemGraph for Rev<'_> {
        type Id = u32;
        fn items(&self) -> Vec<u32> {
            let mut v = self.0.items();
            v.reverse();
            v.rotate_left(3);
            v
        }
        fn kind(&self, id: u32) -> &str {
            self.0.kind(id)
        }
        fn is_kind(&self, id: u32, k: &str) -> bool {
            self.0.is_kind(id, k)
        }
        fn name(&self, id: u32) -> Option<&str> {
            self.0.name(id)
        }
        fn has_class(&self, id: u32, c: &str) -> bool {
            self.0.has_class(id, c)
        }
        fn attr(&self, id: u32, n: &str) -> Option<AttrValue> {
            self.0.attr(id, n)
        }
        fn parent(&self, id: u32) -> Option<u32> {
            self.0.parent(id)
        }
        fn children(&self, id: u32) -> Vec<u32> {
            let mut v = self.0.children(id);
            v.reverse();
            v
        }
        fn pseudo(&self, id: u32, n: &str, a: &[PseudoArg]) -> Option<bool> {
            self.0.pseudo(id, n, a)
        }
    }
    let r = Rev(&b);
    let mut c2: Cascade<f64, u32> = Cascade::new();
    for (_, rule) in c1.rules() {
        c2.push_rule(rule.clone());
    }
    c2.register_property("width", false, Some(0.1));
    c2.register_property("clearance", true, None);
    c2.rebuild(&r);
    assert_eq!(c1.match_table(), c2.match_table());
    for id in b.items.keys() {
        assert_eq!(c1.explain(&b, *id, "width"), c2.explain(&r, *id, "width"));
    }
}

#[test]
fn scale_thousands_of_items() {
    // 20 000 tracks over 2 000 nets, 40 rules; full match and incremental edits stay fast.
    let mut b = Board::default();
    for n in 0..2000u32 {
        let it = b.add(n, "net", None).name(&format!("N{n}"));
        if n % 10 == 0 {
            it.class("power").net_class("power");
        } else {
            it.net_class("default");
        }
    }
    let layers = [
        "F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "In5.Cu", "In6.Cu", "B.Cu",
    ];
    for t in 0..20000u32 {
        b.add(10_000 + t, "track", Some(t % 2000))
            .layer(layers[(t % 8) as usize])
            .attr("width", mm(0.1 + (t % 7) as f64 * 0.05));
    }
    let mut c: Cascade<f64, u32> = Cascade::new();
    for i in 0..40 {
        let s = match i % 4 {
            0 => format!("track[width>={}mm]", 0.1 + i as f64 * 0.01),
            1 => format!("net#N{} > track", i * 37),
            2 => format!("track:layer(In{}.Cu)", 1 + i % 6),
            _ => "net.power track:layer(outer)".to_string(),
        };
        c.push_rule(
            Rule::new(format!("r{i}"), sel(&s))
                .at(
                    SourceKey::new("x", i, i + 1),
                    SourceOrder {
                        file_rank: 0,
                        offset: i,
                    },
                )
                .set("clearance", i as f64),
        );
    }
    let t0 = std::time::Instant::now();
    c.rebuild(&b);
    let full = t0.elapsed();
    let t1 = std::time::Instant::now();
    for k in 0..200u32 {
        b.items.get_mut(&(10_000 + k * 13)).unwrap().layer = Some("B.Cu".into());
        c.items_changed(&b, &[10_000 + k * 13], &[]);
    }
    let inc = t1.elapsed();
    let t2 = std::time::Instant::now();
    let mut n = 0;
    for t in 0..20000u32 {
        if c.resolve(&b, 10_000 + t, "clearance").is_some() {
            n += 1;
        }
    }
    println!(
        "rules scale: rebuild {full:?} (22000 items x 40 rules), 200 incremental edits {inc:?}, 20000 resolves {:?} ({n} with values)",
        t2.elapsed()
    );
    assert!(n > 10_000);
}
