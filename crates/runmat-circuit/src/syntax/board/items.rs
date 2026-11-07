//! Reading one board item's value (footprint, copper, via, pour, keepout, text, graphic,
//! dimension) against the layers, nets and packages the layout declares.

use crate::layer_set::CopperIndex;
use super::*;
use crate::library::Package;
use runmat_types::Span;
use std::collections::HashMap;

pub(super) struct Ctx<'a> {
    pub(super) layers: &'a Layers,
    pub(super) nets: &'a HashMap<String, NetId>,
    pub(super) packages: &'a HashMap<String, Package>,
}

impl Ctx<'_> {
    pub(super) fn net(&self, name: &str, span: Span) -> Read<NetId> {
        self.nets.get(name).copied().ok_or_else(|| {
            diagnostic(
                span,
                format!("`{name}` is not a net of this board; declare it with `net {name}`"),
            )
        })
    }

    pub(super) fn head_net(&self, h: &Head) -> Read<Option<NetId>> {
        h.name
            .as_ref()
            .map(|(n, span)| self.net(n, *span))
            .transpose()
    }

    pub(super) fn footprint(&self, id: ItemId, reference: String, v: &Val) -> Read<Footprint> {
        let mut f = Fields::of(v, "footprint")?;
        let pv = f.need(words::PACKAGE)?;
        let key = text(pv)?;
        let package = self.packages.get(&key).cloned().ok_or_else(|| {
            diagnostic(
                pv.span,
                format!("package `{key}` is not defined in this layout"),
            )
        })?;
        let mut pad_nets = BTreeMap::new();
        if let Some(pads) = f.get("pads") {
            let V::Rec(entries) = &pads.v else {
                return Err(diagnostic(pads.span, "pads are `{ \"1\" = <net>, … }`"));
            };
            for (pad, net) in entries {
                pad_nets.insert(pad.clone(), self.net(&text(net)?, net.span)?);
            }
        }
        let properties = match f.get("properties") {
            Some(p) => {
                let V::Rec(entries) = &p.v else {
                    return Err(diagnostic(p.span, "properties are `{ <name> = \"…\", … }`"));
                };
                entries
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), text(v)?)))
                    .collect::<Read<_>>()?
            }
            None => BTreeMap::new(),
        };
        let fp = Footprint {
            id,
            reference,
            component: None,
            value: f.or("value", String::new(), text)?,
            package,
            at: point(f.need("at")?)?,
            rotation: f.or("rotation", 0.0, angle)?,
            side: f.or("side", Side::Top, side_of)?,
            locked: f.or("locked", false, boolean)?,
            pad_nets,
            fitted: f.or("fitted", true, boolean)?,
            properties,
        };
        f.finish()?;
        Ok(fp)
    }

    pub(super) fn track(&self, id: ItemId, word: &'static str, h: &Head) -> Read<Track> {
        let net = self.head_net(h)?;
        let mut f = Fields::of(need_value(h)?, word)?;
        let layer = self.layers.copper(f.need("layer")?)?;
        let width = len(f.need("width")?)?;
        let geom = if word == words::ARC {
            TrackGeom::Arc {
                start: point(f.need("from")?)?,
                mid: point(f.need("mid")?)?,
                end: point(f.need("to")?)?,
            }
        } else {
            TrackGeom::Segment {
                a: point(f.need("from")?)?,
                b: point(f.need("to")?)?,
            }
        };
        let t = Track {
            id,
            net,
            layer,
            width,
            geom,
            locked: f.or("locked", false, boolean)?,
        };
        f.finish()?;
        Ok(t)
    }

    pub(super) fn via(&self, id: ItemId, h: &Head, copper: CopperIndex) -> Read<Via> {
        let net = self.head_net(h)?;
        let mut f = Fields::of(need_value(h)?, "via")?;
        let (from, to) = match f.get("layers") {
            None => (0, copper.saturating_sub(1)),
            Some(v) => match list(v)? {
                [a, b] => {
                    let (a, b) = (self.layers.copper(a)?, self.layers.copper(b)?);
                    (a.min(b), a.max(b))
                }
                _ => {
                    return Err(diagnostic(
                        v.span,
                        "a via's layers are its first and last copper layer: `[F.Cu, In2.Cu]`",
                    ))
                }
            },
        };
        let v = Via {
            id,
            net,
            at: point(f.need("at")?)?,
            diameter: len(f.need("size")?)?,
            drill: len(f.need("drill")?)?,
            from,
            to,
            micro: f.or("micro", false, boolean)?,
            tented: f.or("tented", true, boolean)?,
            locked: f.or("locked", false, boolean)?,
        };
        f.finish()?;
        Ok(v)
    }

    pub(super) fn copper_list(&self, v: &Val) -> Read<Vec<CopperIndex>> {
        list(v)?.iter().map(|l| self.layers.copper(l)).collect()
    }

    pub(super) fn pour(&self, id: ItemId, h: &Head) -> Read<Zone> {
        let net = self.head_net(h)?;
        let mut f = Fields::of(need_value(h)?, "pour")?;
        let name = f.or("name", String::new(), text)?;
        let layers = self.copper_list(f.need("layers")?)?;
        let outline = polygon(&mut f)?;
        let mut z = Zone::new(id, &name, net, layers, outline);
        z.priority = f.or("priority", 0, whole)? as i32;
        z.clearance = f.opt("clearance", len)?;
        z.min_width = f.or("min_width", z.min_width, len)?;
        z.connect = f.or("connect", z.connect, connect_of)?;
        z.thermal_gap = f.or("thermal_gap", z.thermal_gap, len)?;
        z.spoke_width = f.or("spoke_width", z.spoke_width, len)?;
        z.islands = f.or("islands", z.islands, islands_of)?;
        z.locked = f.or("locked", false, boolean)?;
        f.finish()?;
        Ok(z)
    }

    pub(super) fn keepout(&self, id: ItemId, h: &Head) -> Read<Keepout> {
        let name = h.name.clone().map(|n| n.0).unwrap_or_default();
        let mut f = Fields::of(need_value(h)?, "keepout")?;
        let outline = polygon(&mut f)?;
        let layers = f.or("layers", Vec::new(), |v| self.copper_list(v))?;
        let rules = KeepoutRules {
            tracks: f.or("tracks", false, boolean)?,
            vias: f.or("vias", false, boolean)?,
            pads: f.or("pads", false, boolean)?,
            pours: f.or("pours", false, boolean)?,
            footprints: f.or("footprints", false, boolean)?,
            max_height: f.opt("max_height", len)?,
        };
        f.finish()?;
        Ok(Keepout {
            id,
            name,
            outline,
            layers,
            rules,
        })
    }
}
