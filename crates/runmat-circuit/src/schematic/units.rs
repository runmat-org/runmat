//! Units: how a component splits into schematic symbols, one per gate, and a large gate by bank,
//! port prefix or power.

use crate::library::PinType;

/// A gate with more pins than this splits into units.
pub const SPLIT_ABOVE: usize = 64;
/// The most pins a split unit holds.
pub const UNIT_PINS: usize = 48;

/// One schematic unit of a component: its name (`A`, `B`… for a split gate, else the gate's
/// name), what it holds when split, and its pins (indices into the component's pins).
#[derive(Clone, Debug, PartialEq)]
pub struct Unit {
    pub name: String,
    pub label: Option<String>,
    pub pins: Vec<usize>,
}

/// The bank a pin name declares: a trailing `_34` (`IO_L1P_T0_34`) or a `BANK3` token.
fn bank_of(name: &str) -> Option<String> {
    let upper = name.to_ascii_uppercase();
    for token in upper.split(['_', '.', '-']) {
        if let Some(n) = token.strip_prefix("BANK") {
            if !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()) {
                return Some(n.to_owned());
            }
        }
    }
    let (head, tail) = upper.rsplit_once('_')?;
    (!head.is_empty() && !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()))
        .then(|| tail.to_owned())
}

/// A pin name's leading letters (`PA` of `PA12`, `IO` of `IO7`).
fn prefix_of(name: &str) -> String {
    name.chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect()
}

fn letters(mut k: usize) -> String {
    let mut out = Vec::new();
    loop {
        out.push((b'A' + (k % 26) as u8) as char);
        if k < 26 {
            break;
        }
        k = k / 26 - 1;
    }
    out.iter().rev().collect()
}

/// A component's units, in drawing order: one per gate, a large gate split as the module
/// docs describe. Deterministic: the same pins give the same units.
pub fn units(c: &crate::elab::CompInst) -> Vec<Unit> {
    let mut gates: Vec<&str> = Vec::new();
    for p in &c.pins {
        if !gates.contains(&p.gate.as_str()) {
            gates.push(&p.gate);
        }
    }
    let single = gates.len() == 1;
    let mut out = Vec::new();
    for g in &gates {
        let idx: Vec<usize> = (0..c.pins.len())
            .filter(|i| c.pins[*i].gate == *g)
            .collect();
        if idx.len() <= SPLIT_ABOVE {
            out.push(Unit {
                name: (*g).to_owned(),
                label: None,
                pins: idx,
            });
            continue;
        }
        let power = |i: &usize| matches!(c.pins[*i].kind, PinType::PowerIn | PinType::PowerOut);
        let (pwr, signal): (Vec<usize>, Vec<usize>) = idx.iter().partition(|i| power(i));
        // Groups of signal pins: by bank when every pin names one, else by port prefix.
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        let banks: Vec<Option<String>> = signal.iter().map(|i| bank_of(&c.pins[*i].pin)).collect();
        let by_bank = banks.iter().all(Option::is_some);
        for (k, &pin) in signal.iter().enumerate() {
            let label = match &banks[k] {
                Some(b) if by_bank => format!("bank {b}"),
                _ => prefix_of(&c.pins[pin].pin),
            };
            match groups.iter_mut().find(|(l, _)| *l == label) {
                Some((_, v)) => v.push(pin),
                None => groups.push((label, vec![pin])),
            }
        }
        if !pwr.is_empty() {
            groups.push(("power".to_owned(), pwr));
        }
        let mut n = 0;
        for (label, pins) in groups {
            // Even units: 200 pins make five of 40, not four of 48 and one of 8.
            let chunks = pins.len().div_ceil(UNIT_PINS);
            let size = pins.len().div_ceil(chunks.max(1));
            for (k, chunk) in pins.chunks(size.max(1)).enumerate() {
                let label = if chunks == 1 {
                    label.clone()
                } else {
                    let first = &c.pins[chunk[0]].pin;
                    let last = &c.pins[chunk[chunk.len() - 1]].pin;
                    if label == "power" || label.starts_with("bank ") {
                        format!("{label} ({}/{chunks})", k + 1)
                    } else {
                        format!("{first}–{last}")
                    }
                };
                let name = if single {
                    letters(n)
                } else {
                    format!("{g}{}", letters(n))
                };
                out.push(Unit {
                    name,
                    label: Some(label),
                    pins: chunk.to_vec(),
                });
                n += 1;
            }
        }
    }
    out
}

/// A unit's pins as the symbol layout reads them: (name, pads, type).
pub(crate) fn unit_pins(
    c: &crate::elab::CompInst,
    unit: &Unit,
) -> Vec<(String, Vec<String>, PinType)> {
    unit.pins
        .iter()
        .map(|i| {
            let p = &c.pins[*i];
            let pads = if p.pads.is_empty() {
                vec![p.pin.clone()]
            } else {
                p.pads.clone()
            };
            (p.pin.clone(), pads, p.kind)
        })
        .collect()
}

#[cfg(test)]
mod unit_split_tests {
    use super::*;

    #[test]
    fn a_large_gate_splits_into_even_bank_and_power_units() {
        let (_, _, nl, _) = crate::fixture::large(&crate::fixture::LargeParams {
            channels: 2,
            copper_layers: 4,
            route_every: 0,
            pours: false,
        });
        let fpga = nl
            .components
            .iter()
            .find(|c| c.pins.len() > SPLIT_ABOVE)
            .expect("an FPGA");
        let units = units(fpga);
        let sizes: Vec<usize> = units.iter().map(|u| u.pins.len()).collect();
        // 200 IO pins in five units of 40, then the power unit.
        assert_eq!(sizes, vec![40, 40, 40, 40, 40, 3], "{units:?}");
        assert_eq!(units[0].name, "A");
        assert_eq!(units[0].label.as_deref(), Some("IO0–IO39"));
        assert_eq!(units[5].label.as_deref(), Some("power"));
        let all: std::collections::BTreeSet<usize> =
            units.iter().flat_map(|u| u.pins.clone()).collect();
        assert_eq!(all.len(), fpga.pins.len(), "every pin in exactly one unit");
    }

    #[test]
    fn banks_come_from_the_pin_name() {
        assert_eq!(bank_of("IO_L1P_T0_34").as_deref(), Some("34"));
        assert_eq!(bank_of("BANK3_IO7").as_deref(), Some("3"));
        assert_eq!(bank_of("IO12"), None);
        assert_eq!(bank_of("VCC_"), None);
        assert_eq!(prefix_of("PA12"), "PA");
        assert_eq!(letters(0), "A");
        assert_eq!(letters(25), "Z");
        assert_eq!(letters(26), "AA");
    }
}
