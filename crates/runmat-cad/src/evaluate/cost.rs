//! A cost estimate from a shop's stated rates ([`ShopRates`], a [`CostEstimator`]): every number
//! in the estimate is an input or is measured from the part, and each line says how it was
//! worked out.
//!
//! The rule set, per part:
//! - **Material:** the mass bought times the price per kg. Machining buys the stock (`facts`
//!   `stock_mass`: the part's box, with the stock allowance on each side, at the part's
//!   density); the other processes buy the part's own mass.
//! - **Machine time:** machining removes the stock that isn't part (`stock_volume` − volume) at
//!   the removal rate; additive builds the part's volume at the build rate; molding and sheet
//!   metal take the cycle time per part.
//! - **Machine:** the machine time times the machine rate per hour.
//! - **Setup and tooling:** the setup per batch and the tooling (a mold, a die) divided over the
//!   quantity.
//!
//! The unit cost is their sum, and the batch cost the unit cost times the quantity.

use super::{CostEstimate, CostEstimator, CostInput};
use serde::{Deserialize, Serialize};

/// A shop's rates. Money is in `currency`; a rate left out that the process needs is an error
/// naming it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ShopRates {
    pub currency: String,
    /// Per kg of material bought.
    pub price_per_kg: Option<f64>,
    /// Per machine hour.
    pub machine_rate: Option<f64>,
    /// Per batch.
    pub setup: f64,
    /// Tooling for the batch (a mold, a die), divided over the quantity.
    pub tooling: f64,
    /// Machining: material removed per minute (m³/min).
    pub removal_rate: Option<f64>,
    /// Additive: volume built per hour (m³/h).
    pub build_rate: Option<f64>,
    /// Molding and sheet metal: machine time per part (s).
    pub cycle: Option<f64>,
}

fn need(v: Option<f64>, what: &str) -> Result<f64, String> {
    match v {
        Some(x) if x > 0.0 && x.is_finite() => Ok(x),
        Some(_) => Err(format!("{what} must be more than zero")),
        None => Err(format!("set the {what}")),
    }
}

impl CostEstimator for ShopRates {
    fn estimate(&self, input: &CostInput) -> Result<CostEstimate, String> {
        let quantity = f64::from(input.quantity.max(1));
        let price = need(self.price_per_kg, "material price per kg")?;
        let rate = need(self.machine_rate, "machine rate per hour")?;
        let mass = input
            .mass
            .ok_or("the part has no mass: give it a material so its density is known")?;
        let volume = input.volume.unwrap_or(0.0);
        let fact = |k: &str| input.facts.get(k).copied();
        let mut lines = Vec::new();
        let (bought, material_text) = if input.process == "machining" {
            let stock = fact("stock_mass").ok_or("the part has no stock size")?;
            (
                stock,
                format!("Material: stock {stock:.3} kg × {price} per kg"),
            )
        } else {
            (
                mass,
                format!("Material: part {mass:.3} kg × {price} per kg"),
            )
        };
        lines.push((material_text, bought * price));
        let (hours, time) = match input.process.as_str() {
            "machining" => {
                let removal = need(self.removal_rate, "removal rate")?;
                let removed = (fact("stock_volume").unwrap_or(volume) - volume).max(0.0);
                let minutes = removed / removal;
                (
                    minutes / 60.0,
                    format!(
                        "{minutes:.2} min ({:.2} cm³ removed ÷ {:.2} cm³/min)",
                        removed * 1e6,
                        removal * 1e6
                    ),
                )
            }
            "additive" => {
                let build = need(self.build_rate, "build rate")?;
                let hours = volume / build;
                (
                    hours,
                    format!(
                        "{:.2} min ({:.2} cm³ ÷ {:.2} cm³/h)",
                        hours * 60.0,
                        volume * 1e6,
                        build * 1e6
                    ),
                )
            }
            _ => {
                let cycle = need(self.cycle, "cycle time")?;
                (cycle / 3600.0, format!("{cycle} s cycle"))
            }
        };
        lines.push((format!("Machine: {time} × {rate} per hour"), hours * rate));
        let setup = self.setup / quantity;
        lines.push((
            format!("Setup: {} per batch ÷ {} parts", self.setup, input.quantity),
            setup,
        ));
        if self.tooling > 0.0 {
            lines.push((
                format!("Tooling: {} ÷ {} parts", self.tooling, input.quantity),
                self.tooling / quantity,
            ));
        }
        let unit_cost = lines.iter().map(|(_, v)| v).sum();
        Ok(CostEstimate {
            currency: self.currency.clone(),
            unit_cost,
            setup_cost: self.setup + self.tooling,
            lines,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(process: &str) -> CostInput {
        CostInput {
            process: process.into(),
            material: Some("material.aluminum".into()),
            quantity: 10,
            mass: Some(0.27),
            volume: Some(1e-4),
            facts: [
                ("stock_mass".to_string(), 0.54),
                ("stock_volume".to_string(), 2e-4),
            ]
            .into_iter()
            .collect(),
        }
    }

    #[test]
    fn machining_buys_the_stock_and_removes_what_isnt_part() {
        let rates = ShopRates {
            currency: "EUR".into(),
            price_per_kg: Some(10.0),
            machine_rate: Some(60.0),
            setup: 100.0,
            removal_rate: Some(10e-6), // 10 cm³/min
            ..Default::default()
        };
        let e = rates.estimate(&input("machining")).unwrap();
        // 0.54 kg × 10 = 5.4; 100 cm³ ÷ 10 cm³/min = 10 min = 1/6 h × 60 = 10; 100 ÷ 10 = 10.
        assert!((e.unit_cost - 25.4).abs() < 1e-9, "{e:?}");
        assert_eq!(e.setup_cost, 100.0);
        assert_eq!(e.currency, "EUR");
    }

    #[test]
    fn a_missing_rate_is_named() {
        let rates = ShopRates {
            price_per_kg: Some(10.0),
            machine_rate: Some(60.0),
            ..Default::default()
        };
        assert_eq!(
            rates.estimate(&input("additive")).unwrap_err(),
            "set the build rate"
        );
        assert_eq!(
            ShopRates::default()
                .estimate(&input("molding"))
                .unwrap_err(),
            "set the material price per kg"
        );
    }

    #[test]
    fn molding_adds_the_cycle_and_spreads_the_mold() {
        let rates = ShopRates {
            price_per_kg: Some(4.0),
            machine_rate: Some(36.0),
            cycle: Some(30.0),
            tooling: 5000.0,
            ..Default::default()
        };
        let e = rates.estimate(&input("molding")).unwrap();
        // 0.27 × 4 = 1.08; 30 s = 1/120 h × 36 = 0.3; 5000 ÷ 10 = 500.
        assert!((e.unit_cost - 501.38).abs() < 1e-9, "{e:?}");
    }
}
