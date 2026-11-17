//! `evaluate.cost`: the part's cost for a process and a quantity from the shop's rates you
//! enter ([`runmat_cad::evaluate::cost::ShopRates`]): every input is shown in the results
//! header, and each line of the estimate is a row saying how it was worked out. The part's
//! mass and volume come from Mass properties; machining's stock is the part's box with the
//! stock allowance on each side.
//!
//! Values: `process`, `quantity`, `currency`, `price` (per kg), `rate` (per machine hour),
//! `setup` (per batch), `tooling` (per batch), `removal_rate` (cm³/min, machining),
//! `build_rate` (cm³/h, additive), `cycle` (s per part, molding and sheet metal),
//! `allowance` (mm per side, machining).

use super::{Analysis, Row};
use runmat_cad::evaluate::cost::ShopRates;
use runmat_cad::evaluate::{CostEstimator, CostInput};
use serde_json::{json, Map, Value};

fn number(values: &Map<String, Value>, key: &str) -> Option<f64> {
    values.get(key).and_then(Value::as_f64)
}

pub(super) fn cost(
    path: &str,
    text: &str,
    configuration: Option<&str>,
    values: &Map<String, Value>,
) -> Result<Analysis, String> {
    let process = values
        .get("process")
        .and_then(Value::as_str)
        .unwrap_or("machining")
        .to_owned();
    let quantity = values
        .get("quantity")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .clamp(1, u64::from(u32::MAX)) as u32;
    let currency = values
        .get("currency")
        .and_then(Value::as_str)
        .filter(|c| !c.trim().is_empty())
        .unwrap_or("")
        .trim()
        .to_owned();
    let report = crate::mass::part_mass(path, text, configuration)?;
    let part = report.parts.first();
    let density = part.and_then(|p| p.density);
    let mass = part.and_then(|p| p.mass);
    let volume = report.volume;
    // Machining's stock: the part's box (from its display) with the allowance on each side.
    let allowance = number(values, "allowance").unwrap_or(0.0).max(0.0) / 1000.0;
    let mut facts = std::collections::BTreeMap::new();
    let mut stock_text = None;
    if process == "machining" {
        let regenerated =
            crate::runtime::regenerate_part_in(path, text, configuration, &Default::default())
                .map_err(|d| d.first().map(|d| d.message.clone()).unwrap_or_default())?;
        let b = regenerated
            .display
            .bounds
            .as_ref()
            .ok_or("the part has no bodies to cost")?;
        let size: Vec<f64> = (0..3)
            .map(|i| f64::from(b.max[i] - b.min[i]) + 2.0 * allowance)
            .collect();
        let stock = size[0] * size[1] * size[2];
        facts.insert("stock_volume".to_string(), stock);
        if let Some(d) = density {
            facts.insert("stock_mass".to_string(), stock * d);
        }
        stock_text = Some(format!(
            "Stock: {:.2} × {:.2} × {:.2} mm",
            size[0] * 1000.0,
            size[1] * 1000.0,
            size[2] * 1000.0
        ));
    }
    let rates = ShopRates {
        currency: currency.clone(),
        price_per_kg: number(values, "price"),
        machine_rate: number(values, "rate"),
        setup: number(values, "setup").unwrap_or(0.0).max(0.0),
        tooling: number(values, "tooling").unwrap_or(0.0).max(0.0),
        // cm³/min and cm³/h in the form; m³ per minute and per hour here.
        removal_rate: number(values, "removal_rate").map(|v| v * 1e-6),
        build_rate: number(values, "build_rate").map(|v| v * 1e-6),
        cycle: number(values, "cycle"),
    };
    let input = CostInput {
        process: process.clone(),
        material: part.and_then(|p| p.material.clone()),
        quantity,
        mass,
        volume: Some(volume),
        facts,
    };
    let estimate = rates.estimate(&input)?;
    let money = |v: f64| {
        if currency.is_empty() {
            format!("{v:.2}")
        } else {
            format!("{v:.2} {currency}")
        }
    };
    let rows: Vec<Row> = estimate
        .lines
        .iter()
        .map(|(line, amount)| Row {
            cells: vec![line.clone(), money(*amount)],
            items: Vec::new(),
            severity: None,
            legend: None,
            reference: None,
        })
        .collect();
    let process_label = match process.as_str() {
        "sheet_metal" => "Sheet metal",
        "molding" => "Molding",
        "additive" => "Additive",
        _ => "Machining",
    };
    let mut settings = vec![
        format!("Process: {process_label}"),
        format!("Quantity: {quantity}"),
        format!(
            "Material price: {} per kg",
            rates.price_per_kg.unwrap_or_default()
        ),
        format!(
            "Machine rate: {} per hour",
            rates.machine_rate.unwrap_or_default()
        ),
        format!("Setup: {} per batch", rates.setup),
    ];
    if rates.tooling > 0.0 {
        settings.push(format!("Tooling: {}", rates.tooling));
    }
    match process.as_str() {
        "machining" => {
            settings.push(format!(
                "Removal rate: {} cm³/min",
                number(values, "removal_rate").unwrap_or_default()
            ));
            settings.push(format!(
                "Stock allowance: {} mm per side",
                allowance * 1000.0
            ));
            settings.extend(stock_text);
        }
        "additive" => settings.push(format!(
            "Build rate: {} cm³/h",
            number(values, "build_rate").unwrap_or_default()
        )),
        _ => settings.push(format!(
            "Cycle: {} s per part",
            rates.cycle.unwrap_or_default()
        )),
    }
    Ok(Analysis {
        title: "Cost estimate".into(),
        summary: format!(
            "{process_label} · {} per part · {} for {quantity}",
            money(estimate.unit_cost),
            money(estimate.unit_cost * f64::from(quantity))
        ),
        columns: ["Line", "Per part"].map(String::from).to_vec(),
        rows,
        problems: Vec::new(),
        data: json!({ "input": input, "rates": rates, "estimate": estimate }),
        colors: None,
        settings,
        arrow: None,
        dimensions: Vec::new(),
        field: None,
    })
}
