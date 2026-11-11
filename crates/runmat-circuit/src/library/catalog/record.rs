//! The orderable part record: what a `.circuit` file's `part "<mpn>" = {…}` declaration holds,
//! what the server's part search returns (server-parts uses these types under the `openapi`
//! feature), and what the `library.place` edit reads. The JSON the server sends is the JSON the
//! runtime reads. Supply data (stock, price breaks, lifecycle) comes with each [`PartHit`].

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One pin of a part: its name, the pads it lands on and its electrical type.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct PartPin {
    pub name: String,
    pub pads: Vec<String>,
    /// The ERC pin type by its written word (`input`, `power_in`, `passive` …); `passive` when
    /// absent.
    #[serde(rename = "type", default = "passive")]
    pub kind: String,
}

impl Default for PartPin {
    fn default() -> Self {
        Self {
            name: String::new(),
            pads: vec![],
            kind: passive(),
        }
    }
}

fn passive() -> String {
    "passive".into()
}

/// A part's SPICE model: a model or subcircuit name, the library file that defines it, and the
/// subcircuit's node order as pin names.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct PartSpice {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library: Option<String>,
    /// The subcircuit's nodes in order, each a pin name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<String>,
    /// A primitive device letter (`D`, `Q`, `M`) when the model is a `.model` card.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

/// An orderable part as a catalog describes it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct PartRecord {
    pub mpn: String,
    pub manufacturer: String,
    #[serde(default)]
    pub description: String,
    /// What kind of part (`regulator`, `resistor`, `mcu` …), for search and the designator.
    #[serde(default)]
    pub category: String,
    /// Designator prefix (`U`, `R`); empty takes one from the category.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prefix: String,
    /// A footprint-generator string (`SOT23-6`, `R0402`), a footprint file read in place for
    /// interop, or `none` for a part wired in the field; empty when [`PartRecord::footprint`]
    /// names a declared footprint. A search hit always carries it.
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(required = true))]
    pub package: String,
    /// The `footprint` declaration the part's `package` names by identifier
    /// (`package = gps_lcc18`), from the file or a library it uses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint: Option<String>,
    /// The `symbol` declaration the part draws with; without one the symbol is generated from
    /// the pins. Part pins map to the symbol's pins by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    /// The part's 3D body, an MCAD `.part` or a STEP file, over its footprint's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    /// Pads the part leaves without a pin on purpose (`nc = ["9"]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nc: Vec<String>,
    pub pins: Vec<PartPin>,
    /// Parametrics as written quantities (`vin_max = "36 V"`) or text.
    #[serde(default)]
    pub params: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datasheet: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spice: Option<PartSpice>,
}

/// One distributor's offer for a part.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct PartOffer {
    pub supplier: String,
    pub sku: String,
    #[serde(default)]
    pub stock: Option<u64>,
    /// `[quantity, unit price]` breaks, ascending by quantity.
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Vec<Vec<f64>>))]
    pub price_breaks: Vec<(u64, f64)>,
    #[serde(default)]
    pub currency: String,
    /// The label of the part source that answered with this offer; empty when the offer came
    /// from no search.
    #[serde(default)]
    pub source: String,
}

/// One part search result: the part, its lifecycle and the offers for it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(rename_all = "camelCase")]
pub struct PartHit {
    pub part: PartRecord,
    /// `active`, `nrnd`, `last_time_buy`, `obsolete`, `preview`, or empty when unknown.
    #[serde(default)]
    pub lifecycle: String,
    #[serde(default)]
    pub offers: Vec<PartOffer>,
}

impl PartHit {
    /// Stock summed over the offers.
    pub fn stock(&self) -> u64 {
        self.offers.iter().filter_map(|o| o.stock).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hit_reads_from_the_search_json_with_defaults() {
        let json = r#"{"part":{"mpn":"X","manufacturer":"M","package":"R0402","pins":[{"name":"1","pads":["1"]},{"name":"2","pads":["2"],"type":"input"}]},
            "lifecycle":"active","offers":[{"supplier":"S","sku":"S-1","stock":1200,"priceBreaks":[[1,0.1],[100,0.02]],"currency":"USD","source":"sample"},{"supplier":"T","sku":"T-1","stock":30}]}"#;
        let hit: PartHit = serde_json::from_str(json).unwrap();
        assert_eq!(hit.part.pins[0].kind, "passive");
        assert_eq!(hit.part.pins[1].kind, "input");
        assert_eq!(hit.offers[0].price_breaks[1], (100, 0.02));
        assert_eq!(hit.offers[0].source, "sample");
        assert_eq!(hit.stock(), 1230);
        let back: PartHit = serde_json::from_str(&serde_json::to_string(&hit).unwrap()).unwrap();
        assert_eq!(back, hit);
    }

    #[test]
    fn a_default_pin_is_passive() {
        assert_eq!(PartPin::default().kind, "passive");
    }
}
