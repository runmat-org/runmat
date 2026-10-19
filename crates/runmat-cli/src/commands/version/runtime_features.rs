use runmat_runtime::builtin::migration_inventory::{
    runtime_feature_inventory, CrateFeatureInventory,
};
use serde::Serialize;

const SCHEMA: &str = "runmat.runtime-features.v1";

#[derive(Serialize)]
struct RuntimeFeatureReport {
    schema: &'static str,
    runtime: CrateFeatureInventory,
}

pub(super) fn json() -> serde_json::Result<String> {
    serde_json::to_string(&RuntimeFeatureReport {
        schema: SCHEMA,
        runtime: runtime_feature_inventory(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_is_a_bounded_projection_of_the_canonical_runtime_features() {
        let value: serde_json::Value = serde_json::from_str(&json().unwrap()).unwrap();
        assert_eq!(value["schema"], SCHEMA);
        assert_eq!(value["runtime"]["crate_name"], "runmat-runtime");
        assert_eq!(value["runtime"]["schema_version"], 1);
        let known = value["runtime"]["known_features"].as_array().unwrap();
        let enabled = value["runtime"]["enabled_features"].as_array().unwrap();
        assert!(known.contains(&serde_json::json!("plot-core")));
        assert!(enabled.iter().all(|feature| known.contains(feature)));
        assert_eq!(value.as_object().unwrap().len(), 2);
    }
}
