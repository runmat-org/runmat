use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

pub const ANALYSIS_MESH_SUMMARY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMeshSummary {
    pub schema_version: u16,
    pub mesh_id: String,
    pub solve_ready: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_error_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_error_message: Option<String>,
    pub topology: AnalysisMeshTopologySummary,
    pub regions: AnalysisMeshRegionSummary,
}

impl AnalysisMeshSummary {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.schema_version != ANALYSIS_MESH_SUMMARY_SCHEMA_VERSION {
            return Err("analysis mesh summary schema version is unsupported");
        }
        if self.mesh_id.trim().is_empty() {
            return Err("analysis mesh summary mesh_id must be non-empty");
        }
        if self.solve_ready
            && (self.topology.node_count == 0
                || self.topology.volume_element_count == 0
                || self.regions.physical_regions.is_empty())
        {
            return Err("solve-ready analysis mesh summary must contain volume topology");
        }
        validate_bounds(&self.topology)?;

        let mut region_ids = BTreeSet::new();
        for region in &self.regions.physical_regions {
            if region.region_id.trim().is_empty()
                || region.element_count == 0
                || !region.volume_m3.is_finite()
                || region.volume_m3 < 0.0
                || !region_ids.insert(region.region_id.as_str())
            {
                return Err("analysis mesh summary physical regions are invalid");
            }
        }
        for region in &self.regions.boundary_regions {
            if region.region_id.trim().is_empty()
                || region.face_count == 0
                || !region_ids.insert(region.region_id.as_str())
            {
                return Err("analysis mesh summary boundary regions are invalid");
            }
        }
        Ok(())
    }
}

fn validate_bounds(topology: &AnalysisMeshTopologySummary) -> Result<(), &'static str> {
    match (topology.bounds_min_m, topology.bounds_max_m) {
        (None, None) => Ok(()),
        (Some(minimum), Some(maximum))
            if minimum.into_iter().zip(maximum).all(|(minimum, maximum)| {
                minimum.is_finite() && maximum.is_finite() && minimum <= maximum
            }) =>
        {
            Ok(())
        }
        _ => Err("analysis mesh summary bounds are invalid"),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMeshTopologySummary {
    pub node_count: usize,
    pub volume_element_count: usize,
    pub boundary_face_count: usize,
    pub boundary_edge_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds_min_m: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds_max_m: Option<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMeshRegionSummary {
    pub physical_regions: Vec<AnalysisMeshPhysicalRegion>,
    pub boundary_regions: Vec<AnalysisMeshBoundaryRegion>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMeshPhysicalRegion {
    pub region_id: String,
    pub element_count: usize,
    #[serde(default)]
    pub volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisMeshBoundaryRegion {
    pub region_id: String,
    pub face_count: usize,
    pub edge_count: usize,
    pub fully_recovered: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_summary() -> AnalysisMeshSummary {
        AnalysisMeshSummary {
            schema_version: ANALYSIS_MESH_SUMMARY_SCHEMA_VERSION,
            mesh_id: "mesh".to_owned(),
            solve_ready: true,
            validation_error_code: None,
            validation_error_message: None,
            topology: AnalysisMeshTopologySummary {
                node_count: 4,
                volume_element_count: 1,
                boundary_face_count: 4,
                boundary_edge_count: 6,
                bounds_min_m: Some([0.0; 3]),
                bounds_max_m: Some([1.0; 3]),
            },
            regions: AnalysisMeshRegionSummary {
                physical_regions: vec![AnalysisMeshPhysicalRegion {
                    region_id: "body".to_owned(),
                    element_count: 1,
                    volume_m3: 1.0 / 6.0,
                }],
                boundary_regions: vec![AnalysisMeshBoundaryRegion {
                    region_id: "face".to_owned(),
                    face_count: 1,
                    edge_count: 3,
                    fully_recovered: true,
                }],
            },
        }
    }

    #[test]
    fn validation_rejects_unsupported_duplicate_and_non_finite_inventory() {
        let mut summary = valid_summary();
        assert_eq!(summary.validate(), Ok(()));

        summary.schema_version += 1;
        assert_eq!(
            summary.validate(),
            Err("analysis mesh summary schema version is unsupported")
        );
        summary = valid_summary();
        summary.regions.boundary_regions[0].region_id = "body".to_owned();
        assert_eq!(
            summary.validate(),
            Err("analysis mesh summary boundary regions are invalid")
        );
        summary = valid_summary();
        summary.topology.bounds_max_m = Some([f64::NAN, 1.0, 1.0]);
        assert_eq!(
            summary.validate(),
            Err("analysis mesh summary bounds are invalid")
        );
    }
}
