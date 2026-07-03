use serde::{Deserialize, Serialize};

pub const MODULE_PURPOSE: &str = "composable sizing queries for every meshing stage";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SizingSample {
    pub position_m: [f64; 3],
    pub target_size_m: f64,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnisotropicSizingSample {
    pub position_m: [f64; 3],
    pub target_sizes_m: [f64; 3],
    pub directions: [[f64; 3]; 3],
    #[serde(default)]
    pub reason: Option<String>,
}

impl AnisotropicSizingSample {
    pub fn is_valid_metric(&self) -> bool {
        self.position_m.iter().all(|value| value.is_finite())
            && self
                .target_sizes_m
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && directions_are_finite_orthonormal(self.directions)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SizingSampleRejection {
    pub position_m: [f64; 3],
    pub target_size_m: f64,
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SizingSampleApplication {
    pub position_m: [f64; 3],
    pub target_size_m: f64,
    pub inserted_breakpoint_count: usize,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshSizingField {
    #[serde(default)]
    pub global_target_size_m: Option<f64>,
    #[serde(default)]
    pub min_size_m: Option<f64>,
    #[serde(default)]
    pub max_size_m: Option<f64>,
    #[serde(default)]
    pub growth_rate: Option<f64>,
    #[serde(default)]
    pub samples: Vec<SizingSample>,
    #[serde(default)]
    pub anisotropic_samples: Vec<AnisotropicSizingSample>,
    #[serde(default)]
    pub applied_samples: Vec<SizingSampleApplication>,
    #[serde(default)]
    pub rejected_samples: Vec<SizingSampleRejection>,
}

fn directions_are_finite_orthonormal(directions: [[f64; 3]; 3]) -> bool {
    directions
        .iter()
        .all(|direction| direction.iter().all(|value| value.is_finite()))
        && directions.iter().all(|direction| {
            let norm_squared = dot(*direction, *direction);
            (norm_squared - 1.0).abs() <= 1.0e-9
        })
        && dot(directions[0], directions[1]).abs() <= 1.0e-9
        && dot(directions[0], directions[2]).abs() <= 1.0e-9
        && dot(directions[1], directions[2]).abs() <= 1.0e-9
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anisotropic_sizing_sample_validates_metric_contract() {
        let valid = AnisotropicSizingSample {
            position_m: [0.1, 0.2, 0.3],
            target_sizes_m: [0.01, 0.02, 0.04],
            directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            reason: Some("boundary_layer".to_string()),
        };
        assert!(valid.is_valid_metric());

        let mut invalid_size = valid.clone();
        invalid_size.target_sizes_m[1] = 0.0;
        assert!(!invalid_size.is_valid_metric());

        let mut non_orthogonal = valid;
        non_orthogonal.directions[1] = [1.0, 0.0, 0.0];
        assert!(!non_orthogonal.is_valid_metric());
    }

    #[test]
    fn mesh_sizing_field_deserializes_without_anisotropic_samples() {
        let sizing: MeshSizingField =
            serde_json::from_str(r#"{"global_target_size_m":0.1}"#).expect("sizing field");
        assert_eq!(sizing.global_target_size_m, Some(0.1));
        assert!(sizing.anisotropic_samples.is_empty());
    }
}
