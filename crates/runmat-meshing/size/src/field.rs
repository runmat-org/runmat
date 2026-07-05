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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentSizingQuery {
    pub start_m: [f64; 3],
    pub end_m: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointSizingQuery {
    pub position_m: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SizingQuerySource {
    Unset,
    Global,
    LocalSample,
    AnisotropicMetric,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SizingQueryResult {
    pub target_size_m: Option<f64>,
    pub source: SizingQuerySource,
    pub contributing_sample_count: usize,
}

impl SizingQueryResult {
    pub fn unset() -> Self {
        Self {
            target_size_m: None,
            source: SizingQuerySource::Unset,
            contributing_sample_count: 0,
        }
    }
}

pub trait SizingFieldService {
    fn query_point_size(&self, query: PointSizingQuery) -> SizingQueryResult;
    fn query_segment_size(&self, query: SegmentSizingQuery) -> SizingQueryResult;
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

impl MeshSizingField {
    pub fn target_size_for_segment(&self, query: SegmentSizingQuery) -> Option<f64> {
        self.query_segment_size(query).target_size_m
    }

    pub fn target_size_at_point(&self, query: PointSizingQuery) -> Option<f64> {
        self.query_point_size(query).target_size_m
    }

    fn initial_query_result(&self) -> SizingQueryResult {
        self.global_target_size_m
            .and_then(|target_size_m| self.clamped_target_size_m(target_size_m))
            .map(|target_size_m| SizingQueryResult {
                target_size_m: Some(target_size_m),
                source: SizingQuerySource::Global,
                contributing_sample_count: 0,
            })
            .unwrap_or_else(SizingQueryResult::unset)
    }

    fn merge_query_target(
        &self,
        mut result: SizingQueryResult,
        target_size_m: f64,
        source: SizingQuerySource,
    ) -> SizingQueryResult {
        let Some(target_size_m) = self.clamped_target_size_m(target_size_m) else {
            return result;
        };
        if match result.target_size_m {
            Some(current) => target_size_m < current,
            None => true,
        } {
            result.target_size_m = Some(target_size_m);
            result.source = source;
        }
        result.contributing_sample_count += 1;
        result
    }

    pub fn clamped_target_size_m(&self, target_size_m: f64) -> Option<f64> {
        if !target_size_m.is_finite() || target_size_m <= 0.0 {
            return None;
        }
        let mut target_size_m = target_size_m;
        if let (Some(global_target_size_m), Some(growth_rate)) = (
            self.global_target_size_m
                .filter(|value| value.is_finite() && *value > 0.0),
            self.growth_rate
                .filter(|value| value.is_finite() && *value >= 1.0),
        ) {
            target_size_m = target_size_m.max(global_target_size_m / growth_rate);
        }
        if let Some(min_size_m) = self
            .min_size_m
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            target_size_m = target_size_m.max(min_size_m);
        }
        if let Some(max_size_m) = self
            .max_size_m
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            target_size_m = target_size_m.min(max_size_m);
        }
        (target_size_m.is_finite() && target_size_m > 0.0).then_some(target_size_m)
    }
}

impl SizingFieldService for MeshSizingField {
    fn query_point_size(&self, query: PointSizingQuery) -> SizingQueryResult {
        if !query.position_m.iter().all(|value| value.is_finite()) {
            return SizingQueryResult::unset();
        }
        let mut result = self.initial_query_result();
        for sample in &self.samples {
            if point_matches(sample.position_m, query.position_m) {
                result = self.merge_query_target(
                    result,
                    sample.target_size_m,
                    SizingQuerySource::LocalSample,
                );
            }
        }
        for sample in &self.anisotropic_samples {
            if !sample.is_valid_metric() || !point_matches(sample.position_m, query.position_m) {
                continue;
            }
            let sample_target_size_m = sample
                .target_sizes_m
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            result = self.merge_query_target(
                result,
                sample_target_size_m,
                SizingQuerySource::AnisotropicMetric,
            );
        }
        result
    }

    fn query_segment_size(&self, query: SegmentSizingQuery) -> SizingQueryResult {
        let mut result = self.initial_query_result();
        for sample in &self.samples {
            if point_lies_on_segment(sample.position_m, query) {
                result = self.merge_query_target(
                    result,
                    sample.target_size_m,
                    SizingQuerySource::LocalSample,
                );
            }
        }
        for sample in &self.anisotropic_samples {
            if !sample.is_valid_metric() || !point_lies_on_segment(sample.position_m, query) {
                continue;
            }
            let sample_target_size_m = sample
                .target_sizes_m
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            result = self.merge_query_target(
                result,
                sample_target_size_m,
                SizingQuerySource::AnisotropicMetric,
            );
        }
        result
    }
}

fn point_lies_on_segment(point: [f64; 3], query: SegmentSizingQuery) -> bool {
    if !point.iter().all(|value| value.is_finite())
        || !query.start_m.iter().all(|value| value.is_finite())
        || !query.end_m.iter().all(|value| value.is_finite())
    {
        return false;
    }
    let segment = sub(query.end_m, query.start_m);
    let segment_length_squared = dot(segment, segment);
    if segment_length_squared <= f64::EPSILON {
        return distance(point, query.start_m) <= 1.0e-12;
    }
    let relative = sub(point, query.start_m);
    let parameter = dot(relative, segment) / segment_length_squared;
    if !(-1.0e-12..=1.0 + 1.0e-12).contains(&parameter) {
        return false;
    }
    let closest = [
        query.start_m[0] + segment[0] * parameter.clamp(0.0, 1.0),
        query.start_m[1] + segment[1] * parameter.clamp(0.0, 1.0),
        query.start_m[2] + segment[2] * parameter.clamp(0.0, 1.0),
    ];
    let tolerance = segment_length_squared.sqrt().max(1.0) * 1.0e-9;
    distance(point, closest) <= tolerance.max(1.0e-12)
}

fn point_matches(left: [f64; 3], right: [f64; 3]) -> bool {
    if !left.iter().all(|value| value.is_finite()) || !right.iter().all(|value| value.is_finite()) {
        return false;
    }
    let scale = left
        .iter()
        .chain(right.iter())
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    distance(left, right) <= scale * 1.0e-9
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    dot(sub(left, right), sub(left, right)).sqrt()
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

    #[test]
    fn segment_sizing_query_uses_samples_on_segment() {
        let sizing = MeshSizingField {
            global_target_size_m: Some(1.0),
            samples: vec![
                SizingSample {
                    position_m: [0.5, 0.0, 0.0],
                    target_size_m: 0.2,
                    reason: Some("feature_edge".to_string()),
                },
                SizingSample {
                    position_m: [0.5, 0.2, 0.0],
                    target_size_m: 0.05,
                    reason: Some("off_edge".to_string()),
                },
            ],
            ..MeshSizingField::default()
        };

        let target_size_m = sizing
            .target_size_for_segment(SegmentSizingQuery {
                start_m: [0.0, 0.0, 0.0],
                end_m: [1.0, 0.0, 0.0],
            })
            .expect("segment target");

        assert_eq!(target_size_m, 0.2);
        let query = sizing.query_segment_size(SegmentSizingQuery {
            start_m: [0.0, 0.0, 0.0],
            end_m: [1.0, 0.0, 0.0],
        });
        assert_eq!(query.target_size_m, Some(0.2));
        assert_eq!(query.source, SizingQuerySource::LocalSample);
        assert_eq!(query.contributing_sample_count, 1);
    }

    #[test]
    fn segment_sizing_query_clamps_local_samples() {
        let sizing = MeshSizingField {
            global_target_size_m: Some(1.0),
            min_size_m: Some(0.25),
            max_size_m: Some(0.75),
            samples: vec![SizingSample {
                position_m: [0.5, 0.0, 0.0],
                target_size_m: 0.1,
                reason: Some("feature_edge".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let target_size_m = sizing
            .target_size_for_segment(SegmentSizingQuery {
                start_m: [0.0, 0.0, 0.0],
                end_m: [1.0, 0.0, 0.0],
            })
            .expect("segment target");

        assert_eq!(target_size_m, 0.25);
    }

    #[test]
    fn point_sizing_query_reports_anisotropic_metric_source() {
        let sizing = MeshSizingField {
            global_target_size_m: Some(1.0),
            samples: vec![SizingSample {
                position_m: [0.25, 0.0, 0.0],
                target_size_m: 0.4,
                reason: Some("feature".to_string()),
            }],
            anisotropic_samples: vec![AnisotropicSizingSample {
                position_m: [0.25, 0.0, 0.0],
                target_sizes_m: [0.2, 0.3, 0.5],
                directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                reason: Some("directional".to_string()),
            }],
            ..MeshSizingField::default()
        };

        let query = sizing.query_point_size(PointSizingQuery {
            position_m: [0.25, 0.0, 0.0],
        });

        assert_eq!(query.target_size_m, Some(0.2));
        assert_eq!(query.source, SizingQuerySource::AnisotropicMetric);
        assert_eq!(query.contributing_sample_count, 2);
        assert_eq!(
            sizing.target_size_at_point(PointSizingQuery {
                position_m: [0.25, 0.0, 0.0],
            }),
            Some(0.2)
        );
    }

    #[test]
    fn invalid_point_sizing_query_is_unset() {
        let sizing = MeshSizingField {
            global_target_size_m: Some(1.0),
            ..MeshSizingField::default()
        };

        let query = sizing.query_point_size(PointSizingQuery {
            position_m: [f64::NAN, 0.0, 0.0],
        });

        assert_eq!(query, SizingQueryResult::unset());
    }
}
