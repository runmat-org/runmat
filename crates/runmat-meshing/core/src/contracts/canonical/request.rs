use serde::{Deserialize, Serialize};

use super::{
    validate_finite, validate_token, CancellationPolicy, GeometryTolerancePolicy,
    MeshingContractError, MetricFieldRequest,
};

pub const MESHING_REQUEST_SCHEMA_VERSION: u16 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementOrder {
    Tet4,
    Tet10,
}

/// Topological dimension of the solver elements a request produces.
///
/// `Solid` meshes closed solids with tetrahedra (and their boundary faces and edges).
/// `Surface` meshes sheet faces with linear triangles and their edges with linear segments,
/// the discretization of MITC3 shells and Euler-Bernoulli beams. `Curve` discretizes only the
/// exact edges, for beam frames. Surface and curve meshes are linear (`ElementOrder::Tet4`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshDimension {
    #[default]
    Solid,
    Surface,
    Curve,
}

impl MeshDimension {
    pub const fn is_solid(&self) -> bool {
        matches!(self, Self::Solid)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AlgorithmVersionSet {
    pub geometry: String,
    pub curve: String,
    pub surface: String,
    pub plc: String,
    pub tetrahedron: String,
    pub optimization: String,
    pub validation: String,
}

impl AlgorithmVersionSet {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, version) in [
            ("geometry algorithm version", self.geometry.as_str()),
            ("curve algorithm version", self.curve.as_str()),
            ("surface algorithm version", self.surface.as_str()),
            ("PLC algorithm version", self.plc.as_str()),
            ("tetrahedron algorithm version", self.tetrahedron.as_str()),
            ("optimization algorithm version", self.optimization.as_str()),
            ("validation algorithm version", self.validation.as_str()),
        ] {
            validate_token(field, version, 128)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceQualityTargets {
    pub minimum_metric_edge_length: f64,
    pub minimum_metric_angle_degrees: f64,
    pub maximum_physical_aspect_ratio: f64,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub maximum_chordal_deviation_m: runmat_types::physical::Length,
    pub maximum_normal_deviation_degrees: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurveQualityTargets {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub maximum_chordal_deviation_m: runmat_types::physical::Length,
    pub maximum_tangent_change_degrees: f64,
    pub minimum_metric_edge_length: f64,
    pub maximum_metric_edge_length: f64,
}

impl CurveQualityTargets {
    fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, value) in [
            (
                "maximum curve chordal deviation",
                self.maximum_chordal_deviation_m.canonical_value(),
            ),
            (
                "maximum curve tangent change",
                self.maximum_tangent_change_degrees,
            ),
            (
                "minimum curve metric edge length",
                self.minimum_metric_edge_length,
            ),
            (
                "maximum curve metric edge length",
                self.maximum_metric_edge_length,
            ),
        ] {
            validate_finite(field, value)?;
        }
        if self.maximum_chordal_deviation_m.canonical_value() <= 0.0
            || !(0.0..=180.0).contains(&self.maximum_tangent_change_degrees)
            || self.maximum_tangent_change_degrees == 0.0
            || self.minimum_metric_edge_length <= 0.0
            || self.minimum_metric_edge_length > self.maximum_metric_edge_length
        {
            return Err(MeshingContractError::invalid(
                "curve quality targets",
                "chordal, tangent, and ordered metric-length bounds must be positive",
            ));
        }
        Ok(())
    }
}

impl SurfaceQualityTargets {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, value) in [
            (
                "minimum surface metric edge length",
                self.minimum_metric_edge_length,
            ),
            (
                "minimum surface metric angle",
                self.minimum_metric_angle_degrees,
            ),
            (
                "maximum surface aspect ratio",
                self.maximum_physical_aspect_ratio,
            ),
            (
                "maximum chordal deviation",
                self.maximum_chordal_deviation_m.canonical_value(),
            ),
            (
                "maximum normal deviation",
                self.maximum_normal_deviation_degrees,
            ),
        ] {
            validate_finite(field, value)?;
        }
        if !(0.0..=1.0).contains(&self.minimum_metric_edge_length)
            || self.minimum_metric_edge_length == 0.0
            || !(0.0..60.0).contains(&self.minimum_metric_angle_degrees)
            || self.maximum_physical_aspect_ratio < 1.0
            || self.maximum_chordal_deviation_m.canonical_value() <= 0.0
            || !(0.0..=180.0).contains(&self.maximum_normal_deviation_degrees)
        {
            return Err(MeshingContractError::invalid(
                "surface quality targets",
                "metric edge, angle, aspect, deviation, or normal bound is outside its valid range",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolumeQualityTargets {
    pub maximum_radius_edge_ratio: f64,
    pub minimum_scaled_jacobian: f64,
    pub maximum_metric_edge_length: f64,
}

impl VolumeQualityTargets {
    fn validate(&self) -> Result<(), MeshingContractError> {
        for (field, value) in [
            ("maximum radius-edge ratio", self.maximum_radius_edge_ratio),
            ("minimum scaled Jacobian", self.minimum_scaled_jacobian),
            (
                "maximum metric edge length",
                self.maximum_metric_edge_length,
            ),
        ] {
            validate_finite(field, value)?;
        }
        if self.maximum_radius_edge_ratio <= 0.0
            || self.minimum_scaled_jacobian <= 0.0
            || self.minimum_scaled_jacobian > 1.0
            || self.maximum_metric_edge_length <= 0.0
        {
            return Err(MeshingContractError::invalid(
                "volume quality targets",
                "radius-edge, Jacobian, and metric-edge bounds must be positive and normalized",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingQualityTargets {
    pub curve: CurveQualityTargets,
    pub surface: SurfaceQualityTargets,
    pub volume: VolumeQualityTargets,
}

impl MeshingQualityTargets {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        self.curve.validate()?;
        self.surface.validate()?;
        self.volume.validate()
    }
}

/// The size of the mesh asked for. Part of the request's identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingSizeLimits {
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_nodes: u64,
    #[serde(with = "runmat_types::wire_integer")]
    pub maximum_elements: u64,
    /// Bounds the exact geometric recursions (surface acceptance subdivision, segment split
    /// depth, restart passes); it describes the admitted construction, not the machine.
    pub maximum_recursion_depth: u32,
}

impl MeshingSizeLimits {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.maximum_nodes == 0
            || self.maximum_elements == 0
            || self.maximum_recursion_depth == 0
        {
            return Err(MeshingContractError::invalid(
                "meshing size limits",
                "every size limit must be non-zero",
            ));
        }
        Ok(())
    }
}

/// Machine ceilings for one meshing run. They decide only whether the run completes, never
/// what it produces, so they are not part of the request's identity: the runner resolves them
/// from the host's capacity (or an explicit setting) and the evidence records the ceilings that
/// were applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingExecutionCeilings {
    pub maximum_memory_bytes: u64,
    pub maximum_scratch_bytes: u64,
    pub maximum_wall_time_ms: u64,
    pub maximum_artifact_bytes: u64,
    pub maximum_search_work: u64,
    pub maximum_iterations: u64,
}

impl MeshingExecutionCeilings {
    /// Ceilings for a host whose capacity is unknown (tests, portable fixtures).
    pub const PORTABLE_DEFAULT: Self = Self {
        maximum_memory_bytes: 512 << 20,
        maximum_scratch_bytes: 512 << 20,
        maximum_wall_time_ms: 3_600_000,
        maximum_artifact_bytes: 512 << 20,
        maximum_search_work: 1_000_000_000,
        maximum_iterations: 1_000_000_000,
    };

    /// A wall-time ceiling for a run admitted no wall-time budget: the largest wire integer.
    /// The run's own deadline, the caller's timeout and cancellation bound such a run.
    pub const UNBOUNDED_WALL_TIME_MS: u64 = runmat_types::wire_integer::MAX_WIRE_INTEGER;

    /// Ceilings for a run admitted `memory_bytes` of resident memory and the requested mesh:
    /// memory, scratch and artifact bytes from that budget; search work and iterations
    /// proportional to the requested element count (honest accounting charges per element). The
    /// run is admitted no wall-time budget, so the wall-time ceiling is unbounded; see
    /// [`Self::for_run_budget`].
    pub fn for_host(memory_bytes: u64, limits: &MeshingSizeLimits) -> Self {
        Self::for_run_budget(memory_bytes, None, limits)
    }

    /// Ceilings from the run's admitted budget: `memory_bytes` of resident memory and
    /// `wall_time_ms` of wall time (`None` when the run is admitted no wall-time budget). A
    /// fixed hour or a host-load snapshot is not a budget: a finished mesh on a loaded host must
    /// not be rejected for taking longer than an unrelated constant.
    pub fn for_run_budget(
        memory_bytes: u64,
        wall_time_ms: Option<u64>,
        limits: &MeshingSizeLimits,
    ) -> Self {
        const WORK_PER_ELEMENT: u64 = 100_000;
        let memory = memory_bytes.max(Self::PORTABLE_DEFAULT.maximum_memory_bytes);
        let work = limits
            .maximum_elements
            .saturating_mul(WORK_PER_ELEMENT)
            .max(Self::PORTABLE_DEFAULT.maximum_search_work);
        Self {
            maximum_memory_bytes: memory,
            maximum_scratch_bytes: memory,
            maximum_wall_time_ms: wall_time_ms
                .unwrap_or(Self::UNBOUNDED_WALL_TIME_MS)
                .clamp(1, Self::UNBOUNDED_WALL_TIME_MS),
            maximum_artifact_bytes: memory,
            maximum_search_work: work,
            maximum_iterations: work,
        }
    }

    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if [
            self.maximum_memory_bytes,
            self.maximum_scratch_bytes,
            self.maximum_wall_time_ms,
            self.maximum_artifact_bytes,
            self.maximum_search_work,
            self.maximum_iterations,
        ]
        .contains(&0)
        {
            return Err(MeshingContractError::invalid(
                "meshing execution ceilings",
                "every ceiling must be non-zero",
            ));
        }
        Ok(())
    }
}

/// The complete hard budget one run executes under: the request's size limits and the applied
/// execution ceilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingResourceBudget {
    pub maximum_nodes: u64,
    pub maximum_elements: u64,
    pub maximum_memory_bytes: u64,
    pub maximum_scratch_bytes: u64,
    pub maximum_wall_time_ms: u64,
    pub maximum_artifact_bytes: u64,
    pub maximum_search_work: u64,
    pub maximum_recursion_depth: u32,
    pub maximum_iterations: u64,
}

impl MeshingResourceBudget {
    pub fn new(limits: MeshingSizeLimits, ceilings: MeshingExecutionCeilings) -> Self {
        Self {
            maximum_nodes: limits.maximum_nodes,
            maximum_elements: limits.maximum_elements,
            maximum_memory_bytes: ceilings.maximum_memory_bytes,
            maximum_scratch_bytes: ceilings.maximum_scratch_bytes,
            maximum_wall_time_ms: ceilings.maximum_wall_time_ms,
            maximum_artifact_bytes: ceilings.maximum_artifact_bytes,
            maximum_search_work: ceilings.maximum_search_work,
            maximum_recursion_depth: limits.maximum_recursion_depth,
            maximum_iterations: ceilings.maximum_iterations,
        }
    }

    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if [
            self.maximum_nodes,
            self.maximum_elements,
            self.maximum_memory_bytes,
            self.maximum_scratch_bytes,
            self.maximum_wall_time_ms,
            self.maximum_artifact_bytes,
            self.maximum_search_work,
            u64::from(self.maximum_recursion_depth),
            self.maximum_iterations,
        ]
        .contains(&0)
        {
            return Err(MeshingContractError::invalid(
                "meshing resource budget",
                "every hard budget must be non-zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshingRequest {
    pub schema_version: u16,
    pub element_order: ElementOrder,
    /// Omitted from the canonical encoding for solid meshes so solid request identities are
    /// unchanged by the surface and curve dimensions.
    #[serde(default, skip_serializing_if = "MeshDimension::is_solid")]
    pub dimension: MeshDimension,
    pub deterministic_seed: u64,
    pub algorithms: AlgorithmVersionSet,
    pub tolerance: GeometryTolerancePolicy,
    pub metric: MetricFieldRequest,
    pub quality: MeshingQualityTargets,
    pub limits: MeshingSizeLimits,
    pub cancellation: CancellationPolicy,
    /// Authored local sizing controls on named geometry regions. Omitted when empty so requests
    /// without local controls keep their identities.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub size_controls: Vec<super::MeshSizeControl>,
}

impl MeshingRequest {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != MESHING_REQUEST_SCHEMA_VERSION {
            return Err(MeshingContractError::invalid(
                "meshing request schema version",
                format!("expected {MESHING_REQUEST_SCHEMA_VERSION}"),
            ));
        }
        if !self.dimension.is_solid() && self.element_order != ElementOrder::Tet4 {
            return Err(MeshingContractError::invalid(
                "meshing request dimension",
                "surface and curve meshes use linear elements",
            ));
        }
        self.algorithms.validate()?;
        self.tolerance.validate()?;
        self.metric.validate()?;
        self.quality.validate()?;
        self.limits.validate()?;
        super::validate_size_controls(&self.size_controls)?;
        self.cancellation.validate()
    }
}
