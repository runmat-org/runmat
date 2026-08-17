use serde::{Deserialize, Serialize};

use super::{
    BodyMassProperties, CurveEvaluatorId, MassPropertiesEvaluatorId, ParameterRange,
    PcurveEvaluatorId, SurfaceEvaluatorId, TrimClassifierId,
};

pub const EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactEvaluatorRegistry {
    pub schema_version: u16,
    pub kernel_abi: String,
    pub curves: Vec<ExactCurveEvaluatorRecord>,
    pub pcurves: Vec<ExactPcurveEvaluatorRecord>,
    pub surfaces: Vec<ExactSurfaceEvaluatorRecord>,
    pub trim_classifiers: Vec<ExactTrimClassifierRecord>,
    pub mass_properties: Vec<ExactMassPropertiesRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactCurveEvaluatorRecord {
    pub id: CurveEvaluatorId,
    pub implementation: ExactCurveImplementation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactCurveImplementation {
    Portable { definition: ExactCurveDefinition },
    Kernel { reference: KernelEvaluatorRef },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactCurveDefinition {
    Line {
        origin_m: [f64; 3],
        direction_m_per_parameter: [f64; 3],
        domain: ParameterRange,
    },
    Circle {
        center_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        radius_m: f64,
        domain: ParameterRange,
    },
    Ellipse {
        center_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        major_radius_m: f64,
        minor_radius_m: f64,
        domain: ParameterRange,
    },
    Nurbs {
        definition: NurbsCurve3,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactPcurveEvaluatorRecord {
    pub id: PcurveEvaluatorId,
    pub implementation: ExactPcurveImplementation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactPcurveImplementation {
    Portable { definition: ExactPcurveDefinition },
    Kernel { reference: KernelEvaluatorRef },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactPcurveDefinition {
    Line {
        origin_uv: [f64; 2],
        direction_uv_per_parameter: [f64; 2],
        domain: ParameterRange,
    },
    Circle {
        center_uv: [f64; 2],
        x_axis_uv: [f64; 2],
        y_axis_uv: [f64; 2],
        radius_uv: f64,
        domain: ParameterRange,
    },
    Nurbs {
        definition: NurbsCurve2,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSurfaceEvaluatorRecord {
    pub id: SurfaceEvaluatorId,
    pub implementation: ExactSurfaceImplementation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactSurfaceImplementation {
    Portable { definition: ExactSurfaceDefinition },
    Kernel { reference: KernelEvaluatorRef },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactSurfaceDefinition {
    Plane {
        origin_m: [f64; 3],
        u_axis_m_per_parameter: [f64; 3],
        v_axis_m_per_parameter: [f64; 3],
        domains: [ParameterRange; 2],
    },
    Cylinder {
        origin_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        axis_m_per_v: [f64; 3],
        radius_m: f64,
        domains: [ParameterRange; 2],
    },
    Cone {
        apex_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        axis: [f64; 3],
        semi_angle_rad: f64,
        domains: [ParameterRange; 2],
    },
    Sphere {
        center_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        z_axis: [f64; 3],
        radius_m: f64,
        domains: [ParameterRange; 2],
    },
    Torus {
        center_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        z_axis: [f64; 3],
        major_radius_m: f64,
        minor_radius_m: f64,
        domains: [ParameterRange; 2],
    },
    Nurbs {
        definition: NurbsSurface3,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NurbsCurve3 {
    pub degree: u8,
    pub knots: Vec<f64>,
    pub control_points_m: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub domain: ParameterRange,
    pub periodic: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NurbsCurve2 {
    pub degree: u8,
    pub knots: Vec<f64>,
    pub control_points_uv: Vec<[f64; 2]>,
    pub weights: Vec<f64>,
    pub domain: ParameterRange,
    pub periodic: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NurbsSurface3 {
    pub u_degree: u8,
    pub v_degree: u8,
    pub u_knots: Vec<f64>,
    pub v_knots: Vec<f64>,
    pub u_control_count: u32,
    pub v_control_count: u32,
    /// U-major tensor grid: `control_points_m[u * v_control_count + v]`.
    pub control_points_m: Vec<[f64; 3]>,
    /// Uses the same U-major tensor-grid order as `control_points_m`.
    pub weights: Vec<f64>,
    pub domains: [ParameterRange; 2],
    pub periodic_u: bool,
    pub periodic_v: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelEvaluatorRef {
    pub entity_token: String,
    pub representation_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactTrimClassifierRecord {
    pub id: TrimClassifierId,
    pub implementation: ExactTrimClassifierImplementation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactTrimClassifierImplementation {
    OrientedPcurveWinding,
    Kernel { reference: KernelEvaluatorRef },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactMassPropertiesRecord {
    pub id: MassPropertiesEvaluatorId,
    pub implementation: ExactMassPropertiesImplementation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactMassPropertiesImplementation {
    Kernel {
        reference: KernelEvaluatorRef,
    },
    KernelValidated {
        properties: BodyMassProperties,
        validation_digest: [u8; 32],
    },
}
