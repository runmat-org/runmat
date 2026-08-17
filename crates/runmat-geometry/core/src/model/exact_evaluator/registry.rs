use serde::{Deserialize, Serialize};

use super::{
    BodyMassPropertiesV2, CurveEvaluatorIdV2, MassPropertiesEvaluatorIdV2, ParameterRangeV2,
    PcurveEvaluatorIdV2, SurfaceEvaluatorIdV2, TrimClassifierIdV2,
};

pub const EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION: u16 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactEvaluatorRegistryV2 {
    pub schema_version: u16,
    pub kernel_abi: String,
    pub curves: Vec<ExactCurveEvaluatorRecordV2>,
    pub pcurves: Vec<ExactPcurveEvaluatorRecordV2>,
    pub surfaces: Vec<ExactSurfaceEvaluatorRecordV2>,
    pub trim_classifiers: Vec<ExactTrimClassifierRecordV2>,
    pub mass_properties: Vec<ExactMassPropertiesRecordV2>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactCurveEvaluatorRecordV2 {
    pub id: CurveEvaluatorIdV2,
    pub implementation: ExactCurveImplementationV2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactCurveImplementationV2 {
    Portable { definition: ExactCurveDefinitionV2 },
    Kernel { reference: KernelEvaluatorRefV2 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactCurveDefinitionV2 {
    Line {
        origin_m: [f64; 3],
        direction_m_per_parameter: [f64; 3],
        domain: ParameterRangeV2,
    },
    Circle {
        center_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        radius_m: f64,
        domain: ParameterRangeV2,
    },
    Ellipse {
        center_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        major_radius_m: f64,
        minor_radius_m: f64,
        domain: ParameterRangeV2,
    },
    Nurbs {
        definition: NurbsCurve3V2,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactPcurveEvaluatorRecordV2 {
    pub id: PcurveEvaluatorIdV2,
    pub implementation: ExactPcurveImplementationV2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactPcurveImplementationV2 {
    Portable { definition: ExactPcurveDefinitionV2 },
    Kernel { reference: KernelEvaluatorRefV2 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactPcurveDefinitionV2 {
    Line {
        origin_uv: [f64; 2],
        direction_uv_per_parameter: [f64; 2],
        domain: ParameterRangeV2,
    },
    Circle {
        center_uv: [f64; 2],
        x_axis_uv: [f64; 2],
        y_axis_uv: [f64; 2],
        radius_uv: f64,
        domain: ParameterRangeV2,
    },
    Nurbs {
        definition: NurbsCurve2V2,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSurfaceEvaluatorRecordV2 {
    pub id: SurfaceEvaluatorIdV2,
    pub implementation: ExactSurfaceImplementationV2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactSurfaceImplementationV2 {
    Portable {
        definition: ExactSurfaceDefinitionV2,
    },
    Kernel {
        reference: KernelEvaluatorRefV2,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactSurfaceDefinitionV2 {
    Plane {
        origin_m: [f64; 3],
        u_axis_m_per_parameter: [f64; 3],
        v_axis_m_per_parameter: [f64; 3],
        domains: [ParameterRangeV2; 2],
    },
    Cylinder {
        origin_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        axis_m_per_v: [f64; 3],
        radius_m: f64,
        domains: [ParameterRangeV2; 2],
    },
    Cone {
        apex_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        axis: [f64; 3],
        semi_angle_rad: f64,
        domains: [ParameterRangeV2; 2],
    },
    Sphere {
        center_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        z_axis: [f64; 3],
        radius_m: f64,
        domains: [ParameterRangeV2; 2],
    },
    Torus {
        center_m: [f64; 3],
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        z_axis: [f64; 3],
        major_radius_m: f64,
        minor_radius_m: f64,
        domains: [ParameterRangeV2; 2],
    },
    Nurbs {
        definition: NurbsSurface3V2,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NurbsCurve3V2 {
    pub degree: u8,
    pub knots: Vec<f64>,
    pub control_points_m: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub domain: ParameterRangeV2,
    pub periodic: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NurbsCurve2V2 {
    pub degree: u8,
    pub knots: Vec<f64>,
    pub control_points_uv: Vec<[f64; 2]>,
    pub weights: Vec<f64>,
    pub domain: ParameterRangeV2,
    pub periodic: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NurbsSurface3V2 {
    pub u_degree: u8,
    pub v_degree: u8,
    pub u_knots: Vec<f64>,
    pub v_knots: Vec<f64>,
    pub u_control_count: u32,
    pub v_control_count: u32,
    pub control_points_m: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub domains: [ParameterRangeV2; 2],
    pub periodic_u: bool,
    pub periodic_v: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelEvaluatorRefV2 {
    pub entity_token: String,
    pub representation_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactTrimClassifierRecordV2 {
    pub id: TrimClassifierIdV2,
    pub implementation: ExactTrimClassifierImplementationV2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactTrimClassifierImplementationV2 {
    OrientedPcurveWinding,
    Kernel { reference: KernelEvaluatorRefV2 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactMassPropertiesRecordV2 {
    pub id: MassPropertiesEvaluatorIdV2,
    pub implementation: ExactMassPropertiesImplementationV2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactMassPropertiesImplementationV2 {
    Kernel {
        reference: KernelEvaluatorRefV2,
    },
    KernelValidated {
        properties: BodyMassPropertiesV2,
        validation_digest: [u8; 32],
    },
}
