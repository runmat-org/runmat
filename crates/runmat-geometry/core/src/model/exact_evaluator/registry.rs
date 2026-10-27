use serde::{Deserialize, Serialize};

use super::{
    BodyMassProperties, CurveEvaluatorId, MassPropertiesEvaluatorId, ParameterRange,
    PcurveEvaluatorId, SurfaceEvaluatorId, TrimClassifierId,
};
use crate::GeometryContractError;

pub const EXACT_EVALUATOR_REGISTRY_SCHEMA_VERSION: u16 = 4;

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
    /// A topologically meaningful parameter interval that collapses to one 3D point.
    ///
    /// Degenerate B-Rep edges use this definition while their face-use pcurves retain
    /// the nonzero parameter traversal needed to describe a singular trim.
    Degenerate {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        point_m: runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
        domain: ParameterRange,
    },
    Line {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        origin_m:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
        direction_m_per_parameter: [f64; 3],
        domain: ParameterRange,
    },
    Circle {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        center_m:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        radius_m: runmat_types::physical::Length,
        domain: ParameterRange,
        periodic: bool,
    },
    Ellipse {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        center_m:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        major_radius_m: runmat_types::physical::Length,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        minor_radius_m: runmat_types::physical::Length,
        domain: ParameterRange,
        periodic: bool,
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
        #[serde(with = "runmat_types::physical::canonical_storage")]
        origin_m:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
        u_axis_m_per_parameter: [f64; 3],
        v_axis_m_per_parameter: [f64; 3],
        domains: [ParameterRange; 2],
    },
    Cylinder {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        origin_m:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        axis_m_per_v: [f64; 3],
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        radius_m: runmat_types::physical::Length,
        domains: [ParameterRange; 2],
    },
    Cone {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        apex_m: runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        axis: [f64; 3],
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        semi_angle_rad: runmat_types::physical::Angle,
        domains: [ParameterRange; 2],
    },
    Sphere {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        center_m:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        z_axis: [f64; 3],
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        radius_m: runmat_types::physical::Length,
        domains: [ParameterRange; 2],
    },
    Torus {
        #[serde(with = "runmat_types::physical::canonical_storage")]
        center_m:
            runmat_types::physical::Quantity<runmat_types::physical::kind::Position, [f64; 3]>,
        x_axis: [f64; 3],
        y_axis: [f64; 3],
        z_axis: [f64; 3],
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        major_radius_m: runmat_types::physical::Length,
        #[serde(with = "runmat_types::physical::canonical_scalar")]
        minor_radius_m: runmat_types::physical::Length,
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
    #[serde(with = "runmat_types::physical::canonical_storage")]
    pub control_points_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, Vec<[f64; 3]>>,
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
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// U-major tensor grid: `control_points_m[u * v_control_count + v]`.
    pub control_points_m:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Position, Vec<[f64; 3]>>,
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

impl ExactEvaluatorRegistry {
    /// Returns the one immutable kernel representation required by kernel-backed records.
    /// A registry cannot span representations because entity tokens are meaningful only
    /// within the representation whose digest namespaces them.
    pub fn kernel_representation_digest(&self) -> Result<Option<[u8; 32]>, GeometryContractError> {
        let mut required = None;
        for digest in
            self.curves
                .iter()
                .filter_map(|record| match &record.implementation {
                    ExactCurveImplementation::Kernel { reference } => {
                        Some(reference.representation_digest)
                    }
                    ExactCurveImplementation::Portable { .. } => None,
                })
                .chain(
                    self.pcurves
                        .iter()
                        .filter_map(|record| match &record.implementation {
                            ExactPcurveImplementation::Kernel { reference } => {
                                Some(reference.representation_digest)
                            }
                            ExactPcurveImplementation::Portable { .. } => None,
                        }),
                )
                .chain(
                    self.surfaces
                        .iter()
                        .filter_map(|record| match &record.implementation {
                            ExactSurfaceImplementation::Kernel { reference } => {
                                Some(reference.representation_digest)
                            }
                            ExactSurfaceImplementation::Portable { .. } => None,
                        }),
                )
                .chain(self.trim_classifiers.iter().filter_map(
                    |record| match &record.implementation {
                        ExactTrimClassifierImplementation::Kernel { reference } => {
                            Some(reference.representation_digest)
                        }
                        ExactTrimClassifierImplementation::OrientedPcurveWinding => None,
                    },
                ))
                .chain(self.mass_properties.iter().filter_map(
                    |record| match &record.implementation {
                        ExactMassPropertiesImplementation::Kernel { reference } => {
                            Some(reference.representation_digest)
                        }
                        ExactMassPropertiesImplementation::KernelValidated { .. } => None,
                    },
                ))
        {
            if required.is_some_and(|previous| previous != digest) {
                return Err(GeometryContractError::invalid(
                    "kernel evaluator representation",
                    "one exact evaluator registry cannot reference multiple kernel representations",
                ));
            }
            required = Some(digest);
        }
        Ok(required)
    }
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
