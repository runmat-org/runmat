//! The solver section of a checked `cross_section` geometry and its `triangles` mesh controls.
use runmat_geometry_core::{
    CrossSectionDefinition, CrossSectionLayerRole, CrossSectionTraceBase, CrossSectionTraceRole,
};
use runmat_meshing_core::CrossSectionMeshSettings;

use super::{
    Conductor, ConductorRole, CopperRole, CrossSection, CrossSectionMeshControls, Enclosure,
    StackLayer, StackLayerKind, TraceBase, DEFAULT_GROWTH,
};

impl CrossSection {
    /// The section a checked study solves. Omitted mesh controls take the solver defaults.
    pub fn from_authored(
        definition: &CrossSectionDefinition,
        mesh: Option<&CrossSectionMeshSettings>,
    ) -> Self {
        let mesh = mesh.copied().unwrap_or_default();
        Self {
            layers: definition
                .layers
                .iter()
                .map(|layer| StackLayer {
                    name: layer.name.clone(),
                    thickness_m: layer.thickness_m,
                    kind: match layer.role {
                        CrossSectionLayerRole::Dielectric { permittivity } => {
                            StackLayerKind::Dielectric {
                                relative_permittivity: permittivity,
                            }
                        }
                        CrossSectionLayerRole::SolderMask { permittivity } => {
                            StackLayerKind::SolderMask {
                                relative_permittivity: permittivity,
                            }
                        }
                        CrossSectionLayerRole::Signal { fill_permittivity } => {
                            StackLayerKind::Copper {
                                role: CopperRole::Signal,
                                fill_relative_permittivity: fill_permittivity,
                            }
                        }
                        CrossSectionLayerRole::Plane => StackLayerKind::Copper {
                            role: CopperRole::ReferencePlane,
                            fill_relative_permittivity: None,
                        },
                    },
                })
                .collect(),
            conductors: definition
                .traces
                .iter()
                .map(|trace| Conductor {
                    name: trace.name.clone(),
                    layer: trace.layer.clone(),
                    role: match trace.role {
                        CrossSectionTraceRole::Signal => ConductorRole::Signal,
                        CrossSectionTraceRole::Ground => ConductorRole::Ground,
                    },
                    center_m: trace.center_m,
                    width_m: trace.width_m,
                    etch_angle_rad: trace.etch_angle_rad,
                    base: trace.base.map(|base| match base {
                        CrossSectionTraceBase::Top => TraceBase::Top,
                        CrossSectionTraceBase::Bottom => TraceBase::Bottom,
                    }),
                })
                .collect(),
            mesh: CrossSectionMeshControls {
                edge_size_m: mesh.edge_size_m,
                growth: mesh.growth.unwrap_or(DEFAULT_GROWTH),
                max_size_m: mesh.max_size_m,
            },
            enclosure: definition.enclosure.map(|enclosure| Enclosure {
                half_width_m: enclosure.half_width_m,
                air_m: enclosure.air_m,
            }),
        }
    }
}
