//! Native field publication copies the original physical U/p/q; no solve or interpolation.
//! Fields live on the canonical solver mesh (the neutral volume-mesh topology): fluid cells
//! are element values in projection order, face fluxes are owner-oriented mesh faces, and a
//! fluid-only subset of a coupled mesh is stated as region support, not a second topology.
use super::*;
use crate::{
    execution::{
        FieldRecoveryCost, FieldRecoveryPlan, FieldRecoveryRequest, FieldRecoveryRequirement,
        FieldRecoverySource, FieldTopologyIdentity,
    },
    results::field::{
        FiniteVolumeFieldTopology, RecoveredField, RecoveredFieldArtifact,
        RecoveredFieldDefinition, RecoveredFieldEntityCounts, RecoveredFieldEntityDomain,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample, RecoveredFieldSupport,
    },
    topology::{
        AnalysisProjection, FiniteVolumeCellSelection, FiniteVolumeProjectionPolicy,
        FiniteVolumeTopology,
    },
};
use runmat_analysis_core::{AnalysisField, FieldLocation, ObservationQuantity, RecoveryPolicy};
use runmat_meshing_core::SolverMeshArtifact;
mod profile;
#[cfg(test)]
mod tests;

pub fn fluid_native_field_definition_inventory(
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    [
        (
            "cfd.velocity",
            ObservationQuantity::FluidVelocity,
            FieldLocation::Cell,
            RecoveredFieldEntityDomain::VolumeElements,
        ),
        (
            "cfd.pressure",
            ObservationQuantity::FluidPressure,
            FieldLocation::Cell,
            RecoveredFieldEntityDomain::VolumeElements,
        ),
        (
            "cfd.volume_flux",
            ObservationQuantity::VolumeFlowRate,
            FieldLocation::Face,
            RecoveredFieldEntityDomain::OrientedFaces,
        ),
        (
            "cfd.vorticity",
            ObservationQuantity::Vorticity,
            FieldLocation::Cell,
            RecoveredFieldEntityDomain::VolumeElements,
        ),
        // The tangential viscous traction the owner-side fluid exerts across each face, Pa;
        // on a wall face it is the wall shear stress.
        (
            "cfd.wall_shear_stress",
            ObservationQuantity::WallShearStress,
            FieldLocation::Face,
            RecoveredFieldEntityDomain::OrientedFaces,
        ),
        // Static pressure on each face, Pa: the face value the momentum balance uses
        // (gradient-reconstructed inside, the authored value on a pressure outlet).
        (
            "cfd.face_pressure",
            ObservationQuantity::FluidPressure,
            FieldLocation::Face,
            RecoveredFieldEntityDomain::OrientedFaces,
        ),
    ]
    .into_iter()
    .map(|(id, quantity, location, domain)| {
        RecoveredFieldDefinition::new(
            id,
            quantity,
            location,
            domain,
            quantity.canonical_value_shape(),
            RecoveredFieldNumericRepresentation::Real,
        )
        .map_err(invalid)
    })
    .collect()
}

/// Native fields of one workload: the flow fields, plus the conjugate node temperature when
/// the study couples heat transfer. Temperature is the solved nodal unknown, not a transfer.
pub fn field_definition_inventory(
    coupling: super::workload::FluidCoupling,
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    let mut definitions = fluid_native_field_definition_inventory()?;
    if coupling == super::workload::FluidCoupling::FluidStructure {
        definitions.push(
            RecoveredFieldDefinition::new(
                "fsi.displacement",
                ObservationQuantity::Displacement,
                FieldLocation::Node,
                RecoveredFieldEntityDomain::Nodes,
                ObservationQuantity::Displacement.canonical_value_shape(),
                RecoveredFieldNumericRepresentation::Real,
            )
            .map_err(invalid)?,
        );
    }
    if coupling == super::workload::FluidCoupling::ConjugateHeat {
        definitions.push(
            RecoveredFieldDefinition::new(
                "cht.temperature",
                ObservationQuantity::Temperature,
                FieldLocation::Node,
                RecoveredFieldEntityDomain::Nodes,
                ObservationQuantity::Temperature.canonical_value_shape(),
                RecoveredFieldNumericRepresentation::Real,
            )
            .map_err(invalid)?,
        );
    }
    Ok(definitions)
}

/// Placement of the fluid's native values on the canonical solver mesh: complete-mesh entity
/// counts, the fluid region support, and the stored order of fluid cells and faces.
pub(super) struct VolumeMeshLayout {
    counts: RecoveredFieldEntityCounts,
    support: RecoveredFieldSupport,
    faces_support: RecoveredFieldSupport,
    /// Stored element position -> fluid cell (projection element order).
    cells: Vec<usize>,
    /// Stored face position -> (fluid face, orientation relative to the mesh face owner).
    faces: Vec<(usize, bool)>,
    /// Per fluid face: unit owner normal and area.
    face_frames: Vec<([f64; 3], f64)>,
}

impl VolumeMeshLayout {
    /// Resident bytes of the layout and, for a fluid subset, the complete face derivation.
    pub(super) fn requested_bytes(
        projection: &AnalysisProjection,
        fluid: &FiniteVolumeTopology,
    ) -> AnalysisExecutionResult<u64> {
        let mut bytes = std::mem::size_of::<usize>()
            .checked_mul(fluid.cells.len())
            .and_then(|b| {
                (std::mem::size_of::<(usize, bool)>() + std::mem::size_of::<([f64; 3], f64)>())
                    .checked_mul(fluid.faces.len())
                    .and_then(|f| b.checked_add(f))
            })
            .ok_or_else(|| invalid("fluid layout size overflow"))?;
        if fluid.cells.len() != projection.elements.len() {
            let reservation = FiniteVolumeTopology::reserve(
                projection,
                FiniteVolumeProjectionPolicy::covering(projection),
                &FiniteVolumeCellSelection::AllElements,
            )
            .map_err(invalid)?;
            bytes = bytes
                .checked_add(reservation.scratch_bytes)
                .ok_or_else(|| invalid("fluid layout size overflow"))?;
        }
        u64::try_from(bytes).map_err(invalid)
    }

    /// Upper bound of the complete mesh's owner-oriented face count, for profiles taken
    /// before the layout is derived.
    pub(super) fn face_count_bound(
        projection: &AnalysisProjection,
        fluid: &FiniteVolumeTopology,
    ) -> usize {
        if fluid.cells.len() == projection.elements.len() {
            fluid.faces.len()
        } else {
            projection.elements.len().saturating_mul(4)
        }
    }

    /// Element and face support of the fluid on its mesh: complete for a fluid-only mesh,
    /// its regions in a coupled one.
    fn supports(
        projection: &AnalysisProjection,
        fluid: &FiniteVolumeTopology,
    ) -> AnalysisExecutionResult<Option<[RecoveredFieldSupport; 2]>> {
        if fluid.cells.len() == projection.elements.len() {
            return Ok(None);
        }
        let FiniteVolumeCellSelection::Regions(regions) = &fluid.selection else {
            return Err(invalid("a fluid subset of the mesh is selected by region"));
        };
        Ok(Some([fluid.cells.len(), fluid.faces.len()].map(
            |entities| RecoveredFieldSupport::Regions {
                regions: regions.clone(),
                entities,
            },
        )))
    }

    pub(super) fn derive(
        projection: &AnalysisProjection,
        fluid: &FiniteVolumeTopology,
    ) -> AnalysisExecutionResult<Self> {
        let mut cells = (0..fluid.cells.len()).collect::<Vec<_>>();
        cells.sort_unstable_by_key(|&cell| fluid.cells[cell].source_element_index);
        let Some([support, faces_support]) = Self::supports(projection, fluid)? else {
            // The fluid is the whole mesh: its cell complex is the mesh's, face for face.
            return Ok(Self {
                counts: counts(projection, fluid.faces.len()),
                support: RecoveredFieldSupport::Complete,
                faces_support: RecoveredFieldSupport::Complete,
                cells,
                faces: (0..fluid.faces.len()).map(|face| (face, false)).collect(),
                face_frames: face_frames(fluid),
            });
        };
        let complete = FiniteVolumeTopology::derive_selected(
            projection,
            FiniteVolumeProjectionPolicy::covering(projection),
            FiniteVolumeCellSelection::AllElements,
        )
        .map_err(invalid)?;
        let inside = |cell: usize| {
            fluid
                .selection
                .includes(&projection.elements[complete.cells[cell].source_element_index].region_id)
        };
        let mut faces = Vec::with_capacity(fluid.faces.len());
        for face in &complete.faces {
            if !inside(face.owner) && !face.neighbor.is_some_and(inside) {
                continue;
            }
            let index = fluid
                .faces
                .binary_search_by(|f| f.corner_identities.cmp(&face.corner_identities))
                .map_err(|_| invalid("fluid face absent from the mesh face complex"))?;
            let owner = complete.cells[face.owner].source_element_index;
            faces.push((
                index,
                fluid.cells[fluid.faces[index].owner].source_element_index != owner,
            ));
        }
        if faces.len() != fluid.faces.len() {
            return Err(invalid(
                "fluid faces differ from the mesh faces of fluid cells",
            ));
        }
        Ok(Self {
            counts: counts(projection, complete.faces.len()),
            support,
            faces_support,
            cells,
            faces,
            face_frames: face_frames(fluid),
        })
    }

    /// Native definitions of a workload on its fluid source, before the layout is derived.
    pub(super) fn source_definitions(
        projection: &AnalysisProjection,
        fluid: &FiniteVolumeTopology,
        coupling: super::workload::FluidCoupling,
    ) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
        let [support, faces_support] = Self::supports(projection, fluid)?.unwrap_or([
            RecoveredFieldSupport::Complete,
            RecoveredFieldSupport::Complete,
        ]);
        with_supports(
            field_definition_inventory(coupling)?,
            &support,
            &faces_support,
        )
    }

    /// Native definitions of one workload on this layout.
    pub(super) fn definitions(
        &self,
        coupling: super::workload::FluidCoupling,
    ) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
        with_supports(
            field_definition_inventory(coupling)?,
            &self.support,
            &self.faces_support,
        )
    }

    #[cfg(test)]
    fn face_values(&self, values: &[f64]) -> Vec<f64> {
        self.faces
            .iter()
            .map(|&(face, reversed)| {
                if reversed {
                    -values[face]
                } else {
                    values[face]
                }
            })
            .collect()
    }
}

fn face_frames(fluid: &FiniteVolumeTopology) -> Vec<([f64; 3], f64)> {
    fluid
        .faces
        .iter()
        .map(|face| {
            (
                face.owner_area_vector_m2
                    .map(|x| x / face.area_m2.canonical_value()),
                face.area_m2.canonical_value(),
            )
        })
        .collect()
}

fn with_supports(
    definitions: Vec<RecoveredFieldDefinition>,
    support: &RecoveredFieldSupport,
    faces_support: &RecoveredFieldSupport,
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    definitions
        .into_iter()
        .map(|definition| {
            let support = match definition.entity_domain() {
                RecoveredFieldEntityDomain::VolumeElements => support.clone(),
                RecoveredFieldEntityDomain::OrientedFaces => faces_support.clone(),
                _ => RecoveredFieldSupport::Complete,
            };
            definition.with_support(support).map_err(invalid)
        })
        .collect()
}

fn counts(projection: &AnalysisProjection, oriented_faces: usize) -> RecoveredFieldEntityCounts {
    RecoveredFieldEntityCounts {
        nodes: projection.nodes.len(),
        volume_elements: projection.elements.len(),
        oriented_faces,
        volume_integration_points: 0,
        oriented_edges: 0,
        beam_elements: 0,
        shell_elements: 0,
        connectors: 0,
        constraint_groups: 0,
    }
}

fn plan(
    layout: &VolumeMeshLayout,
    coupling: super::workload::FluidCoupling,
) -> AnalysisExecutionResult<FieldRecoveryPlan> {
    let requirements = layout
        .definitions(coupling)?
        .into_iter()
        .map(|d| {
            let n = u64::try_from(d.stored_entities(layout.counts)).map_err(invalid)?;
            let components = d
                .value_shape()
                .component_count()
                .ok_or_else(|| invalid("native fluid shape"))? as u64;
            Ok(FieldRecoveryRequirement {
                support: d.support().clone(),
                quantity: d.quantity(),
                source_location: d.location(),
                output_location: d.location(),
                recovery: RecoveryPolicy::Native,
                value_shape: d.value_shape().clone(),
                entity_domain: d.entity_domain(),
                cost: FieldRecoveryCost {
                    source_entity_count: n,
                    output_entity_count: n,
                    topology_incidence_count: 0,
                    arithmetic_operation_count: n
                        .checked_mul(components)
                        .ok_or_else(|| invalid("fluid field count overflow"))?,
                },
            })
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    let result = FieldRecoveryPlan { requirements };
    result.validate()?;
    Ok(result)
}

/// Native definitions named by a fluid recovery request: its requirements are the authority
/// for quantities and support; ids follow the native inventory.
pub fn fluid_native_field_definitions(
    request: &FieldRecoveryRequest,
) -> AnalysisExecutionResult<Vec<RecoveredFieldDefinition>> {
    request.validate()?;
    if !matches!(request.topology, FieldTopologyIdentity::VolumeMesh { .. }) {
        return Err(invalid(
            "fluid native fields live on the canonical volume mesh",
        ));
    }
    let coupling = request
        .plan
        .requirements
        .iter()
        .find_map(|r| match r.quantity {
            ObservationQuantity::Temperature => Some(super::workload::FluidCoupling::ConjugateHeat),
            ObservationQuantity::Displacement => {
                Some(super::workload::FluidCoupling::FluidStructure)
            }
            _ => None,
        })
        .unwrap_or(super::workload::FluidCoupling::Standalone);
    let inventory = field_definition_inventory(coupling)?;
    if inventory.len() != request.plan.requirements.len() {
        return Err(invalid(
            "fluid recovery plan differs from the native inventory",
        ));
    }
    let definitions = inventory
        .into_iter()
        .zip(&request.plan.requirements)
        .map(|(definition, requirement)| {
            definition
                .with_support(requirement.support.clone())
                .map_err(invalid)
        })
        .collect::<AnalysisExecutionResult<Vec<_>>>()?;
    request
        .plan
        .validate_definitions(definitions.iter().cloned())?;
    Ok(definitions)
}

impl FluidSteadySolverResult {
    pub fn native_field_topology(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FiniteVolumeFieldTopology> {
        self.on_source(workload, mesh)?.native_field_topology()
    }

    pub fn native_field_request(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        self.on_source(workload, mesh)?.native_field_request()
    }

    fn request_on(
        &self,
        layout: &VolumeMeshLayout,
        projection: &AnalysisProjection,
        workload: &FluidSteadyWorkload,
    ) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        FieldRecoveryRequest::new(
            self.producer_identity(),
            FieldRecoverySource::SolverResult(self.identity()?),
            FieldTopologyIdentity::VolumeMesh {
                mesh_digest: projection.mesh_digest,
            },
            plan(layout, workload.coupling())?,
        )
    }

    /// Native values in stored order: cells by projection element, faces by mesh face with
    /// the orientation of the mesh face owner, coupled node fields as solved. One lazy source
    /// serves recovery and the in-place publication check.
    fn native_value_sources<'s>(
        &'s self,
        layout: &'s VolumeMeshLayout,
        original: &'s crate::physics::cfd::canonical::CanonicalFluidResult,
    ) -> Vec<Box<dyn Iterator<Item = f64> + 's>> {
        let viscous = &original.residual.face_forces.viscous_traction_n;
        let pressure = &original.residual.face_forces.pressure_traction_n;
        let vorticity = original.residual.cell_vorticity_per_s.canonical_values();
        let mut sources: Vec<Box<dyn Iterator<Item = f64> + 's>> = vec![
            Box::new(
                layout
                    .cells
                    .iter()
                    .flat_map(|&cell| original.velocity_m_per_s.canonical_values()[cell]),
            ),
            Box::new(
                layout
                    .cells
                    .iter()
                    .map(|&cell| original.pressure_pa.canonical_values()[cell]),
            ),
            Box::new(layout.faces.iter().map(|&(face, reversed)| {
                let q = original.face_volume_flux_m3_per_s.canonical_values()[face];
                if reversed {
                    -q
                } else {
                    q
                }
            })),
            Box::new(layout.cells.iter().flat_map(|&cell| vorticity[cell])),
            Box::new(layout.faces.iter().flat_map(move |&(face, reversed)| {
                // Shear exerted by the owner fluid across the face: the negated tangential part
                // of the viscous force on the owner, per unit area.
                let (normal, area) = layout.face_frames[face];
                let t = viscous.canonical_values()[face];
                let tn = t[0] * normal[0] + t[1] * normal[1] + t[2] * normal[2];
                let sign = if reversed { 1.0 } else { -1.0 };
                std::array::from_fn::<f64, 3, _>(|k| sign * (t[k] - tn * normal[k]) / area)
            })),
            Box::new(layout.faces.iter().map(move |&(face, _)| {
                // The pressure force on the owner fluid is −p_f S: p_f = −(t·n̂)/A, the same
                // for either orientation of the face.
                let (normal, area) = layout.face_frames[face];
                let t = pressure.canonical_values()[face];
                -(t[0] * normal[0] + t[1] * normal[1] + t[2] * normal[2]) / area
            })),
        ];
        if let Some(energy) = self.conjugate_heat() {
            sources.push(Box::new(energy.temperature_k.iter().copied()));
        }
        if let Some(displacement) = self.coupled_displacement() {
            sources.push(Box::new(displacement.iter().flatten().copied()));
        }
        sources
    }

    /// Retained field samples in stored order. A steady flow has its final state. A transient
    /// has, in increasing physical time, each retained interval whose end an authored history
    /// samples, then its terminal state at the end time: sampled producers keep physical
    /// coordinates, and final outputs select the terminal sample. Coupled node fields
    /// (temperature, displacement) belong to steady or terminal coupled states only.
    pub(crate) fn field_samples(
        &self,
        workload: &FluidSteadyWorkload,
    ) -> AnalysisExecutionResult<
        Vec<(
            RecoveredFieldSample,
            &crate::physics::cfd::canonical::CanonicalFluidResult,
        )>,
    > {
        let Some(history) = self.transient_history() else {
            return Ok(vec![(RecoveredFieldSample::Final, self.original())]);
        };
        let step = workload
            .model
            .sequence
            .steps
            .first()
            .ok_or_else(|| invalid("fluid study has no step"))?;
        let schedule = super::FluidTransientSchedule::derive(&workload.model, step)?;
        let points = schedule.points_s();
        let terminal = points.len() - 2;
        let mut samples = Vec::new();
        for (ordinal, result) in history.retained_results() {
            let sample = RecoveredFieldSample::TimeSeconds(points[ordinal + 1]);
            if *ordinal == terminal
                || crate::results::observation::authored_histories_sample(step, sample)
                    .map_err(invalid)?
            {
                samples.push((sample, result));
            }
        }
        let terminal_retained = match (samples.last(), history.terminal()) {
            (Some((_, last)), Some(expected)) => std::ptr::eq(*last, expected),
            _ => false,
        };
        if !terminal_retained {
            return Err(invalid("transient fluid retained no terminal state"));
        }
        Ok(samples)
    }

    pub fn recover_native_fields(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        self.on_source(workload, mesh)?
            .recover_native_fields(request)
    }

    /// Whether published fields carry exactly this result's native values, bit for bit.
    pub fn validate_native_fields(
        &self,
        workload: &FluidSteadyWorkload,
        mesh: &SolverMeshArtifact,
        fields: &RecoveredFieldArtifact,
    ) -> AnalysisExecutionResult<()> {
        self.on_source(workload, mesh)?
            .validate_native_fields(fields)
    }
}

/// Field checks of a result already validated against its prepared source.
impl FluidResultOnSource<'_> {
    /// Source-derived requested heap for native construction; not measured execution usage.
    pub fn native_field_resident_bytes_estimate(&self) -> AnalysisExecutionResult<u64> {
        profile::requested_bytes(&self.result, &self.workload, &self.source)
    }

    pub fn native_field_topology(&self) -> AnalysisExecutionResult<FiniteVolumeFieldTopology> {
        profile::admit(&self.result, &self.workload, &self.source)?;
        // `prepare` derived the topology from this projection in process.
        FiniteVolumeFieldTopology::from_derived(&self.source.source, &self.source.topology)
            .map_err(invalid)
    }

    /// The fluid's native field layout on the canonical solver mesh, admitted first.
    fn volume_mesh_layout(&self) -> AnalysisExecutionResult<VolumeMeshLayout> {
        profile::admit(&self.result, &self.workload, &self.source)?;
        VolumeMeshLayout::derive(&self.source.source, &self.source.topology)
    }

    pub fn native_field_request(&self) -> AnalysisExecutionResult<FieldRecoveryRequest> {
        let layout = self.volume_mesh_layout()?;
        self.result
            .request_on(&layout, &self.source.source, &self.workload)
    }

    pub fn recover_native_fields(
        &self,
        request: &FieldRecoveryRequest,
    ) -> AnalysisExecutionResult<RecoveredFieldArtifact> {
        let (workload, result) = (&*self.workload, &*self.result);
        let layout = self.volume_mesh_layout()?;
        if request != &result.request_on(&layout, &self.source.source, workload)? {
            return Err(invalid(
                "fluid field request differs from original source/native layout",
            ));
        }
        let definitions = layout.definitions(workload.coupling())?;
        let samples = result.field_samples(workload)?;
        let mut fields = Vec::with_capacity(definitions.len() * samples.len());
        for (sample, original) in samples {
            for (definition, values) in definitions
                .iter()
                .zip(result.native_value_sources(&layout, original))
            {
                let values = values.collect::<Vec<_>>();
                let shape = definition
                    .storage_shape(definition.stored_entities(layout.counts))
                    .map_err(invalid)?;
                let storage = AnalysisField::host_f64(
                    definition.field_id(),
                    definition.layout(),
                    shape,
                    values,
                );
                fields.push(
                    RecoveredField::new(definition.clone(), sample, storage).map_err(invalid)?,
                );
            }
        }
        request.plan.validate_definitions(definitions)?;
        RecoveredFieldArtifact::new(
            request.identity(),
            request.source,
            request.topology.discretization_digest(),
            layout.counts,
            fields,
        )
        .map_err(invalid)
    }

    pub fn validate_native_fields(
        &self,
        fields: &RecoveredFieldArtifact,
    ) -> AnalysisExecutionResult<()> {
        let (workload, result) = (&*self.workload, &*self.result);
        let layout = self.volume_mesh_layout()?;
        let definitions = layout.definitions(workload.coupling())?;
        let samples = result.field_samples(workload)?;
        // Compared in place against the original arrays: no second copy of the values.
        let expected = samples
            .iter()
            .flat_map(|(sample, original)| {
                definitions
                    .iter()
                    .zip(result.native_value_sources(&layout, original))
                    .map(move |(definition, values)| (*sample, definition, values))
            })
            .collect::<Vec<_>>();
        if fields.fields().len() != expected.len()
            || expected.len() != definitions.len() * samples.len()
            || fields.entity_counts() != layout.counts
            || fields.fields().iter().zip(expected).any(
                |(field, (sample, definition, mut values))| {
                    field.definition() != definition
                        || field.sample() != sample
                        || field.storage().as_host_f64().is_none_or(|stored| {
                            !stored
                                .iter()
                                .map(|value| Some(value.to_bits()))
                                .chain(std::iter::once(None))
                                .eq(values
                                    .by_ref()
                                    .map(|value| Some(value.to_bits()))
                                    .chain(std::iter::once(None)))
                        })
                },
            )
        {
            return Err(invalid(
                "fluid fields differ from the original native cell U/p, face q or coupled values",
            ));
        }
        Ok(())
    }
}
