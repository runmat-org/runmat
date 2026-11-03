//! Shared bounded canonical fluid source compilation. No procedure is rewritten.
use super::{invalid, AnalysisExecutionResult, FluidSteadyWorkload, LIMITS};
use crate::{
    physics::cfd::canonical::{
        CanonicalFluidInput, CanonicalFluidPolicy, CanonicalFluidProblem, CanonicalFluidReservation,
    },
    properties::{CompiledPropertyMapIdentity, PreparedProperties},
    topology::{AnalysisProjection, FiniteVolumeProjectionPolicy, FiniteVolumeTopology},
};
use runmat_analysis_core::{
    validate_model, AnalysisBoundary, AnalysisConstraint, AnalysisModel, BoundaryOperator,
    CfdDomain, ExcitationAction, FluidConstraint,
};
use runmat_canonical_codec::serialized_materialization_resident_bytes_estimate;
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::Serialize;
pub(crate) struct PreparedFluidSource {
    /// Whole-mesh projection: every fluid and, in a coupled study, solid element and node.
    pub source: AnalysisProjection,
    /// The fluid cell complex: every element of a standalone flow study, the fluid regions of
    /// a coupled one.
    pub topology: FiniteVolumeTopology,
    pub material: CfdDomain,
    pub properties: crate::properties::CompiledPropertyMap,
    pub property_identity: CompiledPropertyMapIdentity,
    pub property_audit_digest: Digest,
    /// Flow authority only; thermal and structural authority belongs to the coupled physics.
    pub constraints: Vec<FluidConstraint>,
    pub boundaries: Vec<AnalysisBoundary>,
    pub excitations: Vec<ExcitationAction>,
    pub preparation_resident_bytes: u64,
    /// Canonical plans derived from this source; every recovery, accuracy and publication
    /// consumer of the source shares them instead of re-deriving stencils and patterns.
    plans: crate::physics::cfd::canonical::CanonicalFluidPlanCache,
}

/// Prepared sources retained per process. Preparation is a pure function of the authoring
/// workload, the policies, the ceiling and the mesh content; every recovery, accuracy, state,
/// publication and replay consumer of one study asks for the same source. A hit re-encodes the
/// key from the caller's own values and recomputes the mesh's content digest, so it still binds
/// the exact authority and mesh; the shared value is the one a fresh preparation produced from
/// that same content after full validation. Failed preparations are never retained.
const SOURCE_MEMO_CAPACITY: usize = 2;
static SOURCE_MEMO: std::sync::Mutex<Vec<(Digest, std::sync::Arc<PreparedFluidSource>)>> =
    std::sync::Mutex::new(Vec::new());
const SOURCE_KEY_PREFIX: &[u8] = b"runmat-fluid-prepared-source-key/v1\0";

#[derive(Serialize)]
struct SourceKey<'a, T: Serialize> {
    authority: &'a T,
    model: &'a AnalysisModel,
    geometry_policy: FiniteVolumeProjectionPolicy,
    policy: CanonicalFluidPolicy,
    ceiling: u64,
    mesh_content_digest: runmat_meshing_core::StableDigest,
}

impl PreparedFluidSource {
    /// The process's one prepared source for this authority and mesh content.
    pub(super) fn prepare<T: Serialize>(
        authority: &T,
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        geometry_policy: FiniteVolumeProjectionPolicy,
        policy: CanonicalFluidPolicy,
        ceiling: u64,
    ) -> AnalysisExecutionResult<std::sync::Arc<Self>> {
        let mesh_content_digest = mesh
            .content_digest_with_cancellation(&crate::topology::ProjectionCancellation)
            .map_err(invalid)?;
        let key = Digest::sha256(
            runmat_canonical_codec::encode_contract(
                SOURCE_KEY_PREFIX,
                "fluid-prepared-source-key",
                &SourceKey {
                    authority,
                    model,
                    geometry_policy,
                    policy,
                    ceiling,
                    mesh_content_digest,
                },
                LIMITS,
            )
            .map_err(invalid)?,
        );
        if let Some(hit) = SOURCE_MEMO
            .lock()
            .ok()
            .and_then(|memo| memo.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone()))
        {
            return Ok(hit);
        }
        let prepared = std::sync::Arc::new(Self::prepare_uncached(
            authority,
            model,
            mesh,
            geometry_policy,
            policy,
            ceiling,
        )?);
        // The admitted mesh's sealed digest is the one this key was built from.
        if prepared.source.mesh_digest != mesh_content_digest {
            return Err(invalid(
                "prepared fluid source differs from its mesh content",
            ));
        }
        if let Ok(mut memo) = SOURCE_MEMO.lock() {
            memo.retain(|(k, _)| *k != key);
            if memo.len() >= SOURCE_MEMO_CAPACITY {
                memo.remove(0);
            }
            memo.push((key, prepared.clone()));
        }
        Ok(prepared)
    }

    fn prepare_uncached<T: Serialize>(
        authority: &T,
        model: &AnalysisModel,
        mesh: &SolverMeshArtifact,
        geometry_policy: FiniteVolumeProjectionPolicy,
        policy: CanonicalFluidPolicy,
        ceiling: u64,
    ) -> AnalysisExecutionResult<Self> {
        // Count/profile only over borrowed inputs before validation/projection/compiler expansion.
        // These are requested-heap estimates, not allocator/RSS or scheduler accounting.
        if mesh
            .topology
            .volume_elements
            .iter()
            .any(|e| e.order != runmat_meshing_core::ElementOrder::Tet4)
        {
            return Err(invalid(
                "portable canonical fluid currently admits Tet4 only",
            ));
        }
        let geometry =
            FiniteVolumeTopology::reserve_mesh(mesh, geometry_policy).map_err(invalid)?;
        let numerical =
            CanonicalFluidReservation::for_counts(&geometry, geometry.face_incidence_count, policy)
                .map_err(invalid)?;
        let projection =
            AnalysisProjection::derivation_resident_bytes_estimate(mesh).map_err(invalid)?;
        // Two whole authored metadata profiles conservatively cover cloned constraints/material,
        // shared identity serialization and validation scratch coexisting with derived records.
        let metadata = serialized_materialization_resident_bytes_estimate(authority, LIMITS)
            .map_err(invalid)?;
        let base = add_bytes(
            add_bytes(projection, geometry.scratch_bytes as u64)?,
            add_bytes(metadata, metadata)?,
        )?;
        let preliminary = add_bytes(base, numerical.resident_bytes_upper_bound)?;
        require_bytes(preliminary, ceiling).map_err(|error| {
            invalid(format!(
                "{error} (projection {projection}, geometry scratch {}, metadata 2x{metadata}, \
                 solver {} for {} cells)",
                geometry.scratch_bytes, numerical.resident_bytes_upper_bound, geometry.cell_count
            ))
        })?;
        let prepared = PreparedProperties::prepare(model, Some(mesh), ceiling - preliminary)
            .map_err(invalid)?;
        // Material source authentication independently recompiles: both maps can coexist.
        let preparation_resident_bytes = add_bytes(
            base,
            add_bytes(
                prepared.estimated_resident_bytes,
                prepared.estimated_resident_bytes,
            )?,
        )?;
        require_bytes(
            add_bytes(
                preparation_resident_bytes,
                numerical.resident_bytes_upper_bound,
            )?,
            ceiling,
        )
        .map_err(|error| {
            invalid(format!(
                "{error} (projection {projection}, geometry scratch {}, metadata 2x{metadata}, \
                 properties 2x{}, solver {} for {} cells)",
                geometry.scratch_bytes,
                prepared.estimated_resident_bytes,
                numerical.resident_bytes_upper_bound,
                geometry.cell_count
            ))
        })?;
        validate_model(model).map_err(invalid)?;
        let source = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let properties = prepared.materialize().map_err(invalid)?;
        let cells = crate::study::derive_fluid_cells(model, mesh, &properties).map_err(invalid)?;
        let coupled = cells.selection != crate::topology::FiniteVolumeCellSelection::AllElements;
        // The count-only gate above charged geometry scratch over the whole mesh, before the
        // cell selection was known. A coupled study derives finite volumes only on its fluid
        // cells: the admitted preparation carries that selection's scratch.
        let selected_scratch = FiniteVolumeTopology::reserve(&source, geometry_policy, &cells.selection)
            .map_err(invalid)?
            .scratch_bytes as u64;
        let preparation_resident_bytes = preparation_resident_bytes
            .checked_sub(geometry.scratch_bytes as u64)
            .and_then(|bytes| bytes.checked_add(selected_scratch))
            .ok_or_else(|| invalid("fluid preparation residency"))?;
        let topology =
            FiniteVolumeTopology::derive_selected(&source, geometry_policy, cells.selection)
                .map_err(invalid)?;
        let mut constraints = Vec::new();
        for c in &model.constraints {
            match c {
                AnalysisConstraint::Fluid(c) => constraints.push(c.clone()),
                // Coupled physics own their thermal and structural authority.
                _ if coupled => {}
                _ => return Err(invalid("nonfluid constraint")),
            }
        }
        // A declared fluid–structure interface is the fluid's wall: no slip on the structure.
        for interface in &model.interfaces {
            if let runmat_analysis_core::AnalysisInterfaceKind::FluidStructure(_) = interface.kind {
                let mut entities = interface
                    .primary
                    .resolved_entities()
                    .iter()
                    .chain(interface.secondary.resolved_entities())
                    .cloned()
                    .collect::<Vec<_>>();
                entities.sort();
                entities.dedup();
                let mut target =
                    runmat_analysis_core::RegionTarget::boundary(interface.interface_id.clone());
                target.bind_resolved_entities(entities).map_err(invalid)?;
                constraints.push(FluidConstraint {
                    constraint_id: interface.interface_id.clone(),
                    kind: runmat_analysis_core::FluidConstraintKind::NoSlipWall { target },
                });
            }
        }
        let boundaries = model
            .boundaries
            .iter()
            .filter(|b| {
                !coupled || matches!(b.operator, BoundaryOperator::FluidOutletPressure { .. })
            })
            .cloned()
            .collect();
        let excitations = model
            .excitations
            .iter()
            .filter(|a| !coupled || crate::physics::cfd::canonical::supports_action(a))
            .cloned()
            .collect();
        Ok(PreparedFluidSource {
            source,
            topology,
            material: cells.domain,
            property_identity: properties.identity.clone(),
            property_audit_digest: properties.audit_digest,
            properties,
            constraints,
            boundaries,
            excitations,
            preparation_resident_bytes,
            plans: Default::default(),
        })
    }
    pub(super) fn input<'a>(
        &'a self,
        model: &'a AnalysisModel,
        policy: CanonicalFluidPolicy,
    ) -> CanonicalFluidInput<'a> {
        CanonicalFluidInput {
            source: &self.source,
            topology: &self.topology,
            material: &self.material,
            constraints: &self.constraints,
            boundary_operators: &self.boundaries,
            signals: &model.signals,
            excitations: &self.excitations,
            policy,
        }
    }
}

impl PreparedFluidSource {
    pub fn problem<'a>(
        &'a self,
        workload: &'a FluidSteadyWorkload,
    ) -> AnalysisExecutionResult<CanonicalFluidProblem<'a>> {
        // A transient study's source is realized at its authored start; the backward-Euler
        // kernel realizes every later interval endpoint from the same shared signals.
        let sample = match workload.model.sequence.steps.as_slice() {
            // Coupled heat transfer advects with steady flow even across a transient interval.
            [step]
                if workload.is_transient()
                    && workload.coupling() != super::workload::FluidCoupling::ConjugateHeat =>
            {
                let runmat_analysis_core::IndependentDomain::TimeSeconds { start, .. } =
                    step.independent_domain
                else {
                    return Err(invalid("transient fluid requires physical time"));
                };
                crate::excitation::ProjectionSample::TimeSeconds {
                    value: start,
                    approach: runmat_analysis_core::SignalSampleApproach::At,
                }
            }
            _ => crate::excitation::ProjectionSample::Static,
        };
        let problem = CanonicalFluidProblem::prepare_shared(
            self.input(&workload.model, workload.policy),
            sample,
            &self.plans,
        )
        .map_err(invalid)?;
        require_bytes(
            add_bytes(
                self.preparation_resident_bytes,
                problem.reservation().resident_bytes_upper_bound,
            )?,
            workload.resident_ceiling()?,
        )?;
        Ok(problem)
    }
}

pub(super) fn add_bytes(a: u64, b: u64) -> AnalysisExecutionResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("canonical fluid preparation byte estimate overflow"))
}
pub(super) fn require_bytes(required: u64, maximum: u64) -> AnalysisExecutionResult<()> {
    if required > maximum || required > isize::MAX as u64 {
        return Err(invalid(format!(
            "canonical fluid preparation requested heap {required} exceeds ceiling {maximum}"
        )));
    }
    Ok(())
}

impl PreparedFluidSource {
    /// The conjugate energy equation over every element, advected by the retained flow.
    /// Its linear controls are the study's single authored algebraic policy.
    pub fn conjugate_heat<'a>(
        &'a self,
        workload: &'a FluidSteadyWorkload,
        face_volume_flux_m3_per_s: &'a [f64],
    ) -> AnalysisExecutionResult<crate::physics::conjugate_heat::ConjugateHeatProblem<'a>> {
        self.conjugate_heat_sampled(
            workload,
            face_volume_flux_m3_per_s,
            crate::excitation::ExcitationSamplePoint::Static,
        )
    }

    /// The conjugate energy equation with sources and temperatures realized at `time_s`.
    pub fn conjugate_heat_at<'a>(
        &'a self,
        workload: &'a FluidSteadyWorkload,
        face_volume_flux_m3_per_s: &'a [f64],
        time_s: f64,
    ) -> AnalysisExecutionResult<crate::physics::conjugate_heat::ConjugateHeatProblem<'a>> {
        self.conjugate_heat_sampled(
            workload,
            face_volume_flux_m3_per_s,
            crate::excitation::ExcitationSamplePoint::TimeSeconds {
                value: time_s,
                approach: runmat_analysis_core::SignalSampleApproach::At,
            },
        )
    }

    fn conjugate_heat_sampled<'a>(
        &'a self,
        workload: &'a FluidSteadyWorkload,
        face_volume_flux_m3_per_s: &'a [f64],
        sample: crate::excitation::ExcitationSamplePoint<'a>,
    ) -> AnalysisExecutionResult<crate::physics::conjugate_heat::ConjugateHeatProblem<'a>> {
        let [step] = workload.model.sequence.steps.as_slice() else {
            return Err(invalid("conjugate heat requires one authored step"));
        };
        crate::physics::conjugate_heat::ConjugateHeatProblem::prepare(
            crate::physics::conjugate_heat::ConjugateHeatInput {
                model: &workload.model,
                step,
                projection: &self.source,
                properties: &self.properties,
                fluid_topology: &self.topology,
                face_volume_flux_m3_per_s,
                sample,
                policy: crate::physics::conjugate_heat::ConjugateHeatPolicy {
                    linear_relative_tolerance: workload.policy.linear_relative_tolerance,
                },
            },
        )
        .map_err(invalid)
    }
}

impl PreparedFluidSource {
    /// The steady fluid–structure problem on the reference configuration of `mesh`.
    pub fn fluid_structure<'a>(
        &'a self,
        workload: &'a FluidSteadyWorkload,
        mesh: &'a SolverMeshArtifact,
    ) -> AnalysisExecutionResult<crate::physics::fluid_structure::FluidStructureProblem<'a>> {
        crate::physics::fluid_structure::FluidStructureProblem::prepare(
            crate::physics::fluid_structure::FluidStructureInput {
                model: &workload.model,
                mesh,
                projection: &self.source,
                properties: &self.properties,
                fluid_selection: &self.topology.selection,
                geometry_policy: workload.geometry_policy,
                fluid: crate::physics::fluid_structure::FluidAuthority {
                    material: &self.material,
                    constraints: &self.constraints,
                    boundary_operators: &self.boundaries,
                    signals: &workload.model.signals,
                    excitations: &self.excitations,
                    policy: workload.policy,
                },
                policy: crate::physics::fluid_structure::FluidStructurePolicy {
                    coupling_relative_tolerance: workload
                        .policy
                        .physical_relative_tolerance
                        .max(1e-7),
                    ..Default::default()
                },
            },
        )
        .map_err(invalid)
    }
}
