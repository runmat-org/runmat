use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_execution::{
    CompletedStructuralSequence, StructuralSequencePreparation, StructuralSequenceResultClosure,
};
use runmat_analysis_fea::execution::{AnalysisNumericalObjectKind, StructuralLinearWorkload};
use runmat_canonical_codec::{decode_contract, encode_contract};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    ArtifactAccess, LogicalObject, ObjectNamespace, ProgramExecutionObject,
    ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, MeshingRequest};
use runmat_meshing_execution::ExactMeshingDagRunResult;
use serde::{Deserialize, Serialize};

use super::boundary::BoundaryRegionResolution;
use super::contract::{structural_workload_inventory_digest, DecodedExactStructuralStudyRoot};
use super::{
    protocol, read, CONTRACT_LIMITS, PUBLICATION_DOMAIN, PUBLICATION_MEDIA_TYPE,
    PUBLICATION_PREFIX, PUBLICATION_VALUE_SCHEMA,
};
use crate::NativeExecutionResult;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactStructuralStudyPublication {
    schema_version: u16,
    program_revision: ProgramRevision,
    authored_study_digest: Digest,
    unresolved_workloads_digest: Digest,
    meshing_request_digest: Digest,
    meshing_root: ValueRef,
    meshing_objects: Vec<ValueRef>,
    boundary_root: ValueRef,
    boundary_objects: Vec<ValueRef>,
    signal_roots: Vec<ValueRef>,
    signal_objects: Vec<ValueRef>,
    analysis_closures: Vec<StructuralSequenceResultClosure>,
    analysis_objects: Vec<ValueRef>,
}

impl ExactStructuralStudyPublication {
    fn validate(&self, access: &ArtifactAccess) -> NativeExecutionResult<()> {
        if self.schema_version != 1
            || self.authored_study_digest.bytes() == &[0; 32]
            || self.unresolved_workloads_digest.bytes() == &[0; 32]
            || self.meshing_request_digest.bytes() == &[0; 32]
            || self.meshing_objects.is_empty()
            || self.boundary_objects.is_empty()
            || self.analysis_objects.is_empty()
            || self.signal_roots.len() > self.signal_objects.len()
            || self
                .signal_roots
                .iter()
                .any(|root| !self.signal_objects.contains(root))
        {
            return Err(protocol("exact structural study publication is invalid"));
        }
        for reference in self
            .meshing_objects
            .iter()
            .chain(&self.boundary_objects)
            .chain(&self.signal_objects)
            .chain(&self.analysis_objects)
        {
            if reference.kind != ValueRefKind::ResultObject
                || reference.authorization_scope != access.authorization_scope
                || reference.encryption_context != access.encryption_context
            {
                return Err(protocol(
                    "study publication object is outside its result authority",
                ));
            }
        }
        for (root, inventory) in [
            (&self.meshing_root, &self.meshing_objects),
            (&self.boundary_root, &self.boundary_objects),
        ] {
            if !inventory.contains(root) {
                return Err(protocol(
                    "study publication closure omits a declared component root",
                ));
            }
            reference_inventory(inventory)?;
        }
        let mut nodes = BTreeSet::new();
        for closure in &self.analysis_closures {
            if !nodes.insert(closure.node.clone()) {
                return Err(protocol(
                    "study publication contains a duplicate sequence node",
                ));
            }
            let ProgramExecutionResponse::ExternalizedSuccess {
                outputs,
                result_objects,
            } = &closure.response
            else {
                return Err(protocol(
                    "study publication contains a non-successful sequence closure",
                ));
            };
            let [ValuePayload::Object(root)] = outputs.as_slice() else {
                return Err(protocol("sequence result closure has an invalid root"));
            };
            if !result_objects.contains(root)
                || result_objects
                    .iter()
                    .any(|reference| !self.analysis_objects.contains(reference))
            {
                return Err(protocol(
                    "sequence result closure differs from its analysis inventory",
                ));
            }
            reference_inventory(result_objects)?;
        }
        let closure_objects = self
            .analysis_closures
            .iter()
            .flat_map(|closure| match &closure.response {
                ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } => {
                    result_objects.clone()
                }
                _ => Vec::new(),
            })
            .collect::<Vec<_>>();
        let closure_objects = unique_references(closure_objects)?;
        if reference_inventory(&closure_objects)? != reference_inventory(&self.analysis_objects)? {
            return Err(protocol(
                "analysis inventory contains unreferenced or missing sequence objects",
            ));
        }
        reference_inventory(&self.signal_objects)?;
        reference_inventory(&self.analysis_objects)?;
        Ok(())
    }

    fn canonical_bytes(&self, access: &ArtifactAccess) -> NativeExecutionResult<Vec<u8>> {
        self.validate(access)?;
        encode_contract(
            PUBLICATION_PREFIX,
            PUBLICATION_DOMAIN,
            self,
            CONTRACT_LIMITS,
        )
        .map_err(protocol)
    }

    fn decode(bytes: &[u8], access: &ArtifactAccess) -> NativeExecutionResult<Self> {
        let publication = decode_contract(
            PUBLICATION_PREFIX,
            PUBLICATION_DOMAIN,
            bytes,
            CONTRACT_LIMITS,
        )
        .map_err(protocol)?;
        Self::validate(&publication, access)?;
        Ok(publication)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralStudyResult {
    pub frozen_project_digest: Option<Digest>,
    pub mesh: ExactMeshingDagRunResult,
    pub boundary_resolution: BoundaryRegionResolution,
    pub resolved_workloads:
        BTreeMap<runmat_analysis_core::ProcedureStepId, StructuralLinearWorkload>,
    pub sequence: CompletedStructuralSequence,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactStructuralStudyResultExpectation {
    pub program_revision: ProgramRevision,
    pub artifact_access: ArtifactAccess,
    pub authored_study_digest: Digest,
    pub frozen_project_digest: Option<Digest>,
    pub geometry_source_digest: Digest,
    pub workloads: BTreeMap<runmat_analysis_core::ProcedureStepId, StructuralLinearWorkload>,
    pub signal_materializations: Vec<runmat_analysis_execution::SignalMaterializationExpectation>,
    pub expected_meshing_request: MeshingRequest,
}

impl ExactStructuralStudyResultExpectation {
    pub fn validate(&self) -> NativeExecutionResult<()> {
        self.artifact_access.validate().map_err(protocol)?;
        runmat_analysis_execution::validate_structural_sequence_workloads(&self.workloads)
            .map_err(protocol)?;
        self.expected_meshing_request.validate().map_err(protocol)?;
        let authored_signals = self
            .workloads
            .values()
            .next()
            .ok_or_else(|| protocol("whole-study expectation has no workload"))?
            .model
            .signals
            .iter()
            .filter(|signal| signal.generator().is_some())
            .collect::<Vec<_>>();
        if authored_signals.len() != self.signal_materializations.len()
            || authored_signals
                .iter()
                .zip(&self.signal_materializations)
                .any(|(signal, expected)| *signal != &expected.authored)
        {
            return Err(protocol(
                "whole-study signal expectations differ from its authored workload",
            ));
        }
        for materialization in &self.signal_materializations {
            materialization.validate().map_err(protocol)?;
            if materialization.admitted.program_revision != self.program_revision {
                return Err(protocol(
                    "whole-study signal expectation uses another program revision",
                ));
            }
        }
        if self.authored_study_digest.bytes() == &[0; 32]
            || self.geometry_source_digest.bytes() == &[0; 32]
            || self
                .frozen_project_digest
                .is_some_and(|digest| digest.bytes() == &[0; 32])
        {
            return Err(protocol("whole-study result expectation is invalid"));
        }
        Ok(())
    }
}

pub(crate) fn package_exact_structural_study_result(
    request: &ProgramExecutionRequest,
    decoded: &DecodedExactStructuralStudyRoot,
    meshing: &ExactMeshingDagRunResult,
    boundary: &runmat_analysis_execution::PreparedAnalysisObjects,
    signals: &[crate::remote::signal_materialization::CompletedRemoteSignalMaterialization],
    sequence: &CompletedStructuralSequence,
    unresolved_workloads_digest: Digest,
) -> NativeExecutionResult<(ProgramExecutionResponse, Vec<ProgramExecutionObject>)> {
    let boundary_objects = boundary
        .references(&decoded.artifact_access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    let boundary_root = boundary_objects
        .first()
        .ok_or_else(|| protocol("boundary resolution closure is empty"))?
        .clone();
    let signal_closures = signals
        .iter()
        .map(|signal| {
            signal
                .objects
                .references(&decoded.artifact_access, ValueRefKind::ResultObject)
                .map_err(protocol)
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let signal_roots = signal_closures
        .iter()
        .map(|objects| {
            objects
                .first()
                .cloned()
                .ok_or_else(|| protocol("signal realization closure is empty"))
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let signal_objects = unique_references(signal_closures.into_iter().flatten())?;
    let analysis_closures = sequence
        .result_closures(&decoded.artifact_access)
        .map_err(protocol)?;
    let analysis_objects =
        unique_references(
            analysis_closures
                .iter()
                .flat_map(|closure| match &closure.response {
                    ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } => {
                        result_objects.clone()
                    }
                    _ => Vec::new(),
                }),
        )?;
    let publication = ExactStructuralStudyPublication {
        schema_version: 1,
        program_revision: request.recipe.program_revision.clone(),
        authored_study_digest: decoded.authored_study_digest,
        unresolved_workloads_digest,
        meshing_request_digest: Digest::from_bytes(
            *decoded
                .meshing_request
                .canonical_digest()
                .map_err(protocol)?
                .bytes(),
        ),
        meshing_root: meshing.root.clone(),
        meshing_objects: meshing.result_objects.clone(),
        boundary_root,
        boundary_objects: boundary_objects.clone(),
        signal_roots,
        signal_objects: signal_objects.clone(),
        analysis_closures,
        analysis_objects: analysis_objects.clone(),
    };
    let mut new_objects = boundary
        .objects
        .iter()
        .zip(&boundary_objects)
        .map(|(object, reference)| ProgramExecutionObject {
            reference: reference.clone(),
            bytes: object.bytes.clone(),
        })
        .collect::<Vec<_>>();
    let publication_object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        "analysis/exact-structural-study-publications",
        PUBLICATION_MEDIA_TYPE,
        publication.canonical_bytes(&decoded.artifact_access)?,
    )
    .map_err(protocol)?;
    let publication_root = decoded
        .artifact_access
        .reference(
            &publication_object,
            ValueRefKind::ResultObject,
            PUBLICATION_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let mut inventory = BTreeMap::new();
    let mut logical_digests = BTreeSet::new();
    for reference in meshing
        .result_objects
        .iter()
        .chain(&boundary_objects)
        .chain(&signal_objects)
        .chain(&analysis_objects)
        .chain(std::iter::once(&publication_root))
    {
        if inventory.insert(reference.id, reference.clone()).is_some()
            || !logical_digests.insert(reference.logical_digest)
        {
            return Err(protocol(
                "whole-study result inventory contains duplicate object references",
            ));
        }
    }
    let response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: vec![ValuePayload::Object(Box::new(publication_root.clone()))],
        result_objects: inventory.into_values().collect(),
    };
    response.validate_against(request).map_err(protocol)?;
    new_objects.push(ProgramExecutionObject {
        reference: publication_root,
        bytes: publication_object.bytes,
    });
    for object in &new_objects {
        object.validate().map_err(protocol)?;
    }
    Ok((response, new_objects))
}

pub fn import_exact_structural_study_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    expected: &ExactStructuralStudyResultExpectation,
    limits: ObjectInventoryLimits,
) -> NativeExecutionResult<ExactStructuralStudyResult> {
    expected.validate()?;
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        return Err(protocol(
            "whole-study execution did not return a result closure",
        ));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(protocol("whole-study result root is invalid"));
    };
    expected
        .artifact_access
        .validate_reference(
            root,
            ValueRefKind::ResultObject,
            PUBLICATION_MEDIA_TYPE,
            PUBLICATION_VALUE_SCHEMA,
        )
        .map_err(protocol)?;
    let publication =
        ExactStructuralStudyPublication::decode(&read(source, root)?, &expected.artifact_access)?;
    if publication.program_revision != expected.program_revision {
        return Err(protocol("whole-study program revision changed"));
    }
    if publication.authored_study_digest != expected.authored_study_digest {
        return Err(protocol("whole-study authored study identity changed"));
    }
    let expected_workloads_digest = structural_workload_inventory_digest(&expected.workloads)?;
    if publication.unresolved_workloads_digest != expected_workloads_digest {
        return Err(protocol("whole-study workload identity changed"));
    }
    let expected_meshing_request_digest = Digest::from_bytes(
        *expected
            .expected_meshing_request
            .canonical_digest()
            .map_err(protocol)?
            .bytes(),
    );
    if publication.meshing_request_digest != expected_meshing_request_digest {
        return Err(protocol("whole-study meshing request identity changed"));
    }
    let expected_inventory = publication
        .meshing_objects
        .iter()
        .chain(&publication.boundary_objects)
        .chain(&publication.signal_objects)
        .chain(&publication.analysis_objects)
        .cloned()
        .chain(std::iter::once((**root).clone()))
        .collect::<Vec<_>>();
    if reference_inventory(&expected_inventory)? != reference_inventory(&result_objects)? {
        return Err(protocol(
            "whole-study result inventory differs from its canonical closure",
        ));
    }
    let mesh = runmat_meshing_execution::import_exact_meshing_result(
        source,
        publication.meshing_root,
        publication.meshing_objects,
        expected.artifact_access.clone(),
        limits,
    )
    .map_err(protocol)?;
    let mut expected_resolved_request = expected.expected_meshing_request.clone();
    if !crate::exact_geometry_admission::same_requested_tolerance(
        &expected_resolved_request.tolerance,
        &mesh.artifact.resolved_request.tolerance,
    ) {
        return Err(protocol(
            "whole-study mesh tolerance differs from the authored meshing request",
        ));
    }
    expected_resolved_request.tolerance = mesh.artifact.resolved_request.tolerance;
    if mesh.artifact.resolved_request != expected_resolved_request {
        return Err(protocol(
            "whole-study mesh differs from the authored meshing request",
        ));
    }
    let boundary_manifest_bytes = read(source, &publication.boundary_root)?;
    let boundary_manifest =
        runmat_analysis_fea::execution::AnalysisObjectManifest::from_canonical_bytes(
            &boundary_manifest_bytes,
        )
        .map_err(protocol)?;
    let boundary_objects = runmat_analysis_execution::import_analysis_objects(
        source,
        &publication.boundary_root,
        AnalysisNumericalObjectKind::BoundaryResolution,
        boundary_manifest.logical_identity,
        boundary_manifest.payload_digest,
        &expected.artifact_access,
        limits,
    )
    .map_err(protocol)?;
    let expected_boundary_objects = boundary_objects
        .references(&expected.artifact_access, ValueRefKind::ResultObject)
        .map_err(protocol)?;
    if expected_boundary_objects != publication.boundary_objects {
        return Err(protocol(
            "boundary resolution inventory differs from its canonical closure",
        ));
    }
    let boundary_resolution =
        BoundaryRegionResolution::from_canonical_bytes(&boundary_objects.payload())?;
    if boundary_resolution.geometry_source_digest != expected.geometry_source_digest
        || boundary_resolution.solver_mesh_digest
            != Digest::from_bytes(*mesh.artifact.canonical_digest.bytes())
    {
        return Err(protocol(
            "boundary resolution differs from the returned geometry or mesh",
        ));
    }
    if publication.signal_roots.len() != expected.signal_materializations.len() {
        return Err(protocol(
            "whole-study signal result roots differ from the expected materializations",
        ));
    }
    let mut materialized_signals = BTreeMap::new();
    let mut admitted_signal_objects = Vec::new();
    for (root, prepared) in publication
        .signal_roots
        .iter()
        .zip(&expected.signal_materializations)
    {
        let manifest =
            runmat_analysis_fea::execution::AnalysisObjectManifest::from_canonical_bytes(&read(
                source, root,
            )?)
            .map_err(protocol)?;
        let objects = runmat_analysis_execution::import_analysis_objects(
            source,
            root,
            AnalysisNumericalObjectKind::Signal,
            manifest.logical_identity,
            manifest.payload_digest,
            &expected.artifact_access,
            limits,
        )
        .map_err(protocol)?;
        let references = objects
            .references(&expected.artifact_access, ValueRefKind::ResultObject)
            .map_err(protocol)?;
        for reference in references {
            if !publication.signal_objects.contains(&reference) {
                return Err(protocol(
                    "signal realization closure contains an undeclared object",
                ));
            }
            admitted_signal_objects.push(reference);
        }
        let realization =
            runmat_analysis_fea::execution::CanonicalSignalRealization::from_canonical_bytes(
                &objects.payload(),
            )
            .map_err(protocol)?;
        let signal = runmat_analysis_execution::materialize_signal_for_solver(
            &prepared.authored,
            &prepared.admitted,
            &realization,
        )
        .map_err(protocol)?;
        materialized_signals.insert(prepared.authored.signal_id().clone(), signal);
    }
    let admitted_signal_objects = unique_references(admitted_signal_objects)?;
    if reference_inventory(&admitted_signal_objects)?
        != reference_inventory(&publication.signal_objects)?
    {
        return Err(protocol(
            "signal realization inventory contains unreferenced objects",
        ));
    }
    let aliases =
        runmat_analysis_core::BoundaryRegionAliases::new(boundary_resolution.aliases.clone())
            .map_err(protocol)?;
    let mut resolved_workloads = expected.workloads.clone();
    for workload in resolved_workloads.values_mut() {
        workload.model.signals = workload
            .model
            .signals
            .iter()
            .map(|signal| {
                materialized_signals
                    .get(signal.signal_id())
                    .cloned()
                    .unwrap_or_else(|| signal.clone())
            })
            .collect();
        if workload
            .model
            .signals
            .iter()
            .any(|signal| signal.generator().is_some())
        {
            return Err(protocol(
                "whole-study result retained an unmaterialized signal generator",
            ));
        }
        workload.model =
            runmat_analysis_core::resolve_boundary_region_aliases(&workload.model, &aliases)
                .map_err(protocol)?;
    }
    runmat_analysis_execution::validate_structural_sequence_workloads(&resolved_workloads)
        .map_err(protocol)?;
    let sequence = runmat_analysis_execution::import_structural_sequence_completion(
        source,
        StructuralSequencePreparation {
            authored_study_digest: expected.authored_study_digest,
            frozen_project_digest: expected.frozen_project_digest,
            program_revision: expected.program_revision.clone(),
            artifact_access: expected.artifact_access.clone(),
            workloads: resolved_workloads.clone(),
            mesh: mesh.artifact.clone(),
            chunk_bytes: runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: limits,
            required_capabilities: BTreeSet::new(),
        },
        publication.analysis_closures,
    )
    .map_err(protocol)?;
    Ok(ExactStructuralStudyResult {
        frozen_project_digest: expected.frozen_project_digest,
        mesh,
        boundary_resolution,
        resolved_workloads,
        sequence,
    })
}

fn reference_inventory(
    references: &[ValueRef],
) -> NativeExecutionResult<BTreeMap<runmat_execution::identity::ValueId, ValueRef>> {
    let mut inventory = BTreeMap::new();
    let mut logical_digests = BTreeSet::new();
    for reference in references {
        if inventory.insert(reference.id, reference.clone()).is_some()
            || !logical_digests.insert(reference.logical_digest)
        {
            return Err(protocol(
                "whole-study result inventory contains duplicate object references",
            ));
        }
    }
    Ok(inventory)
}

fn unique_references(
    references: impl IntoIterator<Item = ValueRef>,
) -> NativeExecutionResult<Vec<ValueRef>> {
    let mut inventory = BTreeMap::new();
    for reference in references {
        match inventory.insert(reference.id, reference.clone()) {
            Some(existing) if existing != reference => {
                return Err(protocol(
                    "signal object identity is associated with conflicting metadata",
                ))
            }
            _ => {}
        }
    }
    Ok(inventory.into_values().collect())
}

#[cfg(test)]
mod tests {
    use runmat_execution::identity::ValueId;
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_V1;

    use super::*;

    fn reference(identity: &[u8]) -> ValueRef {
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_V1,
            id: ValueId::derive(&[b"whole-study-reference", identity]),
            logical_digest: Digest::sha256(identity),
            encoded_length: identity.len() as u64,
            media_type: "application/vnd.runmat.test".into(),
            value_schema: "runmat.test".into(),
            encryption_context: Digest::sha256(b"whole-study-reference-context"),
            kind: ValueRefKind::ResultObject,
            authorization_scope: "whole-study-test".into(),
            resident_fence: None,
        }
    }

    #[test]
    fn full_reference_inventory_rejects_duplicates_and_metadata_substitution() {
        let first = reference(b"first");
        let second = reference(b"second");
        let canonical = reference_inventory(&[first.clone(), second.clone()]).unwrap();
        assert_eq!(
            canonical,
            reference_inventory(&[second.clone(), first.clone()]).unwrap()
        );
        assert!(reference_inventory(&[first.clone(), first.clone()]).is_err());

        let mut substituted = first.clone();
        substituted.media_type = "application/vnd.runmat.substituted".into();
        assert_ne!(
            canonical,
            reference_inventory(&[substituted, second]).unwrap()
        );
    }
}
