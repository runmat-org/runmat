//! Closed original sources over the shared field transport; no ambient worker lookup.
use super::*;
use runmat_analysis_core::InitialFieldArtifactRef;
use runmat_analysis_execution::{
    decode_recovered_field_artifact, import_analysis_objects, prepare_recovered_field_objects,
    prepared_input_resource_facts,
};
use runmat_analysis_fea::{
    execution::{AnalysisNumericalObjectKind, InitialFieldBinding},
    planning::resource::PreparedInputResourceFacts,
};
use runmat_execution_artifact::object::ObjectInventoryLimits;

const SOURCE_LIMITS: CanonicalLimits = CanonicalLimits::platform_content(65_536, 64);
const SOURCE_CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct InitialFieldInput {
    pub reference: InitialFieldArtifactRef,
    pub manifest_digest: Digest,
    pub payload_digest: Digest,
    pub resources: PreparedInputResourceFacts,
}

pub(super) fn source_ceiling(study: &RealizedStudy) -> NativeExecutionResult<u64> {
    runmat_analysis_fea::planning::resource::initial_field_source_ceiling(&study.sequence)
        .map_err(protocol)
}

fn validate_references<'a>(
    study: &RealizedStudy,
    supplied: impl Iterator<Item = &'a InitialFieldArtifactRef>,
) -> NativeExecutionResult<()> {
    let declared = study.initial_field_references().collect::<Vec<_>>();
    let supplied = supplied.collect::<Vec<_>>();
    for (index, reference) in supplied.iter().enumerate() {
        if supplied[..index].contains(reference) || !declared.contains(reference) {
            return Err(protocol(
                "original source inventory is duplicate or unselected",
            ));
        }
    }
    if declared
        .iter()
        .any(|reference| !supplied.contains(reference))
    {
        return Err(protocol(
            "authored initial field has no closed original source",
        ));
    }
    Ok(())
}

pub(super) fn validate_bindings(
    study: &RealizedStudy,
    fields: &[InitialFieldBinding],
) -> NativeExecutionResult<()> {
    validate_references(study, fields.iter().map(InitialFieldBinding::reference))?;
    for field in fields {
        field.validate().map_err(protocol)?;
    }
    runmat_analysis_fea::planning::resource::require_initial_field_ceiling(
        fields,
        SOURCE_LIMITS,
        source_ceiling(study)?,
    )
    .map_err(protocol)
}

fn charge(resources: &PreparedInputResourceFacts) -> NativeExecutionResult<u64> {
    // Original transport and its store handoff coexist with canonical decoding,
    // reconstructed physical fields and closed source/expectation ownership.
    resources
        .decoded_resident_bytes_admission_budget
        .0
        .checked_mul(3)
        .and_then(|resident| {
            resources
                .transport_closure_bytes
                .0
                .checked_mul(2)
                .and_then(|transport| resident.checked_add(transport))
        })
        .ok_or_else(|| protocol("original source transport residency overflows"))
}

pub(super) fn validate_inputs(
    study: &RealizedStudy,
    inputs: &[InitialFieldInput],
) -> NativeExecutionResult<()> {
    validate_references(study, inputs.iter().map(|input| &input.reference))?;
    let mut remaining = source_ceiling(study)?;
    for input in inputs {
        if input.manifest_digest.bytes() == &[0; 32]
            || input.payload_digest.bytes() == &[0; 32]
            || input.resources.encoded_payload_bytes.0 == 0
            || input.resources.transport_closure_bytes.0 < input.resources.encoded_payload_bytes.0
            || input.resources.decoded_resident_bytes_admission_budget.0 == 0
        {
            return Err(protocol("original field input descriptor is invalid"));
        }
        remaining = remaining
            .checked_sub(charge(&input.resources)?)
            .ok_or_else(|| protocol("original source union exceeds its residency ceiling"))?;
    }
    Ok(())
}

pub(super) struct PreparedSources {
    pub inputs: Vec<InitialFieldInput>,
    pub arguments: Vec<ValuePayload>,
    pub objects: Vec<LogicalObject>,
}

pub(super) fn prepare(
    study: &RealizedStudy,
    fields: &[InitialFieldBinding],
    access: &ArtifactAccess,
) -> NativeExecutionResult<PreparedSources> {
    validate_bindings(study, fields)?;
    let maximum = source_ceiling(study)?;
    let mut construction_remaining = maximum;
    for field in fields {
        let required = runmat_analysis_execution::recovered_field_transport_reservation_bytes(
            field.artifact(),
            SOURCE_CHUNK_BYTES,
        )
        .map_err(protocol)?;
        construction_remaining = construction_remaining
            .checked_sub(required)
            .ok_or_else(|| {
                protocol("original source construction exceeds its cumulative residency ceiling")
            })?;
    }
    let limits = ObjectInventoryLimits {
        max_object_bytes: maximum,
        max_total_bytes: maximum,
        ..ObjectInventoryLimits::default()
    };
    let mut inputs = Vec::new();
    let mut objects = BTreeMap::new();
    let mut references = BTreeMap::new();
    let mut remaining = maximum;
    for field in fields {
        let prepared = prepare_recovered_field_objects(
            field.artifact(),
            SOURCE_CHUNK_BYTES,
            ObjectNamespace::InputValue,
            limits,
        )
        .map_err(protocol)?;
        let resources =
            prepared_input_resource_facts(std::slice::from_ref(&prepared)).map_err(protocol)?;
        remaining = remaining
            .checked_sub(charge(&resources)?)
            .ok_or_else(|| protocol("original source union exceeds its residency ceiling"))?;
        inputs.push(InitialFieldInput {
            reference: field.reference().clone(),
            manifest_digest: prepared.root.digest,
            payload_digest: prepared.manifest.payload_digest,
            resources,
        });
        for reference in prepared
            .unique_references(access, ValueRefKind::DriverObject)
            .map_err(protocol)?
        {
            references.insert(reference.id, ValuePayload::Object(Box::new(reference)));
        }
        for object in prepared.objects {
            objects.insert(object.descriptor.digest, object);
        }
    }
    validate_inputs(study, &inputs)?;
    Ok(PreparedSources {
        inputs,
        arguments: references.into_values().collect(),
        objects: objects.into_values().collect(),
    })
}

pub(super) fn decode(
    study: &RealizedStudy,
    inputs: &[InitialFieldInput],
    arguments: &[ValuePayload],
    access: &ArtifactAccess,
    store: &impl CacheImport,
) -> NativeExecutionResult<Vec<InitialFieldBinding>> {
    validate_inputs(study, inputs)?;
    let supplied = arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(reference) => Ok((reference.id, reference.as_ref())),
            _ => Err(protocol("original source inputs must be externalized")),
        })
        .collect::<NativeExecutionResult<BTreeMap<_, _>>>()?;
    if supplied.len() != arguments.len() {
        return Err(protocol("original source arguments are duplicated"));
    }
    let mut expected = BTreeMap::new();
    let mut fields = Vec::new();
    let maximum = source_ceiling(study)?;
    for input in inputs {
        let root = supplied
            .values()
            .find(|reference| reference.logical_digest == input.manifest_digest)
            .ok_or_else(|| protocol("original source manifest is not transported"))?;
        let limits = ObjectInventoryLimits {
            max_object_bytes: maximum,
            max_total_bytes: input.resources.transport_closure_bytes.0,
            ..ObjectInventoryLimits::default()
        };
        let prepared = import_analysis_objects(
            store,
            root,
            AnalysisNumericalObjectKind::Field,
            input.reference.content_digest.parse().map_err(protocol)?,
            input.payload_digest,
            access,
            limits,
        )
        .map_err(protocol)?;
        if prepared_input_resource_facts(std::slice::from_ref(&prepared)).map_err(protocol)?
            != input.resources
        {
            return Err(protocol(
                "original source transport differs from its admitted resource facts",
            ));
        }
        for reference in prepared
            .unique_references(access, ValueRefKind::DriverObject)
            .map_err(protocol)?
        {
            expected.insert(reference.id, reference);
        }
        let artifact = decode_recovered_field_artifact(&prepared).map_err(protocol)?;
        fields.push(InitialFieldBinding::new(input.reference.clone(), artifact).map_err(protocol)?);
    }
    if supplied.len() != expected.len()
        || expected
            .iter()
            .any(|(id, reference)| supplied.get(id).is_none_or(|actual| *actual != reference))
    {
        return Err(protocol(
            "original field transport inventory differs from its closure",
        ));
    }
    validate_bindings(study, &fields)?;
    Ok(fields)
}
