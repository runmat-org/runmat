//! Cold result import retains the exact predecessor closure for the next task.
use super::*;
use runmat_analysis_fea::execution::NonlinearSolidNumericalReceiptArtifact;
use runmat_execution_artifact::{cache::CacheImport, ProgramExecutionResponse};

pub struct ImportedNonlinearSolidBlock {
    pub block: NonlinearSolidBlockArtifact,
    pub objects: PreparedAnalysisObjects,
    pub numerical_receipt: NonlinearSolidNumericalReceiptArtifact,
    pub receipt_objects: PreparedAnalysisObjects,
}

impl ImportedNonlinearSolidBlock {
    /// Original typed artifact/checkpoint/receipt plus the owned immutable wire
    /// closures. Temporary decoding is separate; this is a shared bound, not RSS.
    pub fn retained_storage_bytes(&self, maximum_bytes: u64) -> AnalysisBridgeResult<u64> {
        use runmat_analysis_fea::planning::resource::ResourcePlanningError;
        let overflow = || ResourcePlanningError::Overflow("original nonlinear block retention");
        let profile = runmat_canonical_codec::profile_serialized(
            &(&self.block, &self.numerical_receipt),
            runmat_canonical_codec::CanonicalLimits::content_bounded(
                usize::try_from(maximum_bytes).map_err(|_| overflow())?,
                65_536,
                64,
            ),
        )?;
        let objects = self.objects.retained_storage_bytes()?;
        let receipts = self.receipt_objects.retained_storage_bytes()?;
        Ok(profile
            .materialization_resident_bytes_estimate
            .checked_add(std::mem::size_of::<Self>() as u64)
            .and_then(|bytes| bytes.checked_add(objects))
            .and_then(|bytes| bytes.checked_add(receipts))
            .ok_or_else(overflow)?)
    }
}

pub fn import_nonlinear_solid_block(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedNonlinearSolidBlockTask,
) -> AnalysisBridgeResult<ImportedNonlinearSolidBlock> {
    prepared.host.validate()?;
    if prepared.host.workload.kind != AnalysisWorkKind::SolveNonlinearSolidBlock
        || prepared.host.workload.outputs
            != [Kind::NonlinearSolidBlockArtifact, Kind::NumericalReceipt]
    {
        return Err(invalid("nonlinear import task differs"));
    }
    require_nonlinear_solid_source_decode_budget(
        &prepared.inputs,
        &prepared.resource_plan,
        prepared.output_policy.inventory_limits(),
    )?;
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output, receipt_output] = output_set.manifest.outputs.as_slice() else {
        return Err(invalid(
            "nonlinear block requires original artifact and numerical receipt",
        ));
    };
    if output.kind != Kind::NonlinearSolidBlockArtifact
        || receipt_output.kind != Kind::NumericalReceipt
    {
        return Err(invalid("nonlinear output kind differs"));
    }
    let source_input = |kind| {
        prepared
            .inputs
            .iter()
            .find(|input| input.manifest.kind == kind)
            .ok_or_else(|| invalid("nonlinear source input missing"))
    };
    let workload = NonlinearSolidWorkload::from_canonical_bytes(
        &source_input(Kind::NonlinearSolidWorkload)?.payload(),
    )?;
    let mesh = SolverMeshArtifact::canonical_decode(&source_input(Kind::Mesh)?.payload())
        .map_err(invalid)?;
    let request = NonlinearSolidBlockRequest::from_canonical_bytes(
        &source_input(Kind::NonlinearSolidBlockRequest)?.payload(),
    )?;
    let previous = predecessor(
        prepared
            .inputs
            .iter()
            .find(|input| input.manifest.kind == Kind::NonlinearSolidBlockArtifact),
    )?;
    if request != prepared.request
        || prepared.host.workload.step != workload.producer_identity()?
        || rederive_nonlinear_solid_block_resource_plan(
            &workload,
            &mesh,
            &request,
            previous.as_ref(),
            &prepared.inputs,
            prepared.output_policy,
        )? != prepared.resource_plan
        || prepared.host.workload.execution_plan
            != ExecutionPlanIdentity::new(
                NumericalPlanIdentity::from_nonlinear_solid_block(&request)?,
                prepared.resource_plan.identity.digest(),
            )?
    {
        return Err(invalid("nonlinear imported source authority differs"));
    }
    let facts = prepared
        .resource_plan
        .input
        .demand
        .output_facts(Kind::NonlinearSolidBlockArtifact)
        .ok_or_else(|| invalid("nonlinear output admission missing"))?;
    let receipt_facts = prepared
        .resource_plan
        .input
        .demand
        .output_facts(Kind::NumericalReceipt)
        .ok_or_else(|| invalid("nonlinear receipt admission missing"))?;
    let mut bounded = prepared.output_policy.inventory_limits();
    bounded.max_total_bytes = bounded
        .max_total_bytes
        .min(facts.transport_closure_bytes_upper_bound.0);
    let objects = import_analysis_objects(
        source,
        &output.root,
        Kind::NonlinearSolidBlockArtifact,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        bounded,
    )?;
    let mut bounded = prepared.output_policy.inventory_limits();
    bounded.max_total_bytes = bounded
        .max_total_bytes
        .min(receipt_facts.transport_closure_bytes_upper_bound.0);
    let receipt_objects = import_analysis_objects(
        source,
        &receipt_output.root,
        Kind::NumericalReceipt,
        receipt_output.logical_identity,
        receipt_output.payload_digest,
        &prepared.host.artifact_access,
        bounded,
    )?;
    if objects.manifest.encoded_length > facts.encoded_payload_bytes_upper_bound.0
        || receipt_objects.manifest.encoded_length
            > receipt_facts.encoded_payload_bytes_upper_bound.0
    {
        return Err(invalid(
            "nonlinear imported block exceeds original wire admission",
        ));
    }
    let block = NonlinearSolidBlockArtifact::from_canonical_bytes(
        &objects.payload(),
        output.logical_identity,
    )?;
    if block.request != prepared.request {
        return Err(invalid(
            "nonlinear output differs from original block request",
        ));
    }
    let numerical_receipt =
        NonlinearSolidNumericalReceiptArtifact::from_canonical_bytes(&receipt_objects.payload())?;
    numerical_receipt.validate_against(&workload, &block)?;
    if numerical_receipt.identity().digest() != receipt_objects.manifest.logical_identity {
        return Err(invalid("nonlinear receipt logical identity differs"));
    }
    let mut expected_objects =
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected_objects.extend(
        receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut expected_objects);
    expected_objects.push(output_set.root);
    if output_set.result_objects != expected_objects {
        return Err(invalid(
            "nonlinear output inventory differs from original closure",
        ));
    }
    Ok(ImportedNonlinearSolidBlock {
        block,
        objects,
        numerical_receipt,
        receipt_objects,
    })
}
