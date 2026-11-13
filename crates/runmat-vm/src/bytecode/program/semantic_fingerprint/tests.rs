use serde::Serialize;

use crate::bytecode::fusion_stage::FusionStageStream;
use crate::bytecode::{BytecodeSourceRef, FusionStageProjection};

use super::{fusion_stage_source_fingerprint, SelectedFunctionLayoutIdentity};

// Keep the pre-origin eight-argument algorithm spelled independently of the
// production helper. In particular, do not serialize the projection as a
// whole: its new optional fields were never part of the historical payload.
#[derive(Serialize)]
enum HistoricalSourceRevision {
    Program(runmat_types::ProgramSourceId),
    Runtime(runmat_execution::Digest),
}

fn append_json<T: Serialize + ?Sized>(payload: &mut Vec<u8>, value: &T) {
    let encoded = serde_json::to_vec(value).expect("historical JSON field");
    payload.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
    payload.extend_from_slice(&encoded);
}

fn historical_payload(
    projection: &FusionStageProjection,
    stream: &FusionStageStream<'_>,
) -> Vec<u8> {
    let mut payload = Vec::new();
    append_json(&mut payload, &"selected-mir-fusion-stage-stream-v1");
    append_json(&mut payload, &projection.revision);
    append_json(&mut payload, &projection.portable_origin);
    append_json(&mut payload, &projection.owner);
    append_json(&mut payload, &projection.mir_local_count);
    append_json(&mut payload, &projection.statements);
    append_json(&mut payload, stream.instructions);
    append_json(&mut payload, stream.catch_scopes);
    append_json(
        &mut payload,
        &stream.resume_points.iter().collect::<Vec<_>>(),
    );
    append_json(&mut payload, &stream.var_count);
    let source = stream.source_id.map(|source| match source {
        BytecodeSourceRef::Program(id) => HistoricalSourceRevision::Program(id),
        BytecodeSourceRef::Runtime(reference) => {
            HistoricalSourceRevision::Runtime(reference.semantic_content_revision())
        }
    });
    append_json(&mut payload, &source);
    payload
}

fn digest(payload: &[u8]) -> String {
    runmat_execution::Digest::sha256(payload).to_string()
}

fn assert_history_and_extensions(
    projection: &FusionStageProjection,
    stream: &FusionStageStream<'_>,
) {
    assert!(
        projection.selection_origin.is_some(),
        "authentic selected compiler origin required"
    );
    assert!(
        !projection.private_temporary_slots.is_empty(),
        "authentic compiler private inventory required"
    );
    let old_payload = historical_payload(projection, stream);
    let old_digest = digest(&old_payload);
    let mut expected_modern_payload = old_payload;
    if let Some(layout) = &projection.selected_function_layout {
        append_json(
            &mut expected_modern_payload,
            &"selected-mir-function-layout-v1",
        );
        append_json(
            &mut expected_modern_payload,
            &SelectedFunctionLayoutIdentity::from_layout(layout),
        );
    }
    append_json(
        &mut expected_modern_payload,
        &"selected-mir-fusion-stage-origin-v1",
    );
    append_json(
        &mut expected_modern_payload,
        &projection.selection_origin.expect("selected origin"),
    );
    append_json(
        &mut expected_modern_payload,
        &"selected-mir-private-temporary-slots-v1",
    );
    append_json(
        &mut expected_modern_payload,
        &projection.private_temporary_slots,
    );
    let modern_digest = fusion_stage_source_fingerprint(projection, stream).unwrap();
    assert_eq!(modern_digest, digest(&expected_modern_payload));
    assert_eq!(modern_digest, projection.source_fingerprint);
    assert_ne!(
        modern_digest, old_digest,
        "new authority must extend the seal"
    );

    let mut legacy = projection.clone();
    legacy.selection_origin = None;
    legacy.private_temporary_slots.clear();
    legacy.selected_function_layout = None;
    let expected_legacy = digest(&historical_payload(&legacy, stream));
    legacy.source_fingerprint = expected_legacy.clone();
    let wire = serde_json::to_value(&legacy).expect("legacy projection wire");
    assert!(wire.get("selection_origin").is_none());
    assert!(wire.get("private_temporary_slots").is_none());
    assert!(wire.get("selected_function_layout").is_none());
    assert_eq!(expected_legacy, old_digest);
    assert_eq!(
        fusion_stage_source_fingerprint(&legacy, stream).unwrap(),
        expected_legacy,
        "fieldless legacy origin/inventory must retain exact historical bytes"
    );
    assert!(legacy.validate(FusionStageStream {
        instructions: stream.instructions,
        catch_scopes: stream.catch_scopes,
        resume_points: stream.resume_points,
        source_id: stream.source_id,
        var_count: stream.var_count,
    }));
}

#[test]
fn authentic_entrypoint_preserves_historical_payload_and_seals_private_inventory() {
    let ast = runmat_parser::parse("x = gpuArray(single([1,2;3,4])); y = min(x+x);")
        .expect("authentic entrypoint source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    let bytecode = crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("selected bytecode");
    let projection = bytecode
        .fusion_metadata
        .stage_projection
        .as_ref()
        .expect("selected entrypoint projection");
    assert_eq!(
        projection.selection_origin,
        Some(crate::bytecode::FusionStageSelectionOrigin::EntrypointStream)
    );
    assert_history_and_extensions(
        projection,
        &FusionStageStream {
            instructions: &bytecode.instructions,
            catch_scopes: &bytecode.catch_scopes,
            resume_points: &bytecode.resume_points,
            source_id: bytecode.source_id,
            var_count: bytecode.var_count,
        },
    );
}

#[test]
fn authentic_function_preserves_historical_payload_and_seals_private_inventory() {
    let ast = runmat_parser::parse("function y = folded_min(x)\ny = min(x+x);\nend\n")
        .expect("authentic function source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("MIR");
    assert!(hir.assembly.entrypoints.is_empty());
    let function = hir.assembly.functions[0].id;
    let registry = crate::bytecode::FunctionRegistry::new(
        crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir)
            .expect("compiled function"),
    );
    let body = registry.get(function).expect("selected function body");
    let projection = body
        .fusion_stage_projection
        .as_ref()
        .expect("selected function projection");
    assert!(projection.selected_function_layout.is_some());
    assert_eq!(
        projection.selection_origin,
        Some(crate::bytecode::FusionStageSelectionOrigin::FunctionBody)
    );
    assert_history_and_extensions(
        projection,
        &FusionStageStream {
            instructions: &body.instructions,
            catch_scopes: &body.catch_scopes,
            resume_points: &body.resume_points,
            source_id: body.source_id,
            var_count: body.var_count,
        },
    );
    let mut reordered = (**projection).clone();
    let layout = reordered
        .selected_function_layout
        .as_mut()
        .expect("selected function layout");
    let mut bindings = layout.binding_slots.drain().collect::<Vec<_>>();
    bindings.sort_by_key(|(id, _)| *id);
    layout.binding_slots.extend(bindings.into_iter().rev());
    let mut locals = layout.mir_local_slots.drain().collect::<Vec<_>>();
    locals.sort_by_key(|(id, _)| *id);
    layout.mir_local_slots.extend(locals.into_iter().rev());
    let stream = FusionStageStream {
        instructions: &body.instructions,
        catch_scopes: &body.catch_scopes,
        resume_points: &body.resume_points,
        source_id: body.source_id,
        var_count: body.var_count,
    };
    assert_eq!(
        fusion_stage_source_fingerprint(&reordered, &stream).unwrap(),
        projection.source_fingerprint,
        "equivalent hydrated map order must preserve the selected source seal"
    );
    assert!(reordered.validate(stream));
}
