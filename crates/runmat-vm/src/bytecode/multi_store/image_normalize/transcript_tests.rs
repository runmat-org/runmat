use super::*;

fn compile_publication_fixture(compile: fn() -> Bytecode) -> Bytecode {
    // Direct compiler fixtures do not traverse the interpreter/dispatcher
    // entrypoint that initializes canonical browser implementation and fusion
    // registrations. Use that same real runtime owner, never a fake catalogue.
    #[cfg(target_arch = "wasm32")]
    runmat_runtime::builtins::wasm_registry::register_all();
    compile()
}

/// Check the publication fields against the selected original bytecode's
/// graph, sealed region and actual stores, not a synthetic producer table.
fn assert_selected_publication_identities(
    bytecode: &Bytecode,
    joined: &ImageNormalizePublicationJoin,
) {
    let selected = SelectedMultiStoreGraph::for_image(bytecode)
        .expect("original selected graph/store authority");
    let projection = bytecode
        .fusion_metadata
        .stage_projection
        .as_ref()
        .expect("sealed statement projection");
    let matching = selected
        .graph
        .detect_fusion_groups()
        .into_iter()
        .filter(|group| group.kind == FusionKind::ImageNormalize)
        .filter_map(|group| {
            let region = selected_image_region(bytecode, projection, &selected, &group)?;
            (region.pcs == joined.pcs()).then_some((group, region))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matching.len(),
        1,
        "one exact selected source region owns this join"
    );
    let (group, region) = &matching[0];
    let FusionPattern::ImageNormalize(pattern) = group.pattern.as_ref().unwrap() else {
        unreachable!("selected image pattern")
    };
    let producers = pattern
        .reduction_routes
        .as_ref()
        .expect("source-exact mean routes")
        .stage_producers()
        .expect("raw arithmetic producers from the original selected graph");
    let source = ImageSourceTranscript::from_graph(&selected.graph, group)
        .expect("authenticated complete selected transcript");

    assert_eq!(
        joined.ordered_stores().len(),
        region.stores.len(),
        "no hidden or public store may disappear"
    );
    assert_eq!(
        joined.live_outs(),
        region.live_outs.as_slice(),
        "publication identities must preserve all selected live-outs"
    );
    for (entry, actual) in joined.ordered_stores().iter().zip(&region.stores) {
        assert_eq!(
            entry.store, actual.store,
            "exact PC, binding and value of the original store"
        );
        assert_eq!(entry.public_output, actual.store.value_id);
        assert_eq!(entry.public_output, entry.store.value_id);
        assert_eq!(entry.slot, actual.slot);
        assert_eq!(entry.mir_defs, actual.defs);
        assert_eq!(entry.statement, actual.stage.contract.point);
        if matches!(entry.destination, ImageStoreDestination::Intermediate) {
            assert_eq!(
                entry.raw_stage_dependency, None,
                "compiler intermediates have no source-visible checkpoint dependency"
            );
        } else {
            assert!(entry.raw_stage_dependency.is_some());
        }
    }

    for (role, raw) in [
        (ImageCheckpoint::Mu, producers.mu),
        (ImageCheckpoint::Sigma, producers.sigma),
    ] {
        let entry = joined
            .ordered_stores()
            .iter()
            .find(|entry| entry.destination == ImageStoreDestination::Checkpoint(role))
            .expect("actual selected moment store");
        assert_eq!(entry.raw_stage_dependency, Some(raw));
        assert_ne!(
            entry.public_output, raw,
            "the fixture's selected Single call must publish its own output, never the raw moment"
        );
        let raw_step = source
            .steps
            .iter()
            .find(|step| step.output == raw)
            .expect("raw reduction/SQRT stays in arithmetic transcript");
        let public_step = source
            .steps
            .iter()
            .find(|step| step.output == entry.public_output)
            .expect("actual stored cast output is a selected source step");
        let expected_raw_binding = match role {
            ImageCheckpoint::Mu => runmat_builtins::MEAN_CATALOG_ENTRY.identity,
            ImageCheckpoint::Sigma => runmat_builtins::SQRT_CATALOG_ENTRY.identity,
            _ => unreachable!("moment role"),
        };
        assert_eq!(
            raw_step
                .selected_binding
                .expect("canonical raw moment binding")
                .builtin,
            expected_raw_binding
        );
        assert_eq!(
            public_step
                .selected_binding
                .expect("canonical public cast binding")
                .builtin,
            runmat_builtins::SINGLE_CATALOG_ENTRY.identity
        );
        assert_eq!(
            public_step.inputs.as_slice(),
            [raw],
            "exact original raw-to-public conversion edge"
        );
        assert!(
            raw_step.pc.end < public_step.pc.start && public_step.pc.end < entry.store.pc,
            "raw producer, public cast and actual store retain source PC order"
        );
        assert!(publication_follows_stage(
            &selected.graph,
            raw,
            entry.public_output,
            region.pcs.clone()
        ));
    }

    let mut expected = vec![
        (ImageCheckpoint::Mu, producers.mu),
        (ImageCheckpoint::Sigma, producers.sigma),
    ];
    expected.extend(
        producers
            .pre_clamp_output
            .map(|output| (ImageCheckpoint::PreClamp, output)),
    );
    expected.extend(
        producers
            .clamp_output
            .map(|output| (ImageCheckpoint::Clamp, output)),
    );
    expected.push((ImageCheckpoint::Final, producers.output));
    assert_eq!(
        source.checkpoints,
        expected.iter().map(|(_, raw)| *raw).collect::<Vec<_>>()
    );
    let checkpoints = joined
        .ordered_stores()
        .iter()
        .filter_map(|entry| {
            let ImageStoreDestination::Checkpoint(role) = entry.destination else {
                return None;
            };
            Some((role, entry))
        })
        .collect::<Vec<_>>();
    assert_eq!(checkpoints.len(), expected.len());
    for ((role, entry), (expected_role, producer)) in checkpoints.iter().zip(&expected) {
        assert_eq!(role, expected_role);
        assert!(publication_follows_stage(
            &selected.graph,
            *producer,
            entry.public_output,
            region.pcs.clone()
        ));
    }
    let final_public = checkpoints
        .last()
        .expect("selected final store")
        .1
        .public_output;
    let selected_regions = bytecode
        .selected_image_regions()
        .expect("selected image region and output-cast source plan");
    let selected_region = selected_regions
        .regions
        .iter()
        .find(|selected| selected.publication.pcs() == joined.pcs())
        .expect("same selected source publication region");
    let selected_source =
        ImageSourceTranscript::from_graph(&selected_regions.graph, &selected_region.plan.group)
            .expect("exact extended image transcript");
    assert_eq!(
        selected_source.steps.last().map(|step| step.output),
        Some(final_public),
        "the physical plan ends at the actual final source cast, not raw POWER"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn compiled_moment_publications_join_actual_cast_outputs_and_preserve_selected_liveouts() {
    for compile in [
        super::tests::compiled_image as fn() -> Bytecode,
        compiled_with_live_entry,
    ] {
        let bytecode = compile_publication_fixture(compile);
        let joined = bytecode
            .prospective_image_normalize_publications()
            .into_iter()
            .find(|join| join.checkpoint(ImageCheckpoint::Final).is_some())
            .expect("original compiled image join");
        assert_selected_publication_identities(&bytecode, &joined);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn compiled_clamped_publication_identities_keep_all_five_source_stores() {
    let bytecode = compile_publication_fixture(super::tests::compiled_clamped_image);
    let joined = bytecode
        .prospective_image_normalize_publications()
        .into_iter()
        .find(|join| join.checkpoint(ImageCheckpoint::Final).is_some())
        .expect("original compiled clamped image join");
    assert_selected_publication_identities(&bytecode, &joined);
    let checkpoints = joined
        .ordered_stores()
        .iter()
        .filter_map(|entry| {
            let ImageStoreDestination::Checkpoint(role) = entry.destination else {
                return None;
            };
            Some((role, entry))
        })
        .collect::<Vec<_>>();
    assert!(checkpoints.iter().map(|(role, _)| *role).eq([
        ImageCheckpoint::Mu,
        ImageCheckpoint::Sigma,
        ImageCheckpoint::PreClamp,
        ImageCheckpoint::Clamp,
        ImageCheckpoint::Final
    ]));
    assert_eq!(checkpoints.len(), 5);
    assert!(checkpoints
        .windows(2)
        .all(|pair| pair[0].1.store.pc < pair[1].1.store.pc));
    let outputs = &checkpoints[2..];
    assert_eq!(outputs.len(), 3);
    assert!(outputs
        .windows(2)
        .all(|pair| pair[0].1.store.binding == pair[1].1.store.binding));
    assert!(outputs
        .windows(2)
        .all(|pair| pair[0].1.public_output != pair[1].1.public_output));
}

fn compiled_with_live_entry() -> Bytecode {
    let source = r#"
        imgs = ones(2,4,5,'single');
        mu = single(mean(imgs,[2 3],'native'));
        sigma = single(sqrt(mean((imgs-mu).^2,[2 3],'native')+single(1e-6)));
        out = single(((imgs-mu)./sigma).^single(1.8));
        later = imgs + out;
    "#;
    let ast = runmat_parser::parse(source).expect("parse image source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile image source")
}

#[test]
fn image_transcript_preserves_hidden_stores_and_exact_last_store_links() {
    let bytecode = super::tests::compiled_image();
    let joined = bytecode
        .prospective_image_normalize_publications()
        .into_iter()
        .find(|join| join.checkpoint(ImageCheckpoint::Final).is_some())
        .expect("compiled image retains its selected region");
    assert!(joined
        .ordered_stores()
        .windows(2)
        .all(|pair| pair[0].store.pc < pair[1].store.pc));
    assert!(joined.ordered_stores().iter().any(|entry| {
        entry.destination == ImageStoreDestination::Intermediate
            && !entry.mir_defs.is_empty()
            && !bytecode.var_names.contains_key(&entry.slot.0)
    }));
    for entry in joined.ordered_stores() {
        assert_eq!(entry.slot.0, entry.store.binding.index);
        let stage = bytecode
            .fusion_metadata
            .stage_projection
            .as_ref()
            .unwrap()
            .statement_at(entry.statement)
            .expect("store retains its exact sealed statement");
        assert!(stage.pc.start <= entry.store.pc && entry.store.pc < stage.pc.end);
        assert!(entry
            .mir_defs
            .iter()
            .all(|value| stage.contract.defs.contains(value)));
        assert!(joined.pcs().contains(&entry.store.pc));
    }
    for live in joined.live_outs() {
        if let ImageLiveOutSource::LastStore(ordinal) = live.source {
            let store = joined
                .store_at(ordinal)
                .expect("bounded typed store ordinal");
            assert_eq!(store.slot, live.slot);
            assert!(store.mir_defs.contains(&live.value));
            assert!(joined.ordered_stores()[ordinal.index() + 1..]
                .iter()
                .all(|later| later.slot != live.slot));
        }
    }
}

#[test]
fn named_workspace_store_cannot_be_demoted_to_hidden_by_layout_loss() {
    let mut bytecode = super::tests::compiled_image();
    let selected = SelectedMultiStoreGraph::for_image(&bytecode).expect("selected graph");
    let function = selected.selected_function().expect("selected function");
    let mu = bytecode
        .prospective_image_normalize_publications()
        .into_iter()
        .find_map(|join| join.checkpoint(ImageCheckpoint::Mu).cloned())
        .expect("selected mu store");
    let layout = bytecode
        .layout
        .as_mut()
        .unwrap()
        .functions
        .get_mut(&function)
        .unwrap();
    layout
        .binding_slots
        .retain(|_, slot| slot.0 != mu.binding.index);
    assert!(bytecode
        .prospective_image_normalize_publications()
        .is_empty());
}

#[test]
fn live_entry_requires_preservation_and_hidden_def_requires_selected_slot() {
    let mut bytecode = compiled_with_live_entry();
    let joined = bytecode
        .prospective_image_normalize_publications()
        .into_iter()
        .find(|join| join.checkpoint(ImageCheckpoint::Final).is_some())
        .expect("selected image with later use");
    assert!(joined.live_outs().iter().any(|live| {
        live.source == ImageLiveOutSource::PreserveEntry
            && joined
                .ordered_stores()
                .iter()
                .all(|store| store.slot != live.slot)
    }));
    let hidden = joined
        .ordered_stores()
        .iter()
        .find(|entry| entry.destination == ImageStoreDestination::Intermediate)
        .expect("compiler emits an internal store");
    let local = runmat_mir::MirLocalId(hidden.mir_defs[0].local as usize);
    let function = SelectedMultiStoreGraph::for_image(&bytecode)
        .and_then(|selected| selected.selected_function())
        .expect("selected function");
    bytecode
        .layout
        .as_mut()
        .unwrap()
        .functions
        .get_mut(&function)
        .unwrap()
        .mir_local_slots
        .remove(&local);
    assert!(bytecode
        .prospective_image_normalize_publications()
        .is_empty());
}

#[test]
fn repeated_source_output_binding_keeps_three_distinct_ordinals() {
    let bytecode = super::tests::compiled_clamped_image();
    let joined = bytecode
        .prospective_image_normalize_publications()
        .into_iter()
        .find(|join| join.checkpoint(ImageCheckpoint::Final).is_some())
        .expect("selected full image");
    let outputs = joined.output_checkpoints().collect::<Vec<_>>();
    assert_eq!(outputs.len(), 3);
    assert!(outputs.windows(2).all(|pair| pair[0].pc < pair[1].pc));
    assert!(outputs
        .windows(2)
        .all(|pair| pair[0].binding == pair[1].binding));
    for role in [
        ImageCheckpoint::PreClamp,
        ImageCheckpoint::Clamp,
        ImageCheckpoint::Final,
    ] {
        assert!(joined.checkpoint(role).is_some());
    }
}
