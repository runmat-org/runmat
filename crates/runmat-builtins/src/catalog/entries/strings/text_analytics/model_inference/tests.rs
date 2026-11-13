use super::*;

fn request(arguments: Vec<ValueFact>) -> CallRequest {
    CallRequest {
        arguments,
        literals: LiteralContext::default(),
        outputs: OutputSelection::default(),
    }
}

#[test]
fn identity_selected_forms_keep_arity_and_definitive_model_errors() {
    use super::super::{
        bag_of_words, remove_short_words, remove_words, tokenized_document, write_word_embedding,
    };
    for (entry, op, counts) in [
        (
            &tokenized_document::TOKENIZED_DOCUMENT_CATALOG_ENTRY,
            ModelOperation::Document,
            vec![(0, true), (1, true), (2, false), (3, true)],
        ),
        (
            &bag_of_words::BAG_OF_WORDS_CATALOG_ENTRY,
            ModelOperation::Bag,
            vec![(0, true), (2, true), (3, false)],
        ),
        (
            &remove_short_words::REMOVE_SHORT_WORDS_CATALOG_ENTRY,
            ModelOperation::Filter,
            vec![(1, false), (2, true), (4, false)],
        ),
        (
            &remove_words::REMOVE_WORDS_CATALOG_ENTRY,
            ModelOperation::Select,
            vec![(1, false), (2, true), (3, false), (4, true)],
        ),
        (
            &write_word_embedding::WRITE_WORD_EMBEDDING_CATALOG_ENTRY,
            ModelOperation::Write,
            vec![(0, false), (1, false), (2, true), (3, false)],
        ),
    ] {
        for (count, valid) in counts {
            let result = infer(
                &request(vec![ValueFact::unknown(DynamicReason::RuntimeValue); count]),
                entry,
                op,
            );
            assert_eq!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| d.code == "RM-CATALOG-TEXT-MODEL-ARITY"),
                !valid,
                "{} {count}",
                entry.identity.name
            );
        }
    }
    let writer = &write_word_embedding::WRITE_WORD_EMBEDDING_CATALOG_ENTRY;
    assert!(writer
        .contract
        .capabilities
        .contains(&CapabilityRequirement::Filesystem));
    assert!(writer
        .contract
        .effects
        .contains(&EffectKind::FilesystemWrite));
    let result = infer(
        &request(vec![
            numeric(NumericDomain::Real, ShapeFact::Scalar, StorageFact::Dense),
            ValueFact::unknown(DynamicReason::RuntimeValue),
        ]),
        &remove_words::REMOVE_WORDS_CATALOG_ENTRY,
        ModelOperation::Select,
    );
    assert!(result
        .diagnostics
        .iter()
        .any(|d| d.code == "RM-CATALOG-TEXT-MODEL-CLASS"));
}

#[test]
fn lookup_preserves_string_and_character_row_geometry() {
    let enc = object(Some(StaticClassIdentity::new("wordEncoding")));
    for (kind, shape, expected) in [
        (
            ValueKindFact::String,
            ShapeFact::from(vec![Some(2), Some(3)]),
            ShapeFact::from(vec![Some(2), Some(3)]),
        ),
        (
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(2), Some(8)]),
            ShapeFact::from(vec![Some(2), Some(1)]),
        ),
        (
            ValueKindFact::String,
            ShapeFact::from(vec![Some(0), Some(3)]),
            ShapeFact::from(vec![Some(0), Some(3)]),
        ),
    ] {
        let result = infer(
            &request(vec![
                enc.clone(),
                ValueFact::proven(kind, shape, StorageFact::Dense),
            ]),
            &super::super::word2ind::WORD2IND_CATALOG_ENTRY,
            ModelOperation::Indices,
        );
        assert!(result.diagnostics.is_empty());
        assert_eq!(result.outputs[0].shape, expected);
        assert_eq!(
            result.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real
            })
        );
    }
}

#[test]
fn similarity_distinguishes_numeric_rows_and_opaque_document_cardinality() {
    let entry = &super::super::cosine_similarity::COSINE_SIMILARITY_CATALOG_ENTRY;
    for (domain, storage) in [
        (NumericDomain::Real, StorageFact::Sparse),
        (NumericDomain::Complex, StorageFact::Dense),
    ] {
        let result = infer(
            &request(vec![numeric(
                domain,
                ShapeFact::from(vec![Some(3), Some(4)]),
                StorageFact::Dense,
            )]),
            entry,
            ModelOperation::Similarity,
        );
        assert_eq!(
            result.outputs[0].shape,
            ShapeFact::from(vec![Some(3), Some(3)])
        );
        assert_eq!(result.outputs[0].storage, storage);
    }
    let result = infer(
        &request(vec![object(Some(standard::TOKENIZED_DOCUMENT))]),
        entry,
        ModelOperation::Similarity,
    );
    assert_eq!(result.outputs[0].shape, ShapeFact::from(vec![None, None]));
}
