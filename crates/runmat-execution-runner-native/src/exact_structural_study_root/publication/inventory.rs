use super::*;

pub(super) fn reference_inventory(
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

pub(super) fn response_objects(
    response: &ProgramExecutionResponse,
) -> NativeExecutionResult<Vec<ValueRef>> {
    let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = response else {
        return Err(protocol(
            "study publication contains a non-successful analysis closure",
        ));
    };
    Ok(result_objects.clone())
}

pub(super) fn validate_externalized_response(
    response: &ProgramExecutionResponse,
    inventory: &[ValueRef],
) -> NativeExecutionResult<()> {
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        return Err(protocol(
            "study publication contains a non-successful analysis closure",
        ));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(protocol("analysis closure has an invalid root"));
    };
    if !result_objects.contains(root)
        || result_objects
            .iter()
            .any(|reference| !inventory.contains(reference))
    {
        return Err(protocol(
            "analysis closure differs from its declared object inventory",
        ));
    }
    reference_inventory(result_objects)?;
    Ok(())
}

pub(super) fn unique_references(
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
    use runmat_execution::schema::VALUE_PAYLOAD_SCHEMA_VERSION;

    use super::*;

    fn reference(identity: &[u8]) -> ValueRef {
        ValueRef {
            schema_version: VALUE_PAYLOAD_SCHEMA_VERSION,
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
