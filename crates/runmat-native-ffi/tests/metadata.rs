use runmat_native_ffi::{
    artifact_identity, normalize_metadata, validate_metadata, NativeInterfaceArtifactManifest,
    NativeLibrary, NativeLibraryMetadata, NativeScalar, NativeType, Parameter, ParameterDirection,
    PointerOwnership, StructureDefinition, StructureField, SymbolPrototype,
    NATIVE_FFI_METADATA_SCHEMA_VERSION,
};

fn metadata() -> NativeLibraryMetadata {
    NativeLibraryMetadata {
        schema_version: NATIVE_FFI_METADATA_SCHEMA_VERSION,
        target_triple: "x86_64-unknown-linux-gnu".into(),
        source_digest: "01".repeat(32),
        libraries: vec![NativeLibrary {
            name: "fixture".into(),
            path: "libfixture.so".into(),
            dependencies: vec![],
            symbols: vec![SymbolPrototype {
                name: "fixture_scale".into(),
                exported_name: "fixture_scale".into(),
                calling_convention: Default::default(),
                return_type: NativeType::Scalar {
                    scalar: NativeScalar::F64,
                },
                return_ownership: None,
                return_nullable: false,
                parameters: vec![Parameter {
                    name: "record".into(),
                    ty: NativeType::Pointer {
                        pointee: Box::new(NativeType::Structure {
                            name: "fixture_record".into(),
                        }),
                        mutability: runmat_native_ffi::PointerMutability::Mutable,
                    },
                    direction: ParameterDirection::InputOutput,
                    ownership: PointerOwnership::Borrowed,
                    nullable: false,
                }],
                variadic: false,
            }],
        }],
        structures: vec![StructureDefinition {
            name: "fixture_record".into(),
            fields: vec![
                StructureField {
                    name: "value".into(),
                    ty: NativeType::Scalar {
                        scalar: NativeScalar::F64,
                    },
                },
                StructureField {
                    name: "tag".into(),
                    ty: NativeType::Scalar {
                        scalar: NativeScalar::U32,
                    },
                },
            ],
        }],
        enumerations: vec![],
        aliases: vec![],
    }
}

#[test]
fn normalization_and_identity_are_deterministic() {
    let metadata = normalize_metadata(metadata());
    validate_metadata(&metadata).expect("valid metadata");
    assert_eq!(
        artifact_identity(&metadata).expect("identity"),
        artifact_identity(&metadata).expect("identity")
    );
}

#[test]
fn structure_field_order_remains_abi_significant() {
    let original = normalize_metadata(metadata());
    let mut reordered = original.clone();
    reordered.structures[0].fields.reverse();
    validate_metadata(&reordered).expect("field order is not alphabetical metadata");
    assert_ne!(
        artifact_identity(&original).expect("original identity"),
        artifact_identity(&reordered).expect("reordered identity")
    );
}

#[test]
fn outputs_must_be_pointer_typed() {
    let mut invalid = normalize_metadata(metadata());
    invalid.libraries[0].symbols[0].parameters[0].ty = NativeType::Scalar {
        scalar: NativeScalar::F64,
    };
    let error = validate_metadata(&invalid).expect_err("invalid output");
    assert!(error
        .to_string()
        .contains("output parameters must have pointer type"));
}

#[test]
fn pointer_returns_cannot_claim_ownership_without_a_release_contract() {
    let mut invalid = normalize_metadata(metadata());
    let symbol = &mut invalid.libraries[0].symbols[0];
    symbol.return_type = NativeType::Pointer {
        pointee: Box::new(NativeType::Scalar {
            scalar: NativeScalar::F64,
        }),
        mutability: runmat_native_ffi::PointerMutability::Mutable,
    };
    symbol.return_ownership = Some(PointerOwnership::LibraryOwned);
    symbol.return_nullable = true;
    let error = validate_metadata(&invalid).expect_err("missing release contract");
    assert!(error
        .to_string()
        .contains("require a matching release contract"));
}

#[test]
fn prepared_artifact_identity_binds_content_not_physical_location() {
    let mut first_metadata = normalize_metadata(metadata());
    first_metadata.target_triple = target_lexicon::HOST.to_string();
    let mut relocated_metadata = first_metadata.clone();
    relocated_metadata.libraries[0].path = "/another/materialization/libfixture.so".into();

    let first = NativeInterfaceArtifactManifest::from_library(
        "fixture",
        first_metadata,
        b"synthetic library bytes",
    )
    .expect("first manifest");
    let relocated = NativeInterfaceArtifactManifest::from_library(
        "fixture",
        relocated_metadata,
        b"synthetic library bytes",
    )
    .expect("relocated manifest");

    assert_eq!(first.identity, relocated.identity);
    assert_eq!(first.metadata, relocated.metadata);
    assert_eq!(
        NativeInterfaceArtifactManifest::from_canonical_bytes(&first.canonical_bytes().unwrap())
            .unwrap(),
        first
    );
    first
        .validate_current_library(b"synthetic library bytes")
        .unwrap();
    assert!(first.validate_library(b"changed library bytes").is_err());
    first.interop_manifest().validate().unwrap();

    let directory = tempfile::tempdir().unwrap();
    let manifest_path = directory.path().join("fixture.runmat.json");
    first.publish(&manifest_path).unwrap();
    assert_eq!(
        NativeInterfaceArtifactManifest::read(&manifest_path).unwrap(),
        first
    );

    let materialized = first
        .materialized_metadata(std::path::Path::new("/materialized/libfixture.so"))
        .unwrap();
    assert_eq!(
        materialized.libraries[0].path,
        "/materialized/libfixture.so"
    );
}
