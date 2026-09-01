use super::*;
use crate::{BuiltinAsyncBehavior, BuiltinCompatibility, BuiltinPurity, BuiltinSemanticKind};
use runmat_types::{CapabilityRequirement, ClassIdentity, EffectKind};

const OUTPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Result.",
}];
const SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "Y = pilot()",
    inputs: &[],
    outputs: &OUTPUTS,
}];
const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &[],
};
const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    summary: "Pilot builtin.",
    keywords: &["pilot"],
    related: &[],
    introduced: None,
    status: None,
    examples: &[],
    ..BuiltinDocumentation::EMPTY
};
const PLACEMENT: BuiltinPlacementContract = BuiltinPlacementContract {
    portability: BuiltinPortability::NativeAndWasm,
    accelerator: BuiltinAcceleratorPolicy::Optional,
    residency: BuiltinResidencyPolicy::PreserveInputs,
    fusion: BuiltinFusionPolicy::Candidate,
    distributed: crate::BuiltinDistributedPolicy::Unsupported,
};
const LINK: BuiltinLinkContract = BuiltinLinkContract {
    reachability: BuiltinReachability::Always,
    policy: BuiltinLinkPolicy::PortableRuntime,
    execution_stack: runmat_types::ExecutionStackRequirement::Any,
    artifact_dependencies: &[],
};
const EFFECTS: [EffectKind; 1] = [EffectKind::MayThrow];
const CAPABILITIES: [CapabilityRequirement; 1] = [CapabilityRequirement::HostRuntime];

const PILOT_ID: BuiltinCatalogIdentity = BuiltinCatalogIdentity { name: "pilot" };
const PILOT_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    variant: "default",
    availability: BuiltinBindingAvailability::Required,
}];
const PILOT: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: PILOT_ID,
    category: "test",
    documentation: DOCUMENTATION,
    descriptor: &DESCRIPTOR,
    contract: BuiltinContractDeclaration {
        maturity: BuiltinContractMaturity::Complete,
        inference_rule: BuiltinInferenceRule::Array(ArrayInferenceRule::Full),
        compatibility: BuiltinCompatibility::Matlab,
        async_behavior: BuiltinAsyncBehavior::NeverSuspends,
        purity: BuiltinPurity::Pure,
        semantic_kind: BuiltinSemanticKind::General,
        workspace_effect: None,
        environment_effect: None,
        effects: &EFFECTS,
        capabilities: &CAPABILITIES,
    },
    placement: PLACEMENT,
    link: LINK,
    bindings: &PILOT_BINDINGS,
    extensions: &[],
    integer_capabilities: &[],
    integer_audit: None,
    suppress_auto_output: false,
};

const SECOND_ID: BuiltinCatalogIdentity = BuiltinCatalogIdentity { name: "second" };
const SECOND_BINDINGS: [BuiltinBindingDeclaration; 1] = [BuiltinBindingDeclaration {
    variant: "default",
    availability: BuiltinBindingAvailability::Required,
}];
const SECOND: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: SECOND_ID,
    bindings: &SECOND_BINDINGS,
    ..PILOT
};

const INCOMPLETE_CANONICAL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    summary: "Incomplete canonical documentation.",
    ..BuiltinDocumentation::EMPTY
};
const INCOMPLETE_CANONICAL: BuiltinCatalogEntry = BuiltinCatalogEntry {
    identity: BuiltinCatalogIdentity {
        name: "incompleteCanonical",
    },
    documentation: INCOMPLETE_CANONICAL_DOCUMENTATION,
    ..PILOT
};

#[test]
fn valid_catalog_has_stable_order_independent_fingerprint() {
    assert!(validate_builtin_catalog(&[&PILOT, &SECOND]).is_empty());
    assert_eq!(
        canonical_catalog_fingerprint(&[&PILOT, &SECOND]).unwrap(),
        canonical_catalog_fingerprint(&[&SECOND, &PILOT]).unwrap()
    );
}

#[test]
fn validation_rejects_duplicate_catalog_and_binding_identities() {
    let errors = validate_builtin_catalog(&[&PILOT, &PILOT]);
    assert!(errors
        .iter()
        .any(|error| error.message == "duplicate builtin catalog identity"));
    assert!(errors
        .iter()
        .any(|error| error.message == "duplicate builtin binding identity"));
}

#[test]
fn validation_rejects_incomplete_canonical_documentation() {
    let errors = validate_builtin_catalog(&[&INCOMPLETE_CANONICAL]);
    assert!(errors
        .iter()
        .any(|error| error.message == "canonical documentation description must not be empty"));
    assert!(errors
        .iter()
        .any(|error| error.message == "canonical documentation keywords must not be empty"));
    assert!(errors.iter().any(|error| {
        error.message == "canonical documentation requires examples or one explicit exemption"
    }));
}

#[test]
fn root_documentation_is_catalog_owned_and_executable() {
    for (name, expected_examples, expected_faqs) in [("sqrt", 6, 7), ("realsqrt", 5, 5)] {
        let entry = builtin_catalog_entry_by_name(name).expect("root catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), expected_examples);
        assert_eq!(documentation.faqs.len(), expected_faqs);
        assert!(documentation
            .sections
            .iter()
            .any(|section| section.heading == "GPU execution"));
        assert!(!documentation.evidence.implementation.is_empty());
        assert!(!documentation.evidence.verification.is_empty());
        assert!(documentation.examples.iter().all(|example| {
            example.compatibility == BuiltinExampleCompatibility::RunMat && !example.id.is_empty()
        }));
    }
}

#[test]
fn numeric_limit_documentation_is_catalog_owned_and_executable() {
    for name in ["intmin", "intmax", "realmin", "realmax", "flintmax"] {
        let entry = builtin_catalog_entry_by_name(name).expect("numeric-limit catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), 3, "{name}");
        assert_eq!(documentation.faqs.len(), 2, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            example.compatibility == BuiltinExampleCompatibility::Matlab
                && example.harness == BuiltinExampleHarness::Portable
                && !example.id.is_empty()
        }));
    }
}

#[test]
fn exponential_documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [("exp", 6, 6), ("expm1", 5, 7)] {
        let entry = builtin_catalog_entry_by_name(name).expect("exponential catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn logarithm_documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [
        ("log", 6, 6),
        ("log10", 5, 6),
        ("log1p", 5, 11),
        ("log2", 7, 6),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("logarithm catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn complex_component_documentation_is_catalog_owned_and_executable() {
    for (name, example_count) in [("conj", 6), ("real", 6), ("imag", 5)] {
        let entry = builtin_catalog_entry_by_name(name).expect("component catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), 7, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation
            .examples
            .iter()
            .all(|example| !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )));
    }
}

#[test]
fn magnitude_phase_sign_documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [("abs", 7, 8), ("angle", 4, 7), ("sign", 7, 8)] {
        let entry = builtin_catalog_entry_by_name(name).expect("magnitude/phase/sign entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn floating_conversion_documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [("double", 8, 9), ("single", 6, 10)] {
        let entry = builtin_catalog_entry_by_name(name).expect("floating conversion entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn integer_conversion_documentation_is_catalog_owned_and_executable() {
    for name in [
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("integer conversion entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), 3, "{name}");
        assert_eq!(documentation.faqs.len(), 6, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn migrated_registry_is_valid_and_case_insensitive() {
    let errors = validate_builtin_catalog(builtin_catalog_entries());
    assert!(errors.is_empty(), "catalog errors: {errors:#?}");
    assert_eq!(
        builtin_catalog_entry_by_name("FULL")
            .expect("full catalog entry")
            .contract
            .inference_rule,
        BuiltinInferenceRule::Array(ArrayInferenceRule::Full)
    );
}

#[test]
fn catalog_entries_keep_domain_modules_out_of_the_root() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/catalog/entries");
    let root_files = std::fs::read_dir(&root)
        .expect("catalog entries directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    assert_eq!(
        root_files,
        vec![root.join("mod.rs")],
        "builtin contract files belong in mirrored domain directories"
    );
}

#[test]
fn catalog_entry_families_own_registration_without_domain_builtin_lists() {
    fn declares_catalog_entry(source: &str) -> bool {
        source.contains("_CATALOG_ENTRY:") || source.contains("CATALOG_ENTRY: BuiltinCatalogEntry")
    }

    fn visit(directory: &std::path::Path) {
        for entry in std::fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        {
            let path = entry.expect("catalog entry").path();
            if path.is_dir() {
                visit(&path);
                continue;
            }
            if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            if path.file_name().and_then(|name| name.to_str()) == Some("mod.rs") {
                let defines_family_entries = source
                    .lines()
                    .any(|line| line.starts_with("pub const ") && line.contains("_CATALOG_ENTRY:"));
                let composes_local_contracts =
                    std::fs::read_dir(path.parent().expect("catalog module directory"))
                        .expect("catalog module directory")
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|candidate| candidate != &path)
                        .filter(|candidate| {
                            candidate
                                .extension()
                                .and_then(|extension| extension.to_str())
                                == Some("rs")
                        })
                        .any(|candidate| {
                            std::fs::read_to_string(candidate)
                                .is_ok_and(|child| declares_catalog_entry(&child))
                        });
                if !defines_family_entries && !composes_local_contracts {
                    assert!(
                        !source.lines().any(|line| {
                            line.trim_start().starts_with('&') && line.contains("_CATALOG_ENTRY")
                        }),
                        "domain modules compose family slices, not per-builtin lists: {}",
                        path.display()
                    );
                }
            } else if source.contains("_CATALOG_ENTRY") && !source.contains("const ENTRIES") {
                let family_module_path = path.parent().expect("family directory").join("mod.rs");
                let family_module =
                    std::fs::read_to_string(&family_module_path).unwrap_or_else(|error| {
                        panic!("read {}: {error}", family_module_path.display())
                    });
                assert!(
                    family_module.contains("const ENTRIES"),
                    "grouped catalog family must own one local entry slice: {}",
                    family_module_path.display()
                );
                for declaration in source.lines().filter_map(|line| {
                    line.strip_prefix("pub const ")
                        .and_then(|rest| rest.split_once("_CATALOG_ENTRY:"))
                        .map(|(prefix, _)| format!("&{prefix}_CATALOG_ENTRY"))
                }) {
                    assert!(
                        family_module.contains(&declaration),
                        "{} does not register {declaration} from {}",
                        family_module_path.display(),
                        path.display()
                    );
                }
                for alias in source.lines().filter_map(|line| {
                    line.trim()
                        .split_once(" as ")
                        .map(|(_, alias)| alias.trim_end_matches(','))
                        .filter(|alias| alias.ends_with("_CATALOG_ENTRY"))
                }) {
                    assert!(
                        family_module.contains(&format!("&{alias}")),
                        "{} does not register aliased entry {alias} from {}",
                        family_module_path.display(),
                        path.display()
                    );
                }
            }
        }
    }

    visit(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/catalog/entries"));
}

#[test]
fn catalog_entry_modules_remain_focused() {
    const MAX_LINES: usize = 500;

    fn visit(directory: &std::path::Path) {
        for entry in std::fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()))
        {
            let path = entry.expect("catalog entry").path();
            if path.is_dir() {
                visit(&path);
                continue;
            }
            if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
                continue;
            }

            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            let line_count = source.lines().count();
            assert!(
                line_count <= MAX_LINES,
                "catalog contract module has {line_count} lines; split it by builtin domain before it exceeds {MAX_LINES}: {}",
                path.display()
            );
        }
    }

    visit(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/catalog/entries"));
}

#[test]
fn distributed_execution_policy_is_declared_by_the_canonical_catalog() {
    assert_eq!(
        builtin_catalog_entry_by_name("abs")
            .expect("abs catalog entry")
            .placement
            .distributed,
        BuiltinDistributedPolicy::MapUnary
    );
    assert_eq!(
        builtin_catalog_entry_by_name("gather")
            .expect("gather catalog entry")
            .placement
            .distributed,
        BuiltinDistributedPolicy::MaterializeArguments
    );
    assert_eq!(
        builtin_catalog_entry_by_name("full")
            .expect("full catalog entry")
            .placement
            .distributed,
        BuiltinDistributedPolicy::Unsupported
    );
}

#[test]
fn gpu_array_catalog_owns_its_accelerator_contract() {
    let entry = builtin_catalog_entry_by_name("gpuArray").expect("gpuArray catalog entry");

    assert_eq!(
        entry.contract.capability_set().0,
        [CapabilityRequirement::Accelerator].into_iter().collect()
    );
    assert_eq!(
        entry.placement.accelerator,
        BuiltinAcceleratorPolicy::Required
    );
    assert_eq!(
        entry.placement.residency,
        BuiltinResidencyPolicy::ProduceResident
    );
    assert_eq!(entry.descriptor.signatures.len(), 5);
    assert_eq!(entry.descriptor.errors.len(), 15);
}

#[test]
fn acceleration_documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [("gpuArray", 7, 7), ("gather", 7, 7)] {
        let entry = builtin_catalog_entry_by_name(name).expect("acceleration catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn array_creation_documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [("full", 5, 6), ("zeros", 8, 10)] {
        let entry = builtin_catalog_entry_by_name(name).expect("array creation catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn aggregate_and_dynamic_dispatch_documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [("struct", 6, 6), ("feval", 6, 6)] {
        let entry = builtin_catalog_entry_by_name(name).expect("catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "GPU execution"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn parallel_pool_and_future_documentation_is_catalog_owned_and_executable() {
    for (name, example_count, faq_count) in [
        ("parpool", 2, 4),
        ("gcp", 2, 4),
        ("parfeval", 2, 4),
        ("parfevalOnAll", 1, 4),
        ("fetchOutputs", 2, 4),
        ("fetchNext", 1, 4),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("parallel catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), faq_count, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "Execution placement"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && example.harness == BuiltinExampleHarness::Native
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn parallel_context_documentation_is_catalog_owned_and_executable() {
    for (name, example_count) in [
        ("spmdIndex", 2),
        ("spmdSize", 2),
        ("labindex", 1),
        ("numlabs", 1),
        ("getCurrentTask", 2),
        ("getCurrentWorker", 2),
        ("getCurrentJob", 1),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("parallel context catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), 4, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "Execution placement"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && example.harness == BuiltinExampleHarness::Native
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn codistributor_documentation_is_catalog_owned_and_executable() {
    for name in [
        "codistributor",
        "codistributor1d",
        "codistributor2dbc",
        "isComplete",
        "iscodistributed",
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("codistributor catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), 2, "{name}");
        assert_eq!(documentation.faqs.len(), 4, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "Execution placement"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && example.harness == BuiltinExampleHarness::Native
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn distributed_array_documentation_is_catalog_owned_and_executable() {
    for (name, example_count) in [
        ("distributed", 2),
        ("codistributed", 2),
        ("codistributed.build", 1),
        ("redistribute", 2),
        ("getCodistributor", 1),
        ("globalIndices", 2),
        ("getLocalPart", 2),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("distributed array catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), example_count, "{name}");
        assert_eq!(documentation.faqs.len(), 5, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "Execution placement"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && example.harness == BuiltinExampleHarness::Native
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn parallel_collective_documentation_is_catalog_owned_and_executable() {
    for name in [
        "labBarrier",
        "labBroadcast",
        "labSend",
        "labReceive",
        "labProbe",
        "labSendReceive",
        "gplus",
        "gcat",
        "gop",
        "spmdBarrier",
        "spmdBroadcast",
        "spmdSend",
        "spmdReceive",
        "spmdProbe",
        "spmdSendReceive",
        "spmdPlus",
        "spmdCat",
        "spmdReduce",
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("parallel collective catalog entry");
        let documentation = &entry.documentation;
        assert_eq!(
            documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(documentation.examples.len(), 1, "{name}");
        assert_eq!(documentation.faqs.len(), 5, "{name}");
        assert!(
            documentation
                .sections
                .iter()
                .any(|section| section.heading == "Failures and cancellation"),
            "{name}"
        );
        assert!(!documentation.evidence.implementation.is_empty(), "{name}");
        assert!(!documentation.evidence.verification.is_empty(), "{name}");
        assert!(documentation.examples.iter().all(|example| {
            !example.id.is_empty()
                && example.harness == BuiltinExampleHarness::Native
                && matches!(
                    example.verification,
                    BuiltinExampleVerification::Assertions { .. }
                )
        }));
    }
}

#[test]
fn gpu_array_inference_preserves_or_converts_typed_facts_before_device_placement() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("gpuArray").expect("gpuArray catalog entry");
    let mut source = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), Some(1)]),
        StorageFact::Dense,
    );
    source.residency = runmat_types::ResidencyFact::Host;

    let preserve = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![source.clone()],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(preserve.outputs[0].kind, source.kind);
    assert_eq!(preserve.outputs[0].shape, source.shape);
    assert_eq!(
        preserve.outputs[0].residency,
        runmat_types::ResidencyFact::Device { provider: None }
    );

    let convert = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![source.clone(), ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("int32".into()),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(
        convert.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Real,
        })
    );

    let reshape = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![
                source,
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(2.0),
                LiteralValue::Number(2.0),
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(
        reshape.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(2)])
    );
}

#[test]
fn distributed_call_inference_preserves_the_execution_owned_value_contract() {
    use runmat_types::{
        CallRequest, DistributedFact, DistributedOwner, DistributedValueId, DistributionScheme,
        LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        ProgramFunctionId, RequestedOutputCount, ValueFact, ValueKindFact,
    };

    let local = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Int16,
        domain: NumericDomain::Complex,
    }));
    let distributed = DistributedFact {
        id: DistributedValueId {
            function: ProgramFunctionId(7),
            ordinal: 3,
        },
        owner: DistributedOwner::Client(ProgramFunctionId(7)),
        scheme: Some(DistributionScheme::Replicated),
        value: Box::new(local),
        materializable: true,
    };
    let request = CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::Distributed(
            distributed.clone(),
        ))],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };

    let mapped = infer_catalog_call(
        builtin_catalog_entry_by_name("abs").expect("abs entry"),
        &request,
    );
    assert!(mapped.diagnostics.is_empty(), "{:#?}", mapped.diagnostics);
    let ValueKindFact::Distributed(mapped) = &mapped.outputs[0].kind else {
        panic!("partition-local mapping must retain a distributed result fact");
    };
    assert_eq!(mapped.id, distributed.id);
    assert_eq!(mapped.owner, distributed.owner);
    assert_eq!(mapped.scheme, distributed.scheme);
    assert_eq!(mapped.materializable, distributed.materializable);
    assert_eq!(
        mapped.value.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int16,
            domain: NumericDomain::Real,
        })
    );

    let gathered = infer_catalog_call(
        builtin_catalog_entry_by_name("gather").expect("gather entry"),
        &request,
    );
    assert!(
        gathered.diagnostics.is_empty(),
        "{:#?}",
        gathered.diagnostics
    );
    assert!(matches!(
        gathered.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int16,
            domain: NumericDomain::Complex,
        })
    ));
}

#[test]
fn numeric_limit_contracts_infer_class_representation_and_distributed_like_outputs() {
    use runmat_types::{
        CallRequest, DistributedFact, DistributedOwner, DistributedValueId, DistributionScheme,
        LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact, OutputSelection,
        ProgramFunctionId, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    let class_request = |class: &str| CallRequest {
        arguments: vec![ValueFact::scalar(ValueKindFact::String)],
        literals: LiteralContext::new(vec![LiteralValue::String(class.into())]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    for (builtin, class) in [
        ("intmin", NumericClass::UInt64),
        ("intmax", NumericClass::UInt64),
        ("realmin", NumericClass::Single),
        ("realmax", NumericClass::Single),
        ("flintmax", NumericClass::Single),
    ] {
        let source_name = if class == NumericClass::UInt64 {
            "uint64"
        } else {
            "single"
        };
        let inference = infer_catalog_call(
            builtin_catalog_entry_by_name(builtin).expect("numeric limit entry"),
            &class_request(source_name),
        );
        assert!(inference.diagnostics.is_empty(), "{builtin}");
        assert_eq!(
            inference.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(inference.outputs[0].shape, ShapeFact::Scalar);
    }

    let mut local = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(8), Some(2)]),
        StorageFact::Dense,
    );
    local.residency = ResidencyFact::Host;
    let distributed = DistributedFact {
        id: DistributedValueId {
            function: ProgramFunctionId(11),
            ordinal: 2,
        },
        owner: DistributedOwner::Client(ProgramFunctionId(11)),
        scheme: Some(DistributionScheme::Replicated),
        value: Box::new(local),
        materializable: true,
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("realmax").expect("realmax entry"),
        &CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Distributed(distributed.clone())),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Keyword("like".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(
        inference.diagnostics.is_empty(),
        "{:#?}",
        inference.diagnostics
    );
    let ValueKindFact::Distributed(output) = &inference.outputs[0].kind else {
        panic!("numeric limit like form must retain distributed placement");
    };
    assert_eq!(output.id, distributed.id);
    assert_eq!(output.scheme, distributed.scheme);
    assert_eq!(output.value.shape, ShapeFact::Scalar);
    assert_eq!(
        output.value.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
}

#[test]
fn current_parallel_context_contracts_are_typed_and_parallel_capable() {
    for name in ["getCurrentTask", "getCurrentWorker", "getCurrentJob"] {
        let entry = builtin_catalog_entry_by_name(name).expect("parallel context catalog entry");
        let effects = entry.contract.effect_set();
        let capabilities = entry.contract.capability_set();

        assert_eq!(
            effects.0,
            [EffectKind::MayThrow].into_iter().collect(),
            "{name} must not acquire an unknown effect"
        );
        assert_eq!(
            capabilities.0,
            [CapabilityRequirement::ParallelRuntime]
                .into_iter()
                .collect(),
            "{name} must retain its execution capability through analysis"
        );
        assert_eq!(
            entry.contract.maturity,
            BuiltinContractMaturity::DynamicByDesign
        );
    }
}

#[test]
fn parallel_surface_retains_pool_future_and_fetch_facts() {
    use runmat_types::{
        CallRequest, CallableFact, ExecutionFact, FutureStateFact, LiteralContext, LiteralValue,
        NumericClass, NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ValueFact,
        ValueKindFact,
    };

    let request = CallRequest {
        arguments: Vec::new(),
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let pool = infer_catalog_call(
        builtin_catalog_entry_by_name("parpool").expect("parpool entry"),
        &request,
    );
    assert!(matches!(
        pool.outputs[0].kind,
        ValueKindFact::Execution(ExecutionFact::Pool)
    ));
    assert_eq!(
        pool.capabilities.0,
        [CapabilityRequirement::ParallelRuntime]
            .into_iter()
            .collect()
    );

    let callable_output = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::UInt16,
        domain: NumericDomain::Real,
    }));
    let callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: vec![ValueFact::unknown(
            runmat_types::DynamicReason::RuntimeValue,
        )],
        parameters_complete: true,
        outputs: vec![callable_output.clone()],
        outputs_complete: true,
        variadic_inputs: false,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: true,
    }));
    let future = infer_catalog_call(
        builtin_catalog_entry_by_name("parfeval").expect("parfeval entry"),
        &CallRequest {
            arguments: vec![
                callable,
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::Number(1.0),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(matches!(
        &future.outputs[0].kind,
        ValueKindFact::Execution(ExecutionFact::Future {
            output,
            state: FutureStateFact::Unknown,
        }) if **output == callable_output
    ));

    let fetched = infer_catalog_call(
        builtin_catalog_entry_by_name("fetchOutputs").expect("fetchOutputs entry"),
        &CallRequest {
            arguments: future.outputs,
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(fetched.outputs, vec![callable_output]);
    assert!(fetched.effects.0.contains(&EffectKind::MaySuspend));
    assert!(fetched.effects.0.contains(&EffectKind::MayThrow));
}

#[test]
fn codistributor_constructors_have_static_value_class_facts() {
    use runmat_types::{
        CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ValueKindFact,
    };

    let request = CallRequest {
        arguments: Vec::new(),
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    for (builtin, expected_class) in [
        ("codistributor1d", "codistributor1d"),
        ("codistributor2dbc", "codistributor2dbc"),
    ] {
        let inference = infer_catalog_call(
            builtin_catalog_entry_by_name(builtin).expect("codistributor catalog entry"),
            &request,
        );
        assert!(inference.diagnostics.is_empty(), "{builtin}");
        let ValueKindFact::Object(object) = &inference.outputs[0].kind else {
            panic!("{builtin} must produce an object fact");
        };
        assert_eq!(
            object
                .runtime_class
                .as_ref()
                .map(ClassIdentity::display_name),
            Some(expected_class)
        );
        assert_eq!(object.handle_semantics, Some(false));
    }
}

#[test]
fn zeros_contract_uses_literal_dimensions_class_and_like_residency() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    let dimension = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    let request = CallRequest {
        arguments: vec![dimension.clone(), dimension],
        literals: LiteralContext::new(vec![LiteralValue::Number(2.0), LiteralValue::Number(3.0)]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("zeros").expect("zeros entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(
        inference.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(3)])
    );
    assert_eq!(inference.outputs[0].storage, StorageFact::Dense);

    let mut prototype = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(4), Some(5)]),
        StorageFact::Dense,
    );
    prototype.residency = ResidencyFact::Device {
        provider: Some("pilot".into()),
    };
    let like = ValueFact::scalar(ValueKindFact::String);
    let request = CallRequest {
        arguments: vec![like, prototype.clone()],
        literals: LiteralContext::new(vec![
            LiteralValue::Keyword("like".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("zeros").expect("zeros entry"),
        &request,
    );
    assert_eq!(inference.outputs[0].kind, prototype.kind);
    assert_eq!(inference.outputs[0].shape, prototype.shape);
    assert_eq!(inference.outputs[0].residency, prototype.residency);
}

#[test]
fn full_contract_densifies_sparse_facts_without_losing_class_shape_or_residency() {
    use runmat_types::{
        CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt32,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Sparse,
    );
    input.residency = runmat_types::ResidencyFact::Host;
    let request = CallRequest {
        arguments: vec![input.clone()],
        literals: runmat_types::LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("full").expect("full entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(inference.outputs[0].kind, input.kind);
    assert_eq!(inference.outputs[0].shape, input.shape);
    assert_eq!(inference.outputs[0].residency, input.residency);
    assert_eq!(inference.outputs[0].storage, StorageFact::Dense);
}

#[test]
fn abs_contract_preserves_class_shape_storage_and_residency_but_makes_complex_real() {
    use runmat_types::{
        CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };
    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Sparse,
    );
    input.residency = runmat_types::ResidencyFact::Device {
        provider: Some("pilot".into()),
    };
    let request = CallRequest {
        arguments: vec![input.clone()],
        literals: runmat_types::LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("abs").expect("abs entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(
        inference.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int64,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(inference.outputs[0].shape, input.shape);
    assert_eq!(inference.outputs[0].storage, input.storage);
    assert_eq!(inference.outputs[0].residency, input.residency);
}

#[test]
fn radian_trigonometric_leaves_own_complete_contracts_documentation_and_examples() {
    for (name, function, example_count) in [
        ("sin", TrigonometricFunction::Sin, 5),
        ("cos", TrigonometricFunction::Cos, 6),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("trigonometric catalog entry");
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::Trigonometric(function))
        );
        assert_eq!(
            entry.documentation.authority,
            BuiltinDocumentationAuthority::Catalog
        );
        assert_eq!(entry.documentation.examples.len(), example_count);
        assert_eq!(entry.documentation.faqs.len(), 8);
        assert!(entry
            .documentation
            .examples
            .iter()
            .all(|example| !example.id.is_empty() && !example.program.is_empty()));
        assert_eq!(entry.bindings, REQUIRED_DEFAULT_BINDING.as_slice());
        assert_eq!(entry.extensions.len(), 4);
        assert_eq!(entry.integer_capabilities.len(), 1);
    }
}

#[test]
fn radian_trigonometric_contracts_preserve_floating_class_and_apply_like_representation() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    for name in ["sin", "cos"] {
        let entry = builtin_catalog_entry_by_name(name).expect("trigonometric catalog entry");
        let mut input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        input.residency = ResidencyFact::Device {
            provider: Some("source-provider".into()),
        };
        let inferred = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![input.clone()],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(inferred.diagnostics.is_empty());
        assert_eq!(inferred.outputs[0].kind, input.kind);
        assert_eq!(inferred.outputs[0].shape, input.shape);
        assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
        assert_eq!(inferred.outputs[0].residency, ResidencyFact::Unknown);

        let mut prototype = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        }));
        prototype.residency = ResidencyFact::Device {
            provider: Some("prototype-provider".into()),
        };
        let like = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![
                    input,
                    ValueFact::scalar(ValueKindFact::String),
                    prototype.clone(),
                ],
                literals: LiteralContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::String("like".into()),
                    LiteralValue::Unknown,
                ]),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(like.diagnostics.is_empty());
        assert_eq!(
            like.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            })
        );
        assert_eq!(like.outputs[0].residency, prototype.residency);
    }
}

#[test]
fn radian_trigonometric_contracts_promote_extensions_and_reject_unsupported_representations() {
    use runmat_types::{
        CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    for name in ["sin", "cos"] {
        let entry = builtin_catalog_entry_by_name(name).expect("trigonometric catalog entry");
        let infer = |input| {
            infer_catalog_call(
                entry,
                &CallRequest {
                    arguments: vec![input],
                    literals: LiteralContext::default(),
                    outputs: OutputSelection::new(RequestedOutputCount::One),
                },
            )
        };
        let integer = infer(ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt16,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(4), Some(1)]),
            StorageFact::Dense,
        ));
        assert_eq!(
            integer.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(
            integer.outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(1)])
        );

        for kind in [ValueKindFact::Logical, ValueKindFact::Character] {
            let extension = infer(ValueFact::scalar(kind));
            assert!(extension.diagnostics.is_empty());
            assert_eq!(
                extension.outputs[0].kind,
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })
            );
        }

        let sparse = infer(ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(3), Some(3)]),
            StorageFact::Sparse,
        ));
        assert!(sparse
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-TRIGONOMETRIC-SPARSE"));

        let complex_integer = infer(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Complex,
        })));
        assert!(complex_integer
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-TRIGONOMETRIC-COMPLEX-INTEGER"));
    }
}

#[test]
fn exp_contract_preserves_floating_facts_and_marks_conversion_residency_dynamic() {
    use runmat_types::{
        AliasFact, CallRequest, ContiguityFact, LayoutFact, NumericClass, NumericDomain,
        NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact,
        ValueFact, ValueKindFact, ViewFact,
    };

    let entry = builtin_catalog_entry_by_name("exp").expect("exp catalog entry");
    let shape = ShapeFact::from(vec![Some(3), Some(2)]);
    let mut complex_single = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        shape.clone(),
        StorageFact::Dense,
    );
    complex_single.residency = ResidencyFact::Device {
        provider: Some("wgpu".into()),
    };
    let preserve = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![complex_single],
            literals: runmat_types::LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(preserve.diagnostics.is_empty());
    assert_eq!(
        preserve.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(preserve.outputs[0].shape, shape);
    assert_eq!(
        preserve.outputs[0].residency,
        ResidencyFact::Device {
            provider: Some("wgpu".into())
        }
    );
    assert_eq!(preserve.outputs[0].layout, LayoutFact::ColumnMajor);
    assert_eq!(preserve.outputs[0].contiguity, ContiguityFact::Contiguous);
    assert_eq!(preserve.outputs[0].view, ViewFact::Materialized);
    assert_eq!(preserve.outputs[0].alias, AliasFact::Unique);

    let mut integer = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        }),
        shape.clone(),
        StorageFact::Dense,
    );
    integer.residency = ResidencyFact::Device {
        provider: Some("integer-provider".into()),
    };
    let converted = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![integer],
            literals: runmat_types::LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(
        converted.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(converted.outputs[0].shape, shape);
    assert_eq!(converted.outputs[0].residency, ResidencyFact::Unknown);

    let sparse = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        shape,
        StorageFact::Sparse,
    );
    let densified = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![sparse],
            literals: runmat_types::LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(densified.outputs[0].storage, StorageFact::Dense);
    assert_eq!(densified.outputs[0].residency, ResidencyFact::Host);
}

#[test]
fn integer_conversion_contracts_preserve_shape_domain_storage_and_residency_with_typed_output() {
    use runmat_types::{
        AliasFact, CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
        ViewFact,
    };

    let mut source = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Sparse,
    );
    source.residency = ResidencyFact::Device {
        provider: Some("integer-provider".into()),
    };
    for (name, class) in [
        ("int8", NumericClass::Int8),
        ("int16", NumericClass::Int16),
        ("int32", NumericClass::Int32),
        ("int64", NumericClass::Int64),
        ("uint8", NumericClass::UInt8),
        ("uint16", NumericClass::UInt16),
        ("uint32", NumericClass::UInt32),
        ("uint64", NumericClass::UInt64),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("integer conversion catalog entry");
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::NumericConversion(class))
        );
        let converted = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![source.clone()],
                literals: runmat_types::LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );

        assert!(converted.diagnostics.is_empty(), "{name}");
        assert_eq!(
            converted.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            }),
            "{name}"
        );
        assert_eq!(converted.outputs[0].shape, source.shape, "{name}");
        assert_eq!(converted.outputs[0].storage, StorageFact::Sparse, "{name}");
        assert_eq!(converted.outputs[0].residency, source.residency, "{name}");
        assert_eq!(converted.outputs[0].view, ViewFact::Materialized, "{name}");
        assert_eq!(converted.outputs[0].alias, AliasFact::Unique, "{name}");
    }

    let entry = builtin_catalog_entry_by_name("uint16").expect("uint16 catalog entry");
    let invalid = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: runmat_types::LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(
        invalid.outputs[0].kind,
        ValueKindFact::Unknown,
        "unsupported input must not be assigned a numeric class"
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-NUMERIC-CONVERSION"));
}

#[test]
fn floating_conversion_contracts_type_class_shape_and_like_residency_separately() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    let mut source = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    source.residency = ResidencyFact::Device {
        provider: Some("source-provider".into()),
    };
    let mut prototype = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Single,
        domain: NumericDomain::Real,
    }));
    prototype.residency = ResidencyFact::Device {
        provider: Some("prototype-provider".into()),
    };

    for (name, class) in [
        ("double", NumericClass::Double),
        ("single", NumericClass::Single),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("floating conversion entry");
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::NumericConversionWithLike(class))
        );

        let default_conversion = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![source.clone()],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(default_conversion.diagnostics.is_empty(), "{name}");
        assert_eq!(
            default_conversion.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            }),
            "{name}"
        );
        assert_eq!(default_conversion.outputs[0].shape, source.shape, "{name}");
        assert_eq!(
            default_conversion.outputs[0].residency,
            ResidencyFact::Unknown,
            "provider capability decides default resident conversion"
        );

        let like_conversion = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![
                    source.clone(),
                    ValueFact::scalar(ValueKindFact::String),
                    prototype.clone(),
                ],
                literals: LiteralContext::new(vec![
                    LiteralValue::Unknown,
                    LiteralValue::String("like".into()),
                    LiteralValue::Unknown,
                ]),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(like_conversion.diagnostics.is_empty(), "{name}");
        assert_eq!(
            like_conversion.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class,
                domain: NumericDomain::Complex,
            }),
            "the prototype must not change {name}'s fixed result class"
        );
        assert_eq!(
            like_conversion.outputs[0].residency, prototype.residency,
            "{name}"
        );
    }

    let invalid = infer_catalog_call(
        builtin_catalog_entry_by_name("double").expect("double entry"),
        &CallRequest {
            arguments: vec![
                source,
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::String),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("like".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(invalid
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE"));

    let complex_prototype = infer_catalog_call(
        builtin_catalog_entry_by_name("single").expect("single entry"),
        &CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                })),
                ValueFact::scalar(ValueKindFact::String),
                ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Single,
                    domain: NumericDomain::Complex,
                })),
            ],
            literals: LiteralContext::new(vec![
                LiteralValue::Unknown,
                LiteralValue::String("like".into()),
                LiteralValue::Unknown,
            ]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(complex_prototype
        .diagnostics
        .iter()
        .any(|diagnostic| { diagnostic.code == "RM-CATALOG-NUMERIC-CONVERSION-PROTOTYPE" }));
}

#[test]
fn numeric_component_contracts_preserve_exact_facts_and_transform_only_the_component_domain() {
    use runmat_types::{
        AliasFact, CallRequest, ContiguityFact, LayoutFact, MutationFact, NumericClass,
        NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact,
        ShapeFact, StorageFact, ValueFact, ValueKindFact, ViewFact,
    };

    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt64,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3), Some(4)]),
        StorageFact::Dense,
    );
    input.residency = ResidencyFact::Device {
        provider: Some("component-provider".into()),
    };
    input.layout = LayoutFact::ColumnMajor;
    input.contiguity = ContiguityFact::Contiguous;
    input.view = ViewFact::ReadOnlyView;
    input.alias = AliasFact::Shared;
    input.mutation = MutationFact::Immutable;

    for (name, rule, expected_domain) in [
        (
            "conj",
            NumericComponentRule::Conjugate,
            NumericDomain::Complex,
        ),
        ("real", NumericComponentRule::RealPart, NumericDomain::Real),
        (
            "imag",
            NumericComponentRule::ImaginaryPart,
            NumericDomain::Real,
        ),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("numeric component entry");
        assert_eq!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::NumericComponent(rule))
        );
        let inference = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![input.clone()],
                literals: Default::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(inference.diagnostics.is_empty(), "{name}");
        let output = &inference.outputs[0];
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: expected_domain,
            }),
            "{name}"
        );
        assert_eq!(output.shape, input.shape, "{name}");
        assert_eq!(output.storage, input.storage, "{name}");
        assert_eq!(output.residency, input.residency, "{name}");
        assert_eq!(output.layout, input.layout, "{name}");
        assert_eq!(output.contiguity, input.contiguity, "{name}");
        assert_eq!(output.view, ViewFact::Materialized, "{name}");
        assert_eq!(output.alias, AliasFact::Unique, "{name}");
        assert_eq!(output.mutation, MutationFact::ValueSemantics, "{name}");
    }

    let mut real_input = input;
    let ValueKindFact::Numeric(numeric) = &mut real_input.kind else {
        unreachable!()
    };
    numeric.domain = NumericDomain::Real;
    let identity = infer_catalog_call(
        builtin_catalog_entry_by_name("conj").expect("conj entry"),
        &CallRequest {
            arguments: vec![real_input.clone()],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(identity.outputs[0], real_input);
}

#[test]
fn numeric_component_contracts_encode_logical_character_and_rejection_boundaries() {
    use runmat_types::{
        AliasFact, CallRequest, DynamicReason, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
        ValueKindFact, ViewFact,
    };

    let request = |argument: ValueFact| CallRequest {
        arguments: vec![argument],
        literals: Default::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let conj = infer_catalog_call(
        builtin_catalog_entry_by_name("conj").expect("conj entry"),
        &request(logical.clone()),
    );
    assert!(conj.diagnostics.is_empty());
    assert_eq!(conj.outputs[0].kind, ValueKindFact::Logical);
    assert_eq!(conj.outputs[0].shape, logical.shape);

    for name in ["real", "imag"] {
        let inference = infer_catalog_call(
            builtin_catalog_entry_by_name(name).expect("component entry"),
            &request(logical.clone()),
        );
        assert!(inference.diagnostics.is_empty(), "{name}");
        assert_eq!(
            inference.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );
        assert_eq!(inference.outputs[0].shape, logical.shape, "{name}");
        assert_eq!(inference.outputs[0].view, ViewFact::Materialized, "{name}");
        assert_eq!(inference.outputs[0].alias, AliasFact::Unique, "{name}");
    }

    let mut resident_logical = logical.clone();
    resident_logical.residency = ResidencyFact::Device {
        provider: Some("single-only-provider".into()),
    };
    let converted = infer_catalog_call(
        builtin_catalog_entry_by_name("real").expect("real entry"),
        &request(resident_logical),
    );
    assert_eq!(
        converted.outputs[0].residency,
        ResidencyFact::Unknown,
        "a class-changing logical projection cannot promise resident double output"
    );

    let scalar = infer_catalog_call(
        builtin_catalog_entry_by_name("imag").expect("imag entry"),
        &request(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }))),
    );
    assert_eq!(scalar.outputs[0].storage, StorageFact::Scalar);

    for name in ["conj", "real", "imag"] {
        let character = ValueFact::proven(
            ValueKindFact::Character,
            ShapeFact::from(vec![Some(1), Some(5)]),
            StorageFact::Dense,
        );
        let inference = infer_catalog_call(
            builtin_catalog_entry_by_name(name).expect("component entry"),
            &request(character),
        );
        assert!(inference.diagnostics.is_empty(), "{name}");
        assert_eq!(
            inference.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );
    }

    let sparse = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(2)]),
        StorageFact::Sparse,
    );
    let rejected = infer_catalog_call(
        builtin_catalog_entry_by_name("real").expect("real entry"),
        &request(sparse),
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-NUMERIC-COMPONENT-SPARSE"));
    assert_eq!(rejected.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        rejected.outputs[0].certainty,
        runmat_types::CertaintyFact::Dynamic(DynamicReason::UnsupportedRepresentation)
    );

    let rejected = infer_catalog_call(
        builtin_catalog_entry_by_name("imag").expect("imag entry"),
        &request(ValueFact::scalar(ValueKindFact::String)),
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-NUMERIC-COMPONENT-INPUT"));

    let dynamic_input = ValueFact {
        shape: ShapeFact::from(vec![Some(7), Some(9)]),
        ..ValueFact::unknown(DynamicReason::RuntimeValue)
    };
    let dynamic = infer_catalog_call(
        builtin_catalog_entry_by_name("conj").expect("conj entry"),
        &request(dynamic_input.clone()),
    );
    assert!(dynamic.diagnostics.is_empty());
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(dynamic.outputs[0].shape, dynamic_input.shape);
}

#[test]
fn phase_angle_contract_preserves_float_class_shape_and_residency() {
    use runmat_types::{
        AliasFact, CallRequest, ContiguityFact, LayoutFact, NumericClass, NumericDomain,
        NumericFact, OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact,
        ValueFact, ValueKindFact, ViewFact,
    };

    let mut input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    input.residency = ResidencyFact::Device {
        provider: Some("phase-provider".into()),
    };
    let entry = builtin_catalog_entry_by_name("angle").expect("angle entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::PhaseAngle)
    );
    let inference = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![input.clone()],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(inference.diagnostics.is_empty());
    let output = &inference.outputs[0];
    assert_eq!(
        output.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(output.shape, input.shape);
    assert_eq!(output.storage, StorageFact::Dense);
    assert_eq!(output.residency, input.residency);
    assert_eq!(output.layout, LayoutFact::ColumnMajor);
    assert_eq!(output.contiguity, ContiguityFact::Contiguous);
    assert_eq!(output.view, ViewFact::Materialized);
    assert_eq!(output.alias, AliasFact::Unique);

    let rejected = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }))],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-PHASE-ANGLE-INPUT"));
}

#[test]
fn signum_contract_preserves_numeric_facts_and_types_conversion_boundaries() {
    use runmat_types::{
        AliasFact, CallRequest, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact, ValueKindFact,
        ViewFact,
    };

    let request = |argument| CallRequest {
        arguments: vec![argument],
        literals: Default::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let entry = builtin_catalog_entry_by_name("sign").expect("sign entry");
    assert_eq!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Signum)
    );
    for (class, domain) in [
        (NumericClass::UInt64, NumericDomain::Real),
        (NumericClass::Single, NumericDomain::Complex),
    ] {
        let mut input = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact { class, domain }),
            ShapeFact::from(vec![Some(4), Some(2)]),
            StorageFact::Dense,
        );
        input.residency = ResidencyFact::Device {
            provider: Some("sign-provider".into()),
        };
        let inference = infer_catalog_call(entry, &request(input.clone()));
        assert!(inference.diagnostics.is_empty(), "{class:?} {domain:?}");
        let output = &inference.outputs[0];
        assert_eq!(output.kind, input.kind);
        assert_eq!(output.shape, input.shape);
        assert_eq!(output.residency, input.residency);
        assert_eq!(output.view, ViewFact::Materialized);
        assert_eq!(output.alias, AliasFact::Unique);
    }

    let mut logical = ValueFact::proven(
        ValueKindFact::Logical,
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    logical.residency = ResidencyFact::Device {
        provider: Some("single-provider".into()),
    };
    let converted = infer_catalog_call(entry, &request(logical));
    assert_eq!(
        converted.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(converted.outputs[0].residency, ResidencyFact::Unknown);

    let rejected = infer_catalog_call(
        entry,
        &request(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Complex,
        }))),
    );
    assert!(rejected
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SIGNUM-COMPLEX-INTEGER"));
}

#[test]
fn exponential_contracts_share_class_rules_but_keep_exact_sparse_semantics() {
    use std::collections::BTreeMap;

    use runmat_types::{
        AliasFact, CallRequest, NumericClass, NumericDomain, NumericFact, ObjectFact,
        OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
        ViewFact,
    };

    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(5), Some(3)]),
        StorageFact::Sparse,
    );
    let request = CallRequest {
        arguments: vec![input],
        literals: Default::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };

    let exp = infer_catalog_call(
        builtin_catalog_entry_by_name("exp").expect("exp entry"),
        &request,
    );
    let expm1 = infer_catalog_call(
        builtin_catalog_entry_by_name("expm1").expect("expm1 entry"),
        &request,
    );
    assert!(exp.diagnostics.is_empty());
    assert!(expm1.diagnostics.is_empty());
    assert_eq!(exp.outputs[0].kind, expm1.outputs[0].kind);
    assert_eq!(exp.outputs[0].shape, expm1.outputs[0].shape);
    assert_eq!(exp.outputs[0].storage, StorageFact::Dense);
    assert_eq!(expm1.outputs[0].storage, StorageFact::Sparse);
    assert_eq!(expm1.outputs[0].view, ViewFact::Materialized);
    assert_eq!(expm1.outputs[0].alias, AliasFact::Unique);

    let integer = infer_catalog_call(
        builtin_catalog_entry_by_name("expm1").expect("expm1 entry"),
        &CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::UInt64,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(1), Some(4)]),
                StorageFact::Dense,
            )],
            ..request
        },
    );
    assert_eq!(
        integer.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    assert_eq!(
        integer.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(4)])
    );

    let complex_integer = infer_catalog_call(
        builtin_catalog_entry_by_name("expm1").expect("expm1 entry"),
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int32,
                domain: NumericDomain::Complex,
            }))],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(complex_integer
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-EXPONENTIAL-COMPLEX-INTEGER"));

    let table = ValueFact::proven(
        ValueKindFact::Object(ObjectFact {
            class: None,
            runtime_class: Some(runmat_types::standard::TABLE.owned()),
            properties: BTreeMap::from([(
                "Variables".into(),
                ValueFact::scalar(ValueKindFact::Logical),
            )]),
            properties_complete: true,
            handle_semantics: Some(false),
        }),
        ShapeFact::Scalar,
        StorageFact::Opaque,
    );
    let table_output = infer_catalog_call(
        builtin_catalog_entry_by_name("expm1").expect("expm1 entry"),
        &CallRequest {
            arguments: vec![table],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    let ValueKindFact::Object(table_output) = &table_output.outputs[0].kind else {
        panic!("expected tabular object fact");
    };
    assert_eq!(
        table_output.runtime_class,
        Some(runmat_types::standard::TABLE.owned())
    );
    assert!(table_output.properties.is_empty());
    assert!(!table_output.properties_complete);

    let user_object = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(ClassIdentity::new("UserClass").unwrap()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: None,
    }));
    let overloaded = infer_catalog_call(
        builtin_catalog_entry_by_name("expm1").expect("expm1 entry"),
        &CallRequest {
            arguments: vec![user_object],
            literals: Default::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(overloaded.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(overloaded.outputs[0].shape, ShapeFact::Scalar);
}

#[test]
fn log1p_contract_tracks_value_dependent_real_to_complex_promotion() {
    use runmat_types::{
        CallRequest, CertaintyFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
        NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
        ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("log1p").expect("log1p entry");
    assert!(matches!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Log1p)
    ));

    let infer = |input: ValueFact, literal: LiteralValue| {
        infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![input],
                literals: LiteralContext::new(vec![literal]),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        )
    };
    let single = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::Scalar,
        StorageFact::Dense,
    );
    let promoted = infer(single.clone(), LiteralValue::Number(-2.0));
    assert!(promoted.diagnostics.is_empty());
    assert_eq!(
        promoted.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    let boundary = infer(single, LiteralValue::Number(-1.0));
    assert_eq!(
        boundary.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        })
    );

    let vector = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(3)]),
        StorageFact::Dense,
    );
    let vector_result = infer(
        vector,
        LiteralValue::Vector(vec![
            LiteralValue::Number(0.0),
            LiteralValue::Number(-2.0),
            LiteralValue::Number(1.0),
        ]),
    );
    assert_eq!(
        vector_result.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(
        vector_result.outputs[0].shape,
        ShapeFact::from(vec![Some(1), Some(3)])
    );

    let signed_dynamic = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Int64,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(2), Some(4)]),
                StorageFact::Dense,
            )],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(signed_dynamic.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(
        signed_dynamic.outputs[0].shape,
        ShapeFact::from(vec![Some(2), Some(4)])
    );
    assert_eq!(
        signed_dynamic.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    );

    let unsigned = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }))],
            literals: LiteralContext::new(vec![LiteralValue::Unknown]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(
        unsigned.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
}

#[test]
fn log1p_contract_rejects_unsupported_representations_without_fabricating_facts() {
    use runmat_types::{
        CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("log1p").expect("log1p entry");
    let request_for = |input| CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let complex_integer = infer_catalog_call(
        entry,
        &request_for(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Int32,
            domain: NumericDomain::Complex,
        }))),
    );
    assert!(complex_integer
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG1P-COMPLEX-INTEGER"));

    let sparse = infer_catalog_call(
        entry,
        &request_for(ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(3), Some(3)]),
            StorageFact::Sparse,
        )),
    );
    assert!(sparse
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG1P-SPARSE"));

    let too_many = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![
                ValueFact::scalar(ValueKindFact::Logical),
                ValueFact::scalar(ValueKindFact::Logical),
            ],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(too_many
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG1P-ARITY"));
}

#[test]
fn logarithm_contracts_preserve_shape_class_and_value_dependent_domain() {
    use runmat_types::{
        CallRequest, CertaintyFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
        NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
        ValueFact, ValueKindFact,
    };

    for (name, base) in [
        ("log", LogarithmBase::Natural),
        ("log10", LogarithmBase::Common),
    ] {
        let entry = builtin_catalog_entry_by_name(name).expect("logarithm entry");
        assert!(matches!(
            entry.contract.inference_rule,
            BuiltinInferenceRule::Math(MathInferenceRule::Logarithm(actual)) if actual == base
        ));
        let infer = |input: ValueFact, literal: LiteralValue| {
            infer_catalog_call(
                entry,
                &CallRequest {
                    arguments: vec![input],
                    literals: LiteralContext::new(vec![literal]),
                    outputs: OutputSelection::new(RequestedOutputCount::One),
                },
            )
        };

        let single = ValueFact::proven(
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            ShapeFact::from(vec![Some(2), Some(3)]),
            StorageFact::Dense,
        );
        let negative = infer(single.clone(), LiteralValue::Number(-1.0));
        assert!(negative.diagnostics.is_empty(), "{name}");
        assert_eq!(
            negative.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Complex,
            }),
            "{name}"
        );
        assert_eq!(
            negative.outputs[0].shape,
            ShapeFact::from(vec![Some(2), Some(3)]),
            "{name}"
        );

        let nonnegative = infer(single, LiteralValue::Number(1.0));
        assert_eq!(
            nonnegative.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );

        let unsigned = infer(
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            })),
            LiteralValue::Unknown,
        );
        assert_eq!(
            unsigned.outputs[0].kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            }),
            "{name}"
        );

        let signed = infer(
            ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Int64,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(4), Some(2)]),
                StorageFact::Dense,
            ),
            LiteralValue::Unknown,
        );
        assert_eq!(signed.outputs[0].kind, ValueKindFact::Unknown, "{name}");
        assert_eq!(
            signed.outputs[0].shape,
            ShapeFact::from(vec![Some(4), Some(2)]),
            "{name}"
        );
        assert_eq!(
            signed.outputs[0].certainty,
            CertaintyFact::Dynamic(DynamicReason::RuntimeValue),
            "{name}"
        );
    }
}

#[test]
fn logarithm_contracts_cover_tabular_symbolic_and_rejected_inputs() {
    use std::collections::BTreeMap;

    use runmat_types::{
        CallRequest, ClassIdentity, LiteralContext, NumericClass, NumericDomain, NumericFact,
        ObjectFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    let request_for = |input| CallRequest {
        arguments: vec![input],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    for name in ["log", "log10"] {
        let entry = builtin_catalog_entry_by_name(name).expect("logarithm entry");
        for class in [
            runmat_types::standard::TABLE,
            runmat_types::standard::TIMETABLE,
        ] {
            let table = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
                class: None,
                runtime_class: Some(class.owned()),
                properties: BTreeMap::from([(
                    "Variables".into(),
                    ValueFact::scalar(ValueKindFact::Unknown),
                )]),
                properties_complete: true,
                handle_semantics: None,
            }));
            let inferred = infer_catalog_call(entry, &request_for(table));
            let ValueKindFact::Object(output) = &inferred.outputs[0].kind else {
                panic!("{name} must preserve tabular identity");
            };
            assert_eq!(output.runtime_class, Some(class.owned()), "{name}");
            assert!(output.properties.is_empty(), "{name}");
            assert!(!output.properties_complete, "{name}");
        }

        let complex_integer = infer_catalog_call(
            entry,
            &request_for(ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Int32,
                domain: NumericDomain::Complex,
            }))),
        );
        assert!(complex_integer
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGARITHM-COMPLEX-INTEGER"));

        let sparse = infer_catalog_call(
            entry,
            &request_for(ValueFact::proven(
                ValueKindFact::Numeric(NumericFact {
                    class: NumericClass::Double,
                    domain: NumericDomain::Real,
                }),
                ShapeFact::from(vec![Some(3), Some(3)]),
                StorageFact::Sparse,
            )),
        );
        assert!(sparse
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGARITHM-SPARSE"));

        let too_many = infer_catalog_call(
            entry,
            &CallRequest {
                arguments: vec![
                    ValueFact::scalar(ValueKindFact::Logical),
                    ValueFact::scalar(ValueKindFact::Logical),
                ],
                literals: LiteralContext::default(),
                outputs: OutputSelection::new(RequestedOutputCount::One),
            },
        );
        assert!(too_many
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGARITHM-ARITY"));
    }

    let symbolic = ValueFact::scalar(ValueKindFact::Symbolic);
    let log = infer_catalog_call(
        builtin_catalog_entry_by_name("log").expect("log entry"),
        &request_for(symbolic.clone()),
    );
    assert_eq!(log.outputs[0].kind, ValueKindFact::Symbolic);
    let log10 = infer_catalog_call(
        builtin_catalog_entry_by_name("log10").expect("log10 entry"),
        &request_for(symbolic),
    );
    assert!(log10
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOGARITHM-INPUT"));

    let user_object = ValueFact::scalar(ValueKindFact::Object(ObjectFact {
        class: None,
        runtime_class: Some(ClassIdentity::new("UserClass").unwrap()),
        properties: BTreeMap::new(),
        properties_complete: false,
        handle_semantics: None,
    }));
    let overloaded = infer_catalog_call(
        builtin_catalog_entry_by_name("log").expect("log entry"),
        &request_for(user_object),
    );
    assert_eq!(overloaded.outputs[0].kind, ValueKindFact::Unknown);
}

#[test]
fn log2_contract_distinguishes_logarithm_and_dissection_outputs() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ShapeFact, StorageFact, ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("log2").expect("log2 entry");
    assert!(matches!(
        entry.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Log2)
    ));
    assert_eq!(
        entry.descriptor.output_mode,
        BuiltinOutputMode::ByRequestedOutputCount
    );

    let input = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let one = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![input.clone()],
            literals: LiteralContext::new(vec![LiteralValue::Number(-1.0)]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(one.diagnostics.is_empty());
    assert_eq!(
        one.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );

    let two = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![input],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert!(two.diagnostics.is_empty());
    assert_eq!(two.outputs.len(), 2);
    for output in &two.outputs {
        assert_eq!(
            output.kind,
            ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Single,
                domain: NumericDomain::Real,
            })
        );
        assert_eq!(output.shape, ShapeFact::from(vec![Some(2), Some(3)]));
    }

    let integer = infer_catalog_call(
        entry,
        &CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::UInt64,
                domain: NumericDomain::Real,
            }))],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
        },
    );
    assert!(integer.diagnostics.is_empty());
    assert!(integer.outputs.iter().all(|output| {
        output.kind
            == ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })
    }));
}

#[test]
fn log2_dissection_contract_rejects_complex_gpu_and_excess_outputs() {
    use runmat_types::{
        CallRequest, LiteralContext, NumericClass, NumericDomain, NumericFact, OutputSelection,
        RequestedOutputCount, ResidencyFact, ValueFact, ValueKindFact,
    };

    let entry = builtin_catalog_entry_by_name("log2").expect("log2 entry");
    let request = |argument, requested| CallRequest {
        arguments: vec![argument],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(requested),
    };
    let complex = infer_catalog_call(
        entry,
        &request(
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Complex,
            })),
            RequestedOutputCount::Exactly(2),
        ),
    );
    assert!(complex
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG2-COMPLEX-DISSECTION"));

    let mut resident = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    resident.residency = ResidencyFact::Device { provider: None };
    let gpu = infer_catalog_call(entry, &request(resident, RequestedOutputCount::Exactly(2)));
    assert!(gpu
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-LOG2-GPU-DISSECTION"));

    let excessive = infer_catalog_call(
        entry,
        &request(
            ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
                class: NumericClass::Double,
                domain: NumericDomain::Real,
            })),
            RequestedOutputCount::Exactly(3),
        ),
    );
    assert!(excessive
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-TYPE-CALL-ARITY"));
}

#[test]
fn gather_contract_maps_corresponding_outputs_to_host_and_checks_output_count() {
    use runmat_types::{
        CallRequest, OutputSelection, RequestedOutputCount, ResidencyFact, ValueFact, ValueKindFact,
    };
    let mut first = ValueFact::scalar(ValueKindFact::Logical);
    first.residency = ResidencyFact::Device {
        provider: Some("gpu-a".into()),
    };
    let mut second = ValueFact::scalar(ValueKindFact::String);
    second.residency = ResidencyFact::Device {
        provider: Some("gpu-b".into()),
    };
    let request = CallRequest {
        arguments: vec![first.clone(), second.clone()],
        literals: runmat_types::LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("gather").expect("gather entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(inference.outputs.len(), 2);
    assert_eq!(inference.outputs[0].kind, first.kind);
    assert_eq!(inference.outputs[1].kind, second.kind);
    assert!(inference
        .outputs
        .iter()
        .all(|fact| fact.residency == ResidencyFact::Host));

    let bad_request = CallRequest {
        outputs: OutputSelection::new(RequestedOutputCount::One),
        ..request
    };
    let bad = infer_catalog_call(
        builtin_catalog_entry_by_name("gather").expect("gather entry"),
        &bad_request,
    );
    assert!(bad
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-GATHER-OUTPUTS"));
}

#[test]
fn struct_contract_tracks_literal_field_names_and_value_facts() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, OutputSelection, RequestedOutputCount,
        ValueFact, ValueKindFact,
    };
    let field_name = ValueFact::scalar(ValueKindFact::String);
    let value = ValueFact::scalar(ValueKindFact::Logical);
    let request = CallRequest {
        arguments: vec![field_name, value.clone()],
        literals: LiteralContext::new(vec![
            LiteralValue::String("Flag".into()),
            LiteralValue::Unknown,
        ]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("struct").expect("struct entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    let ValueKindFact::Struct(fact) = &inference.outputs[0].kind else {
        panic!("expected struct fact")
    };
    assert!(fact.fields_complete);
    assert_eq!(fact.fields.get("Flag"), Some(&value));
}

#[test]
fn feval_contract_preserves_known_callable_outputs_and_dynamic_effects() {
    use runmat_types::{
        CallRequest, CallableFact, EffectKind, LiteralContext, OutputSelection,
        RequestedOutputCount, ValueFact, ValueKindFact,
    };
    assert_eq!(
        builtin_catalog_entry_by_name("feval")
            .expect("feval entry")
            .link
            .execution_stack,
        runmat_types::ExecutionStackRequirement::Process
    );
    let first_output = ValueFact::scalar(ValueKindFact::Logical);
    let second_output = ValueFact::scalar(ValueKindFact::String);
    let callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: vec![first_output.clone(), second_output.clone()],
        outputs_complete: true,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: true,
    }));
    let request = CallRequest {
        arguments: vec![callable],
        literals: LiteralContext::default(),
        outputs: OutputSelection::new(RequestedOutputCount::Exactly(2)),
    };
    let inference = infer_catalog_call(
        builtin_catalog_entry_by_name("feval").expect("feval entry"),
        &request,
    );
    assert!(inference.diagnostics.is_empty());
    assert_eq!(inference.outputs, vec![first_output, second_output]);
    assert!(inference.effects.0.contains(&EffectKind::HostCallback));
    assert!(inference.effects.0.contains(&EffectKind::MaySuspend));
    assert!(inference.effects.0.contains(&EffectKind::MayThrow));
    assert!(inference.effects.0.contains(&EffectKind::Unknown));
    assert!(inference.capabilities.0.is_empty());

    let partially_known_callable = ValueFact::scalar(ValueKindFact::Callable(CallableFact {
        identity: None,
        capabilities: Default::default(),
        parameters: Vec::new(),
        parameters_complete: false,
        outputs: vec![ValueFact::scalar(ValueKindFact::Character)],
        outputs_complete: false,
        variadic_inputs: true,
        variadic_outputs: false,
        captures: Vec::new(),
        captures_complete: false,
    }));
    let dynamic = infer_catalog_call(
        builtin_catalog_entry_by_name("feval").expect("feval entry"),
        &CallRequest {
            arguments: vec![partially_known_callable],
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(3)),
        },
    );
    assert!(dynamic.diagnostics.is_empty());
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Character);
    assert!(dynamic.outputs[1..]
        .iter()
        .all(|output| output.kind == ValueKindFact::Unknown));

    let missing_target = infer_catalog_call(
        builtin_catalog_entry_by_name("feval").expect("feval entry"),
        &CallRequest {
            arguments: Vec::new(),
            literals: LiteralContext::default(),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert!(missing_target
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-FEVAL-ARITY"));
}

#[test]
fn root_contracts_distinguish_principal_promotion_from_real_domain_validation() {
    use runmat_types::{
        CallRequest, CertaintyFact, DynamicReason, LiteralContext, LiteralValue, NumericClass,
        NumericDomain, NumericFact, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
        ValueFact, ValueKindFact,
    };

    let request = |input, literal| CallRequest {
        arguments: vec![input],
        literals: LiteralContext::new(vec![literal]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let sqrt = builtin_catalog_entry_by_name("sqrt").expect("sqrt entry");
    let realsqrt = builtin_catalog_entry_by_name("realsqrt").expect("realsqrt entry");
    assert_eq!(
        sqrt.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Root(RootKind::Principal))
    );
    assert_eq!(
        realsqrt.contract.inference_rule,
        BuiltinInferenceRule::Math(MathInferenceRule::Root(RootKind::RealOnly))
    );
    assert_eq!(sqrt.placement.fusion, BuiltinFusionPolicy::Never);
    assert_eq!(realsqrt.placement.fusion, BuiltinFusionPolicy::Never);

    let single = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(2), Some(3)]),
        StorageFact::Dense,
    );
    let promoted = infer_catalog_call(sqrt, &request(single.clone(), LiteralValue::Number(-4.0)));
    assert!(promoted.diagnostics.is_empty());
    assert_eq!(
        promoted.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Complex,
        })
    );
    assert_eq!(promoted.outputs[0].shape, single.shape);

    let dynamic = infer_catalog_call(sqrt, &request(single.clone(), LiteralValue::Unknown));
    assert_eq!(dynamic.outputs[0].kind, ValueKindFact::Unknown);
    assert_eq!(dynamic.outputs[0].shape, single.shape);
    assert_eq!(
        dynamic.outputs[0].certainty,
        CertaintyFact::Dynamic(DynamicReason::RuntimeValue)
    );

    let invalid_domain = infer_catalog_call(realsqrt, &request(single, LiteralValue::Number(-4.0)));
    assert!(invalid_domain
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-REALSQRT-DOMAIN"));

    let symbolic = ValueFact::scalar(ValueKindFact::Symbolic);
    let symbolic_root = infer_catalog_call(sqrt, &request(symbolic.clone(), LiteralValue::Unknown));
    assert!(symbolic_root.diagnostics.is_empty());
    assert_eq!(symbolic_root.outputs[0], symbolic);
}

#[test]
fn root_contracts_preserve_only_supported_storage_and_classes() {
    use runmat_types::{
        CallRequest, LiteralContext, LiteralValue, NumericClass, NumericDomain, NumericFact,
        OutputSelection, RequestedOutputCount, ResidencyFact, ShapeFact, StorageFact, ValueFact,
        ValueKindFact,
    };

    let request = |input| CallRequest {
        arguments: vec![input],
        literals: LiteralContext::new(vec![LiteralValue::Unknown]),
        outputs: OutputSelection::new(RequestedOutputCount::One),
    };
    let sqrt = builtin_catalog_entry_by_name("sqrt").expect("sqrt entry");
    let realsqrt = builtin_catalog_entry_by_name("realsqrt").expect("realsqrt entry");
    let mut sparse_single = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(4), Some(2)]),
        StorageFact::Sparse,
    );
    sparse_single.residency = ResidencyFact::Device {
        provider: Some("sparse-provider".into()),
    };
    let preserved = infer_catalog_call(realsqrt, &request(sparse_single.clone()));
    assert!(preserved.diagnostics.is_empty());
    assert_eq!(preserved.outputs[0].kind, sparse_single.kind);
    assert_eq!(preserved.outputs[0].shape, sparse_single.shape);
    assert_eq!(preserved.outputs[0].storage, StorageFact::Sparse);
    assert_eq!(preserved.outputs[0].residency, ResidencyFact::Host);

    let sparse_principal = infer_catalog_call(sqrt, &request(sparse_single));
    assert!(sparse_principal
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-SQRT-SPARSE"));

    let integer_real = ValueFact::proven(
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::UInt16,
            domain: NumericDomain::Real,
        }),
        ShapeFact::from(vec![Some(1), Some(4)]),
        StorageFact::Dense,
    );
    let integer_sqrt = infer_catalog_call(sqrt, &request(integer_real.clone()));
    assert_eq!(
        integer_sqrt.outputs[0].kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Double,
            domain: NumericDomain::Real,
        })
    );
    let integer_realsqrt = infer_catalog_call(realsqrt, &request(integer_real));
    assert!(integer_realsqrt
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "RM-CATALOG-REALSQRT-INPUT"));

    let mut resident = ValueFact::scalar(ValueKindFact::Numeric(NumericFact {
        class: NumericClass::Double,
        domain: NumericDomain::Real,
    }));
    resident.residency = ResidencyFact::Device {
        provider: Some("root-provider".into()),
    };
    let resident_result = infer_catalog_call(
        realsqrt,
        &CallRequest {
            arguments: vec![resident],
            literals: LiteralContext::new(vec![LiteralValue::Number(4.0)]),
            outputs: OutputSelection::new(RequestedOutputCount::One),
        },
    );
    assert_eq!(resident_result.outputs[0].residency, ResidencyFact::Unknown);
}
