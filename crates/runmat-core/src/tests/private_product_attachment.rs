//! One authentic selected product, projected into the complete executing graph
//! and its named-only durable view. Mutations are deliberately made only in
//! local staging clones; no malformed product is admitted as an authority.

use std::rc::Rc;
use std::sync::Arc;

use super::*;

#[derive(Clone, Copy)]
enum Rejection {
    PublishedOwner,
    PrivateBody,
    PublishedSourceOrdinal,
    DuplicateOrigin,
    MissingOrigin,
    PriorProduct,
}

#[test]
fn private_selected_product_publication_checks_complete_graph_and_is_failure_atomic() {
    run_deep_semantic_test(|| {
        let _guard = cwd_lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        std::fs::create_dir_all(tmp.path().join("private")).expect("root private dir");
        std::fs::create_dir_all(tmp.path().join("+pkg/private")).expect("package private dir");
        std::fs::write(
            tmp.path().join("runmat.toml"),
            "[package]\nname = \"demo\"\n[sources]\nroots = [\".\"]\n",
        )
        .expect("project manifest");
        std::fs::write(
            tmp.path().join("private/helper.m"),
            "function y = helper(x); y = x + 100; end",
        )
        .expect("root private helper");
        std::fs::write(
            tmp.path().join("+pkg/private/helper.m"),
            "function y = helper(x); y = x + 1; end",
        )
        .expect("package private helper");
        std::fs::write(
            tmp.path().join("+pkg/entry.m"),
            "function y = entry(); y = helper(41); end",
        )
        .expect("package entry");
        std::fs::write(
            tmp.path().join("+pkg/other.m"),
            "function y = other(); y = 7; end",
        )
        .expect("second durable package body");
        let source = "root_value = helper(1); pkg_value = pkg.entry(); other_value = pkg.other();";
        std::fs::write(tmp.path().join("main.m"), source).expect("main source");
        let _cwd = push_cwd(tmp.path());
        let mut session = RunMatSession::with_options(false, false).expect("session");
        let unit = block_on(session.compile_executable_unit(
            ExecutableSource::frozen(
                "demo",
                "main.m",
                tmp.path().join("main.m").to_string_lossy(),
                source,
            ),
            None,
        ))
        .expect("selected unit");
        let map = unit
            .source_map()
            .portable_v2()
            .expect("portable source map");
        let runtime = runmat_runtime::source_context::bind_portable_source_map_v2(
            runmat_runtime::context::RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            )),
            &map,
        )
        .expect("exact runtime source binding");
        let mut executing = unit
            .runtime_bytecode(&runtime)
            .expect("issued execution view")
            .function_registry();
        let envelope = unit
            .portable_envelope()
            .expect("admitted selected envelope");
        let owner = unit
            .selected_source_owner(&runtime)
            .expect("selected issuer");
        let product = runmat_vm::SelectedExecutableProduct::admit(
            Arc::from(
                envelope
                    .canonical_bytes()
                    .expect("canonical selected bytes")
                    .into_boxed_slice(),
            ),
            owner,
        )
        .expect("admitted selected product");
        let selected_ids = executing
            .functions
            .iter()
            .filter_map(|(id, body)| {
                body.binding_origin
                    .filter(|origin| product.body_digest(*origin).is_some())
                    .map(|_| *id)
            })
            .collect::<Vec<_>>();
        let package_entry = executing
            .functions
            .iter()
            .find_map(|(id, body)| (body.display_name == "pkg.entry").then_some(*id))
            .expect("package entry");
        let package_private = executing
            .functions
            .iter()
            .find_map(|(id, body)| (body.display_name == "pkg.__private__.helper").then_some(*id))
            .expect("package-private callee");
        let package_other = executing
            .functions
            .iter()
            .find_map(|(id, body)| (body.display_name == "pkg.other").then_some(*id))
            .expect("second named package body");
        assert!(selected_ids.contains(&package_entry));
        assert!(selected_ids.contains(&package_private));
        assert!(selected_ids.contains(&package_other));
        let other_source = executing
            .get_current(package_private)
            .and_then(|body| body.source_id)
            .expect("different admitted source ordinal");
        assert_ne!(
            executing.get_current(package_entry).unwrap().source_id,
            Some(other_source)
        );
        let mut durable = executing.clone();
        // These two named package functions are the fixture's durable
        // publication set. A private source owner also marks the public
        // package callee's lookup scope, so it is not an execution-only test.
        let named_ids = [package_entry, package_other];
        for id in selected_ids.iter().copied() {
            if !named_ids.contains(&id) {
                durable.remove(id).expect("hide execution-only companion");
            }
        }
        let mut durable_ids = named_ids.to_vec();
        assert!(durable_ids.contains(&package_entry));
        assert!(!durable_ids.contains(&package_private));
        assert!(
            durable_ids.len() >= 2,
            "failure atomicity needs two named bodies"
        );
        durable_ids.sort_by_key(|id| (*id == package_entry, id.0));
        executing
            .attach_current_selected_executable_product(selected_ids, Rc::clone(&product))
            .expect("complete executing graph");

        let foreign_runtime = runmat_runtime::source_context::bind_portable_source_map_v2(
            runmat_runtime::context::RuntimeContext::new(Rc::new(
                runmat_runtime::execution::RuntimeExecutionService::new(),
            )),
            &map,
        )
        .expect("independent source issuance");
        let foreign_product = runmat_vm::SelectedExecutableProduct::admit(
            Arc::clone(product.canonical_v3()),
            unit.selected_source_owner(&foreign_runtime)
                .expect("independent selected issuer"),
        )
        .expect("independent authentic product");

        for rejection in [
            Rejection::PublishedOwner,
            Rejection::PrivateBody,
            Rejection::PublishedSourceOrdinal,
            Rejection::DuplicateOrigin,
            Rejection::MissingOrigin,
            Rejection::PriorProduct,
        ] {
            let mut active = executing.clone();
            let mut published = durable.clone();
            match rejection {
                Rejection::PublishedOwner => {
                    let foreign_owner = Arc::clone(
                        &active
                            .get_current(package_private)
                            .expect("private body")
                            .definition_owner,
                    );
                    assert!(!Arc::ptr_eq(
                        &foreign_owner,
                        &published
                            .get_current(package_entry)
                            .unwrap()
                            .definition_owner
                    ));
                    published
                        .functions
                        .get_mut(&package_entry)
                        .expect("published entry")
                        .definition_owner = foreign_owner;
                }
                Rejection::PrivateBody => {
                    active
                        .functions
                        .get_mut(&package_private)
                        .expect("private callee")
                        .display_name
                        .push_str("_altered");
                }
                Rejection::PublishedSourceOrdinal => {
                    published
                        .functions
                        .get_mut(&package_entry)
                        .expect("published entry")
                        .source_id = Some(other_source);
                }
                Rejection::DuplicateOrigin => {
                    let origin = active.get_current(package_entry).unwrap().binding_origin;
                    active
                        .functions
                        .get_mut(&package_private)
                        .expect("private callee")
                        .binding_origin = origin;
                }
                Rejection::MissingOrigin => {
                    let mut foreign = active
                        .get_current(package_private)
                        .and_then(|body| body.binding_origin)
                        .expect("private admitted origin");
                    foreign.function =
                        runmat_types::ProgramFunctionId(foreign.function.0.saturating_add(100_000));
                    assert!(product.body_digest(foreign).is_none());
                    active
                        .functions
                        .get_mut(&package_private)
                        .expect("private callee")
                        .binding_origin = Some(foreign);
                }
                Rejection::PriorProduct => {
                    published
                        .functions
                        .get_mut(&package_entry)
                        .expect("published entry")
                        .selected_executable_product = Some(Rc::clone(&foreign_product));
                }
            }
            let error = published
                .attach_published_selected_executable_product(
                    durable_ids.iter().copied(),
                    &active,
                    Rc::clone(&product),
                )
                .expect_err("malformed selected or published definition was admitted");
            assert!(
                match rejection {
                    Rejection::PublishedOwner
                    | Rejection::PrivateBody
                    | Rejection::PublishedSourceOrdinal => matches!(
                        &error,
                        runmat_vm::SelectedExecutableAttachmentError::AuthorityMismatch(_)
                    ),
                    Rejection::DuplicateOrigin | Rejection::MissingOrigin => matches!(
                        &error,
                        runmat_vm::SelectedExecutableAttachmentError::Fingerprints(_)
                    ),
                    Rejection::PriorProduct => matches!(
                        &error,
                        runmat_vm::SelectedExecutableAttachmentError::PriorProductConflict(_)
                    ),
                },
                "wrong rejection source: {error:?}"
            );
            for id in &durable_ids {
                let attached = published
                    .get_current(*id)
                    .and_then(|body| body.selected_executable_product.as_ref());
                if matches!(rejection, Rejection::PriorProduct) && *id == package_entry {
                    assert!(attached.is_some_and(|prior| Rc::ptr_eq(prior, &foreign_product)));
                } else {
                    assert!(
                        attached.is_none(),
                        "rejection partially published a product"
                    );
                }
            }
        }

        durable
            .attach_published_selected_executable_product(
                durable_ids.iter().copied(),
                &executing,
                Rc::clone(&product),
            )
            .expect("named publication from the complete selected graph");
        assert!(durable.get_current(package_private).is_none());
        assert!(durable_ids.iter().all(|id| {
            durable
                .get_current(*id)
                .and_then(|body| body.selected_executable_product.as_ref())
                .is_some_and(|attached| Rc::ptr_eq(attached, &product))
        }));
    });
}
