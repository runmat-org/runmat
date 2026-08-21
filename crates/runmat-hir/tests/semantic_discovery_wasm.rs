use runmat_hir::testing::{discover_tests, SemanticDiscoveryInput, SemanticTestSource};
use runmat_hir::{lower, LoweringContext};
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource, UnsavedRunBuffer};
use sha2::{Digest, Sha256};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

fn source(path: &str, content: &str) -> SavedRunSource {
    SavedRunSource {
        owner_identity: "registry:acme/math@1.2.3#sha256:tree".into(),
        relative_path: path.into(),
        content: content.into(),
    }
}

fn digest(value: &str) -> String {
    runmat_execution::Digest::sha256(value).to_string()
}

fn environment() -> runmat_execution::ProgramEnvironment {
    runmat_execution::ProgramEnvironment::new(
        3,
        4,
        runmat_execution::Digest::sha256(b"runtime"),
        runmat_execution::Digest::sha256(b"catalog"),
        "matlab",
    )
    .unwrap()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn frozen_semantic_discovery_has_a_native_wasm_golden() {
    let snapshot = FrozenTestRunSnapshot::freeze(
        digest("graph"),
        "sha256:base-sources",
        environment(),
        digest("config"),
        vec![
            source(
                "tests/BrowserTest.m",
                "classdef (TestTags={'browser','fast'}) BrowserTest < matlab.unittest.TestCase\n properties(TestParameter)\n Mode = {'saved'}\n end\n methods(Test)\n function testValue(obj)\n end\n end\nend",
            ),
            source(
                "tests/section_test.m",
                "%% saved section\nassert(1 == 1)\n",
            ),
        ],
        vec![UnsavedRunBuffer {
            owner_identity: "registry:acme/math@1.2.3#sha256:tree".into(),
            relative_path: "tests/section_test.m".into(),
            content: "%% browser overlay\nassert(1 == 1)\n".into(),
        }],
    )
    .unwrap();
    let assemblies = snapshot
        .sources
        .iter()
        .map(|source| {
            let program = runmat_parser::parse(&source.content).unwrap();
            lower(&program, &LoweringContext::empty()).unwrap().assembly
        })
        .collect::<Vec<_>>();
    let sources = snapshot
        .sources
        .iter()
        .zip(&assemblies)
        .map(|(source, assembly)| SemanticTestSource {
            owner_identity: &source.owner_identity,
            relative_source_identity: &source.relative_path,
            source_text: &source.content,
            assembly,
        })
        .collect::<Vec<_>>();
    let discovery = discover_tests(&SemanticDiscoveryInput {
        program_revision: snapshot.program_revision,
        sources: &sources,
    });
    assert!(discovery.diagnostics.is_empty());
    let encoded = serde_json::to_vec(&discovery).unwrap();
    let digest = format!("sha256:{:x}", Sha256::digest(&encoded));
    assert_eq!(
        digest,
        "sha256:4f819a1450b5b24125dc79b3d570502eb7dbedb0ff1763abd6a2b4427514701e",
        "serialized discovery: {}",
        String::from_utf8(encoded).unwrap()
    );
}
