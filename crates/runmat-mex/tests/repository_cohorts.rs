#![cfg(not(target_family = "wasm"))]

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use runmat_mex::{MexBuild, MexModule};
use runmat_value::{Tensor, Value};

const FIELDTRIP_REVISION: &str = "2e14f7291090b19568827799096daec7dfb99de8";

struct RepositoryCheckout {
    path: PathBuf,
}

impl RepositoryCheckout {
    fn from_environment(variable: &str, revision: &'static str, license: &'static str) -> Self {
        let path = PathBuf::from(
            env::var_os(variable)
                .unwrap_or_else(|| panic!("{variable} must name the pinned repository checkout")),
        );
        assert!(
            path.join(license).is_file(),
            "missing {license} in {}",
            path.display()
        );

        let output = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&path)
            .output()
            .expect("git must be available for repository-cohort verification");
        assert!(
            output.status.success(),
            "could not inspect repository checkout {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        let actual = String::from_utf8(output.stdout).unwrap();
        assert_eq!(
            actual.trim(),
            revision,
            "unexpected revision for {}",
            path.display()
        );

        Self { path }
    }

    fn source(&self, relative: &str) -> PathBuf {
        let source = self.path.join(relative);
        assert!(
            source.is_file(),
            "missing cohort source {}",
            source.display()
        );
        source
    }
}

fn build_module(source: &Path, output_name: &str, directory: &Path) -> MexModule {
    let artifact = MexBuild::new(source, directory)
        .output_name(output_name)
        .compile()
        .unwrap_or_else(|error| panic!("could not build {}: {error}", source.display()));
    MexModule::load(&artifact.module)
        .unwrap_or_else(|error| panic!("could not load {}: {error}", artifact.module.display()))
}

fn tensor(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(values, shape).unwrap())
}

#[test]
#[ignore = "requires the pinned external repository checkouts named by its environment variables"]
fn pinned_fieldtrip_c_gateways_build_and_execute() {
    let checkout = RepositoryCheckout::from_environment(
        "RUNMAT_MEX_FIELDTRIP_CHECKOUT",
        FIELDTRIP_REVISION,
        "COPYING",
    );
    let output = tempfile::tempdir().unwrap();

    let determinant = build_module(
        &checkout.source("src/det2x2.c"),
        "fieldtrip_det2x2",
        output.path(),
    );
    let determinant_result = determinant
        .invoke(
            &[tensor(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2])],
            1,
            determinant.api_mode(),
        )
        .unwrap();
    assert_eq!(determinant_result.outputs, vec![Value::Num(-2.0)]);

    let process_id = build_module(
        &checkout.source("src/getpid.c"),
        "fieldtrip_getpid",
        output.path(),
    );
    let process_id_result = process_id.invoke(&[], 1, process_id.api_mode()).unwrap();
    let Value::Int(process_id) = &process_id_result.outputs[0] else {
        panic!("getpid must return a fixed-width integer scalar");
    };
    assert!(process_id.try_to_u64().is_some_and(|value| value > 0));
}
