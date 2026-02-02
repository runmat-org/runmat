use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let spec_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest"))
        .join("../../openapi/runmat-public.yaml");
    println!("cargo:rerun-if-changed={}", spec_path.display());

    let spec_text = fs::read_to_string(&spec_path).expect("read openapi spec");
    let spec: openapiv3::OpenAPI = serde_yaml::from_str(&spec_text).expect("parse openapi spec");

    let settings = progenitor::GenerationSettings::default();
    let mut generator = progenitor::Generator::new(&settings);
    let tokens = generator
        .generate_tokens(&spec)
        .expect("generate client tokens");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("out dir"));
    let out_file = out_dir.join("runmat_public.rs");
    fs::write(&out_file, tokens.to_string()).expect("write openapi client");
}
