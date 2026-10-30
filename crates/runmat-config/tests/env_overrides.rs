mod support;

use runmat_config::document::RunmatConfigFormat;
use runmat_config::runtime::{
    ConfigLoadOptions, ConfigLoader, ConfigSource, ConfigurationPolicy, UserSettingsSource,
};
use tempfile::TempDir;

const ENV_VARS: &[&str] = &[
    "RUNMAT_CONFIG",
    "RUNMAT_CALLSTACK_LIMIT",
    "RUNMAT_JIT_THRESHOLD",
    "RUNMAT_RUNTIME__JIT__THRESHOLD",
];

#[test]
fn runmat_config_env_selects_config_file() {
    let _lock = support::env_lock();
    support::clear_env(ENV_VARS);

    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("custom.toml");
    std::fs::write(
        &config_path,
        r#"
[runtime]
callstack_limit = 123
jit = { enabled = true, threshold = 42, optimization_level = "speed" }
"#,
    )
    .unwrap();

    std::env::set_var("RUNMAT_CONFIG", &config_path);
    std::env::set_var("RUNMAT_CALLSTACK_LIMIT", "999");
    std::env::set_var("RUNMAT_JIT_THRESHOLD", "999");

    let config = ConfigLoader::load().unwrap();
    assert_eq!(config.runtime.callstack_limit, 999);
    assert_eq!(config.jit.threshold, 999);

    std::env::set_var("RUNMAT_RUNTIME__JIT__THRESHOLD", "1001");
    let mut options = ConfigLoadOptions {
        user: UserSettingsSource::Document {
            source: r#"{"runtime":{"callstack_limit":321,"jit":{"enabled":false}}}"#.into(),
            format: RunmatConfigFormat::Json,
        },
        project_document: Some((
            "[runtime.jit]\nthreshold = 72\n".into(),
            RunmatConfigFormat::Toml,
        )),
        environment: true,
        ..Default::default()
    };
    let resolved = ConfigLoader::load_with_options(&options).unwrap();
    assert_eq!(resolved.config.jit.threshold, 1001);
    assert_eq!(
        resolved.source("runtime.jit.threshold"),
        Some(&ConfigSource::Environment {
            variable: "RUNMAT_RUNTIME__JIT__THRESHOLD".into()
        })
    );
    assert_eq!(
        resolved.source("runtime.callstack_limit"),
        Some(&ConfigSource::Environment {
            variable: "RUNMAT_CALLSTACK_LIMIT".into()
        })
    );
    assert!(
        matches!(resolved.source("runtime.jit.enabled"), Some(ConfigSource::Explicit { path }) if path == &config_path)
    );
    options.policy = ConfigurationPolicy::ClusterWorker;
    let worker = ConfigLoader::load_with_options(&options).unwrap();
    assert_eq!(worker.config.jit.threshold, 72);
    assert!(worker.config.jit.enabled);
    assert_eq!(
        worker.source("runtime.callstack_limit"),
        Some(&ConfigSource::BuiltIn)
    );

    support::clear_env(ENV_VARS);
}

#[test]
fn otel_endpoint_activation_and_explicit_disable_have_exact_sources() {
    let _lock = support::env_lock();
    let variables = [
        "RUNMAT_CONFIG",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "RUNMAT_OTEL_ENDPOINT",
        "RUNMAT_OTEL_ENABLED",
        "RUNMAT_RUNTIME__LOGGING__OTEL__ENABLED",
    ];
    support::clear_env(&variables);
    let options = ConfigLoadOptions {
        user: UserSettingsSource::Disabled,
        environment: true,
        ..Default::default()
    };
    for variable in [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "RUNMAT_OTEL_ENDPOINT",
    ] {
        std::env::set_var(variable, "http://collector.example:4318");
        let resolved = ConfigLoader::load_with_options(&options).unwrap();
        assert!(resolved.config.logging.otel.enabled, "{variable}");
        assert_eq!(
            resolved.source("runtime.logging.otel.enabled"),
            Some(&ConfigSource::Environment {
                variable: variable.into()
            })
        );
        std::env::remove_var(variable);
    }
    std::env::set_var("RUNMAT_OTEL_ENDPOINT", "http://collector.example:4318");
    std::env::set_var("RUNMAT_OTEL_ENABLED", "false");
    assert!(
        !ConfigLoader::load_with_options(&options)
            .unwrap()
            .config
            .logging
            .otel
            .enabled
    );
    std::env::set_var("RUNMAT_RUNTIME__LOGGING__OTEL__ENABLED", "true");
    let resolved = ConfigLoader::load_with_options(&options).unwrap();
    assert!(resolved.config.logging.otel.enabled);
    assert!(resolved.config.telemetry.enabled);
    assert_eq!(
        resolved.source("runtime.logging.otel.enabled"),
        Some(&ConfigSource::Environment {
            variable: "RUNMAT_RUNTIME__LOGGING__OTEL__ENABLED".into()
        })
    );
    support::clear_env(&variables);
}

#[test]
fn runmat_config_env_missing_file_is_error() {
    let _lock = support::env_lock();
    support::clear_env(ENV_VARS);

    std::env::set_var(
        "RUNMAT_CONFIG",
        "/tmp/definitely-missing-runmat-config.toml",
    );
    let err = ConfigLoader::load().expect_err("missing RUNMAT_CONFIG path should fail");
    let message = err.to_string();
    assert!(message.contains("RUNMAT_CONFIG points to a missing file"));

    support::clear_env(ENV_VARS);
}
