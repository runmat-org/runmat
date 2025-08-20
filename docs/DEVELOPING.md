# Developing RunMat

This project is organised as a Cargo workspace. Each component lives in a separate crate under `crates/` and uses kebab-case names. To build the entire workspace run:

```bash
cargo build
```

## Repository Layout

- `crates/` - all library and binary crates
- `docs/` - design documents like `ARCHITECTURE.md`
- `examples/` - small `.m` and Rust examples (coming later)
- `tests/` - language and performance test suites (coming later)

## Running the REPL

At this early stage the REPL simply tokenizes input and prints the token names:

```bash
cargo run -p runmat-repl
```

Future milestones will integrate the parser, VM and JIT.

## Contributing

1. Fork the repository and create a feature branch.
2. Keep commits focused and include a meaningful message.
3. Run `cargo fmt` and `cargo check` before submitting a pull request.
4. Update `docs/` if the design changes.

This developing guide will grow as the runtime gains more features.
