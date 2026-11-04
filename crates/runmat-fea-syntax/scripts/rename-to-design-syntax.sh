#!/bin/sh
# The one-time rename of this crate, runmat-fea-syntax -> runmat-design-syntax (M1).
#
# Run it once, from the repository root, in the merge window the FEA effort agrees (FEA freezes
# .fea syntax, language and LSP edits for the merge). It renames the crate directory, the
# package, every Cargo dependency, every `runmat_fea_syntax` path in Rust and every
# `crates/runmat-fea-syntax/` path in scripts, then refreshes Cargo.lock. It changes no code
# behaviour: the golden .fea lock must pass unchanged afterwards.
#
#   crates/runmat-fea-syntax/scripts/rename-to-design-syntax.sh            # rename in place
#   crates/runmat-fea-syntax/scripts/rename-to-design-syntax.sh --dry-run  # list the files only
#
# Afterwards: cargo test -p runmat-design-syntax (the golden test included),
# runmat-analysis-language and runmat-lsp tests, then commit as one change.
set -eu

old_package=runmat-fea-syntax
new_package=runmat-design-syntax
old_crate=runmat_fea_syntax
new_crate=runmat_design_syntax

files=$(git grep -l -e "$old_package" -e "$old_crate" -- . ':!Cargo.lock' ':!crates/runmat-fea-syntax/scripts/rename-to-design-syntax.sh')
if [ "${1:-}" = "--dry-run" ]; then
    printf '%s\n' "$files"
    exit 0
fi
[ -d "crates/$old_package" ] || { echo "crates/$old_package is not here; run from the repository root" >&2; exit 1; }
for file in $files; do
    sed -i.rename-backup \
        -e "s#crates/$old_package/#crates/$new_package/#g" \
        -e "s#\"crates/$old_package\"#\"crates/$new_package\"#g" \
        -e "s#$old_package#$new_package#g" \
        -e "s#$old_crate#$new_crate#g" \
        "$file"
    rm -f "$file.rename-backup"
done
git mv "crates/$old_package" "crates/$new_package"
sed -i.rename-backup -e 's#^description = "FEA lexical and lossless syntax contracts for RunMat"#description = "The RunMat design language: lexer, lossless syntax tree, dialects, document engine, structured edits and projections"#' "crates/$new_package/Cargo.toml"
rm -f "crates/$new_package/Cargo.toml.rename-backup"
cargo metadata --format-version 1 > /dev/null
echo "renamed $old_package -> $new_package in $(printf '%s\n' "$files" | wc -l | tr -d ' ') files"
