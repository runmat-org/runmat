from pathlib import Path
import re

ROOT = Path("crates/runmat-runtime/src/builtins")

ATTRIBUTE_FN_RE = re.compile(r"#\[\s*runtime_builtin[^\]]*\]\s*fn\s+(\w+)", re.MULTILINE)
PROVIDER_CALL = "runmat_accelerate_api::provider()"


def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    functions = ATTRIBUTE_FN_RE.findall(text)
    if len(functions) != 1:
        return False
    fn_name = functions[0]
    if PROVIDER_CALL not in text:
        return False
    context_const = f"__runmat_accel_context_{fn_name}"
    replacement = f"crate::accel_provider::maybe_provider({context_const})"
    if replacement in text:
        return False
    text = text.replace(PROVIDER_CALL, replacement)
    path.write_text(text, encoding="utf-8")
    print(f"updated {path}")
    return True

def main():
    updated = 0
    for file in ROOT.rglob("*.rs"):
        if process_file(file):
            updated += 1
    print(f"updated {updated} files")

if __name__ == "__main__":
    main()
