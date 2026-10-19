# Frozen Native IR compatibility fixture

`native-ir-5.json` is a canonical serialization of the final Native IR 5 field layout with the pre-StructArray compiler identity. It is intentionally minimal because current admission must reject the top-level Native IR revision before target ABI or instruction validation. Tests read these bytes directly; do not regenerate the fixture from the current `NativeAssembly` type.
