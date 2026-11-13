# Frozen interpreter artifacts

These files are checked-in wire artifacts, not values synthesized or version-mutated by tests. The `interpreter-program-current.artifact` control uses the interpreter V2 framing with bytecode schema 7 and function-registry schema 5. The stale artifacts preserve the immediately preceding schema number while retaining a body encoded with that historical field layout. The `*-legacy-raw.json` files preserve the unframed V1-era JSON payloads and prove that V2 readers do not silently admit legacy bytes.

Revision tests must treat these files as immutable. Add a new fixture when a wire schema changes; do not rewrite an old fixture with the current serializer.
