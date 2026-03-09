# Commit Message Template

Use the format:

`type(scope): short why-focused subject`

## Suggested types

- `feat`: new user-visible capability
- `fix`: bug fix or behavioral correction
- `perf`: measurable performance improvement
- `refactor`: internal restructure without behavior change
- `docs`: documentation-only updates
- `test`: tests only
- `chore`: maintenance work

## Subject rules

- Keep it short (about 50-72 chars when possible).
- Focus on **why** and effect, not implementation details.
- Use lowercase type/scope for consistency.

## Body template

```
<what changed in 1-2 bullets>
<why this change matters>

Validation:
- <tests/benchmarks run>
```

## Examples

- `perf(cache): avoid eager deserialization during graph pruning`
- `fix(runtime): recover dependencies when pruned cache hit turns miss`
- `docs(readme): clarify dimension cache behavior`
