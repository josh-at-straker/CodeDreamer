# Proactive Memory

Instead of waiting for the model to request context, Proactive Memory anticipates what will be useful and pre-fetches it.

## How It Works

When starting a dream about a file, ProactiveMemory gathers:

1. **Imports** - What modules does this file depend on?
2. **Related Files** - What other files are often mentioned alongside this one? (from graph edges)
3. **Graph Context** - What previous insights exist about this file?
4. **TRM Context** - What recent thoughts might be relevant?

This context is included in the prompt before the model even asks for it.

## Example Output

```
## Proactive Context (anticipated relevant info)

**Imports**: This file uses: logging, time, dataclasses, pathlib

**Related Files**: Often seen with: validator.py, graph.py

**Previous Insights**:
- [DREAM] Consider adding error handling for edge cases...
- [CONCEPT] This follows the singleton pattern...

**Recent Thoughts**:
- Noticed similar patterns in conductor.py...
```

## API

```python
from codedreamer.proactive import get_proactive_memory

proactive = get_proactive_memory()

# Get context for a file
ctx = proactive.get_context("/path/to/file.py", code_content)

# Format for prompt
prompt_section = ctx.to_prompt_section()

# Access individual components
print(ctx.imported_modules)  # ['logging', 'time', ...]
print(ctx.related_files)     # ['validator.py', 'graph.py']
print(ctx.confidence)        # 0.75 (how much context was found)
```

## Confidence Score

The confidence score (0.0 - 1.0) indicates how much context was found:

| Score | Meaning |
|-------|---------|
| 0.0 | No context found (cold start) |
| 0.25 | One signal found |
| 0.50 | Two signals found |
| 0.75 | Three signals found |
| 1.0 | All four signals found (imports, related, graph, TRM) |

Higher confidence = richer context for the model.

## Integration with Dream Cycle

Proactive Memory is automatically used in the dream cycle:

```
Dream Cycle:
1. Select random chunk
2. ← ProactiveMemory.get_context() ← NEW
3. Get TRM context
4. Build prompt with all context
5. Run reasoning model
6. Run coder model
7. Run critic loop
8. Validate and save
```

## Future Enhancements

- **Pre-warming**: Load related files into memory before they're needed
- **Predictive fetching**: Learn which files tend to be accessed together
- **Context compression**: Summarize long context to fit token limits

