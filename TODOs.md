# TODOs

1. We should consider documentation from the outset. Consider opting for Sphinx or MkDocs?
2. 

## Open Issues

1. Specifying 'auto' rather than 'vllm' no longer calls the correct `parse` fn in `manager.py` since the latest version of vllm is in use (previously an exception would be thrown and handled, but we need to change how this behaviour is implemented).