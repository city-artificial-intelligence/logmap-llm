'''
MISC utils
'''
from pydantic import BaseModel

###
# DEBUG HELPER (TYPE SWITCH FOR RESPONSE FMT)
###

def resolve_response_format_to_str(resp_fmt) -> str:
    '''
    ad-hoc type-switch
    returns the resp_fmt value as str for debug logging
    '''
    if resp_fmt is None:
        return "Plain (None)"
    
    if isinstance(resp_fmt, type) and issubclass(resp_fmt, BaseModel):
        return resp_fmt.__name__

    if isinstance(resp_fmt, BaseModel):
        return f"UNEXPECTED instance of {type(resp_fmt).__name__} (expected the class itself, not an instance)"
    
    if isinstance(resp_fmt, dict):
        return f"raw dict schema (keys: {sorted(resp_fmt.keys())})"

    return f"UNKNOWN type={type(resp_fmt).__name__}"
