from typing import Optional

from fastapi import Header


def get_session_id(x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id")) -> str:
    return x_session_id or "anonymous"