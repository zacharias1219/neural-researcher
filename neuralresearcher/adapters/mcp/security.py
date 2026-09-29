import secrets
import logging
from typing import Callable, Awaitable
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, auth_token: str):
        super().__init__(app)
        self.auth_token = auth_token

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]) -> JSONResponse:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse({"error": "UNAUTHORIZED", "message": "Missing or invalid authorization header"}, status_code=401)
            
        token = auth_header.replace("Bearer ", "", 1)
        if not secrets.compare_digest(token, self.auth_token):
            return JSONResponse({"error": "UNAUTHORIZED", "message": "Invalid token"}, status_code=401)
            
        return await call_next(request)
