import asyncio
from fastapi import WebSocket

class WSManager:
    def __init__(self):
        self.active: set = set()
        self._loop = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active.discard(websocket)

    async def broadcast(self, data: dict):
        dead = set()
        for ws in tuple(self.active):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self.active -= dead

    def broadcast_from_thread(self, data: dict):
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(data), self._loop)

manager = WSManager()