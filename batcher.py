"""Groups requests that arrive while the gpu is busy into one pass."""
import asyncio


class MicroBatcher:
    def __init__(self, fn, max_batch=16, max_wait_ms=2.0):
        self.fn = fn  # sync: list of items -> list of results
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue = None
        self._task = None
        self.batches = 0
        self.items = 0

    def start(self):
        self.queue = asyncio.Queue()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()

    @property
    def mean_batch_size(self):
        return round(self.items / self.batches, 2) if self.batches else None

    async def submit(self, item):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((item, fut))
        return await fut

    async def _collect(self):
        loop = asyncio.get_running_loop()
        batch = [await self.queue.get()]
        deadline = loop.time() + self.max_wait
        while len(batch) < self.max_batch:
            if not self.queue.empty():
                batch.append(self.queue.get_nowait())
                continue
            left = deadline - loop.time()
            if left <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self.queue.get(), left))
            except asyncio.TimeoutError:
                break
        return batch

    async def _loop(self):
        while True:
            batch = await self._collect()
            try:
                results = await asyncio.to_thread(self.fn, [item for item, _ in batch])
            except Exception as exc:  # fail the whole batch, keep serving
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            self.batches += 1
            self.items += len(batch)
            for (_, fut), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)
