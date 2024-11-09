import pytest

from best_shot.utils.asyncio import (
    asyncify_class,
    syncify_class,
    is_target,
    iscoroutinefunction,
)


def test_is_target():
    assert is_target("__call__") is True
    assert is_target("public_method") is True
    assert is_target("_private_method") is False


class SyncClass:
    def __call__(self, x):
        return x * 2

    def method(self, x):
        return x + 1

    def _private(self):
        return "private"


@asyncify_class
class AsyncClass(SyncClass):
    pass


@pytest.mark.anyio
async def test_asyncify_class():
    obj = AsyncClass()

    result = await obj(2)
    assert result == 4, "__call__ should become async"

    result = await obj.method(1)
    assert result == 2, "public method should become async"

    assert not iscoroutinefunction(obj._private), "private method should remain sync"


class AsyncBaseClass:
    async def __call__(self, x):
        return x * 2

    async def method(self, x):
        return x + 1

    async def _private(self):
        return "private"


@syncify_class
class SyncFromAsyncClass(AsyncBaseClass):
    pass


def test_syncify_class():
    obj = SyncFromAsyncClass()

    assert obj(2) == 4, "__call__ should become sync"

    assert obj.method(1) == 2, "Public method should become sync"

    assert iscoroutinefunction(obj._private), "Private method should remain async"
