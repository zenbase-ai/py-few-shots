from asyncio import iscoroutinefunction
from typing import TypeVar

from asyncer import asyncify, syncify

C = TypeVar("C")


def is_target(method_name: str) -> bool:
    return method_name == "__call__" or not method_name.startswith("_")


def syncify_class(cls: C) -> C:
    for base in [cls] + list(cls.__bases__):
        for name, method in base.__dict__.items():
            if iscoroutinefunction(method) and is_target(name):
                setattr(cls, name, syncify(method, raise_sync_error=False))
    return cls


def asyncify_class(cls: C) -> C:
    for base in [cls] + list(cls.__bases__):
        for name, method in base.__dict__.items():
            if not iscoroutinefunction(method) and is_target(name):
                setattr(cls, name, asyncify(method))
    return cls
