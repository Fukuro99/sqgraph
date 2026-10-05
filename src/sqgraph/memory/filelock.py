"""Inter-process exclusive file lock (stdlib only).

``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows. The lock is held on an open file descriptor, so the
operating system releases it when the process dies. Not re-entrant: acquiring twice (even from the same
thread) blocks; :class:`sqgraph.memory.Memory` tracks nesting itself.
"""

from __future__ import annotations

import os
import sys
import time
from types import TracebackType

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    """Exclusive lock on ``path`` (created if missing; the file is left in place).

    >>> with FileLock("memory.db.lock", timeout=30):
    ...     ...  # only one process at a time gets here

    ``timeout`` (seconds) bounds the wait: ``None`` waits forever, ``0`` tries once. On expiry
    :class:`TimeoutError` is raised.
    """

    def __init__(self, path: str | os.PathLike[str], timeout: float | None = 300.0, poll_interval: float = 0.05) -> None:
        self.path = os.fspath(path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._fd: int | None = None

    @property
    def locked(self) -> bool:
        return self._fd is not None

    def acquire(self, timeout: float | None = None) -> None:
        """Take the lock, waiting up to ``timeout`` (default: the instance's) seconds."""
        if self._fd is not None:
            raise RuntimeError(f"file lock {self.path!r} is already held by this object")
        wait = self.timeout if timeout is None else timeout
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        deadline = None if wait is None else time.monotonic() + wait
        try:
            while not _try_lock(fd):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out after {wait:g}s waiting for the write lock {self.path!r}: "
                        "another process is writing to the same memory database"
                    )
                time.sleep(self.poll_interval)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        """Release the lock (no-op if not held)."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            _unlock(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.release()
