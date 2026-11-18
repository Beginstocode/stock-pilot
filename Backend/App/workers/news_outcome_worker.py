"""Idempotent worker for ready historical news outcome horizons."""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from Backend.App.config import DATA_DIR
from Backend.App.database.connection import SessionLocal
from Backend.App.database.models import NewsIntelligenceEvent, NewsIntelligenceOutcome
from Backend.App.services.news_intelligence_service import calculate_ready_news_outcomes

LOGGER = logging.getLogger(__name__)
_PROCESS_RUN_LOCK = threading.Lock()


@dataclass(frozen=True)
class NewsOutcomeWorkerStats:
    events_checked: int = 0
    outcomes_calculated: int = 0
    events_pending: int = 0
    events_skipped: int = 0
    failures: int = 0
    status: str = "completed"
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _outcome_horizon_count(row: NewsIntelligenceOutcome | None) -> int:
    if row is None:
        return 0
    count = 0
    for horizon in ("1d", "3d", "5d"):
        fields = (
            f"stock_return_{horizon}_pct",
            f"nifty_return_{horizon}_pct",
            f"excess_return_{horizon}_pct",
            f"observed_at_{horizon}",
        )
        count += int(all(getattr(row, field) is not None for field in fields))
    return count


class _CrossProcessFileLock:
    """A short local-filesystem lock shared by worker processes on one host."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None
        self.windows = os.name == "nt"

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            if self.windows:
                import msvcrt

                self.handle.seek(0, os.SEEK_END)
                if self.handle.tell() == 0:
                    self.handle.write(b"\0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            self.handle.close()
            self.handle = None
            return False

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            if self.windows:
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


class NewsOutcomeWorker:
    """Runs one bounded pass, reports useful counts, and safely retries next cycle."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        calculator: Callable[[Session], int] = calculate_ready_news_outcomes,
        lock_path: Path | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.calculator = calculator
        self.lock_path = lock_path or Path(
            os.getenv(
                "NEWS_OUTCOME_WORKER_LOCK_PATH",
                str(DATA_DIR / "processed" / ".news_outcome_worker.lock"),
            )
        )

    @staticmethod
    def _snapshot(db: Session) -> dict[int, int]:
        events = list(db.scalars(select(NewsIntelligenceEvent.event_id)).all())
        outcomes = {
            row.event_id: row
            for row in db.scalars(select(NewsIntelligenceOutcome)).all()
        }
        return {event_id: _outcome_horizon_count(outcomes.get(event_id)) for event_id in events}

    def run_once(self) -> NewsOutcomeWorkerStats:
        if not _PROCESS_RUN_LOCK.acquire(blocking=False):
            LOGGER.info("News outcome worker skipped: another run is active in this process.")
            return NewsOutcomeWorkerStats(status="overlap_skipped")
        lock = _CrossProcessFileLock(self.lock_path)
        try:
            if not lock.acquire():
                LOGGER.info("News outcome worker skipped: another process holds %s.", self.lock_path)
                return NewsOutcomeWorkerStats(status="overlap_skipped")
            return self._run_locked()
        except Exception as exc:
            LOGGER.exception("News outcome worker failed before processing.")
            stats = NewsOutcomeWorkerStats(failures=1, status="failed", errors=(str(exc),))
            LOGGER.error("News outcome worker stats: %s", json.dumps(stats.as_dict()))
            return stats
        finally:
            lock.release()
            _PROCESS_RUN_LOCK.release()

    def _run_locked(self) -> NewsOutcomeWorkerStats:
        checked = 0
        before: dict[int, int] = {}
        failures: list[str] = []
        try:
            with self.session_factory() as db:
                before = self._snapshot(db)
                checked = len(before)
                try:
                    self.calculator(db)
                except Exception:
                    db.rollback()
                    raise
                after = self._snapshot(db)
            deltas = {
                event_id: max(0, value - before.get(event_id, 0))
                for event_id, value in after.items()
            }
            outcomes_calculated = sum(deltas.values())
            events_processed = sum(value > 0 for value in deltas.values())
            pending = sum(value < 3 for value in after.values())
            stats = NewsOutcomeWorkerStats(
                events_checked=checked,
                outcomes_calculated=outcomes_calculated,
                events_pending=pending,
                events_skipped=max(0, checked - events_processed),
            )
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            LOGGER.exception("News outcome worker pass failed; next scheduled run may retry.")
            after: dict[int, int] = {}
            try:
                with self.session_factory() as db:
                    after = self._snapshot(db)
            except Exception:
                LOGGER.exception("Could not read post-failure news outcome statistics.")
            deltas = {
                event_id: max(0, value - before.get(event_id, 0))
                for event_id, value in after.items()
            }
            stats = NewsOutcomeWorkerStats(
                events_checked=checked,
                outcomes_calculated=sum(deltas.values()),
                events_pending=sum(value < 3 for value in after.values()) if after else checked,
                events_skipped=max(0, checked - sum(value > 0 for value in deltas.values())),
                failures=len(failures),
                status="failed",
                errors=tuple(failures),
            )
        LOGGER.info("News outcome worker stats: %s", json.dumps(stats.as_dict()))
        return stats


__all__ = ["NewsOutcomeWorker", "NewsOutcomeWorkerStats"]
