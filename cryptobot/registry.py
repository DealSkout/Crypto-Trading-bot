"""The model registry: every strategy version the bot has run or considered, why it was promoted
or rejected, and which one trades where.

A model is a strategy + params + filters. Filters are what the learning loop may add on top of
a strategy: skip rules on decision features and per-regime size weights (never above 1, so the
learning loop can only make positions smaller, never bigger than the risk limits allow).

Status lifecycle:
  candidate -> shadow (passed the gate; runs in simulation next to the champion for 2 weeks)
            -> champion (the paper bot trades it) -> retired (replaced)
  candidate/shadow -> rejected (with the reason)
`live_approved` is set ONLY by a human with `python -m cryptobot approve-live <version>`.
Nothing automated ever sets it. Rollback is one command: `python -m cryptobot rollback <ver>`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY, created_at TEXT, strategy TEXT, params TEXT, filters TEXT,
  status TEXT, reason TEXT, metrics TEXT, parent TEXT, shadow_start TEXT, promoted_at TEXT,
  live_approved INTEGER DEFAULT 0, live_approved_at TEXT, live_approved_by TEXT, source TEXT,
  fingerprint TEXT);
CREATE TABLE IF NOT EXISTS model_events (
  id INTEGER PRIMARY KEY, ts TEXT, version TEXT, event TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS learning_trials (
  id INTEGER PRIMARY KEY, ts TEXT, run_id TEXT, n_trials INTEGER);
CREATE TABLE IF NOT EXISTS registry_state (key TEXT PRIMARY KEY, value TEXT);
"""

STATUSES = ("candidate", "shadow", "champion", "retired", "rejected")


@dataclass
class Model:
    id: str
    strategy: str
    params: dict[str, Any]
    filters: dict[str, Any] = field(default_factory=dict)
    status: str = "candidate"
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    parent: str = ""
    created_at: str = ""
    shadow_start: str = ""
    promoted_at: str = ""
    live_approved: bool = False
    live_approved_at: str = ""
    live_approved_by: str = ""
    source: str = ""

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.strategy, self.params, self.filters)

    def describe(self) -> str:
        extra = f" filters={json.dumps(self.filters, sort_keys=True)}" if self.filters else ""
        return f"{self.id} {self.strategy} {json.dumps(self.params, sort_keys=True)}{extra}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fingerprint(strategy: str, params: dict[str, Any], filters: dict[str, Any] | None) -> str:
    blob = json.dumps([strategy, params, filters or {}], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RegistryError(ValueError):
    pass


class Registry:
    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self.conn = sqlite3.connect(path, timeout=10)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ reading
    def _row(self, r) -> Model:
        cols = [d[0] for d in self._cols]
        d = dict(zip(cols, r, strict=True))
        return Model(
            id=d["id"],
            strategy=d["strategy"],
            params=json.loads(d["params"] or "{}"),
            filters=json.loads(d["filters"] or "{}"),
            status=d["status"],
            reason=d["reason"] or "",
            metrics=json.loads(d["metrics"] or "{}"),
            parent=d["parent"] or "",
            created_at=d["created_at"] or "",
            shadow_start=d["shadow_start"] or "",
            promoted_at=d["promoted_at"] or "",
            live_approved=bool(d["live_approved"]),
            live_approved_at=d["live_approved_at"] or "",
            live_approved_by=d["live_approved_by"] or "",
            source=d["source"] or "",
        )

    def _query(self, where: str = "1=1", args: tuple = ()) -> list[Model]:
        cur = self.conn.execute(f"SELECT * FROM models WHERE {where} ORDER BY created_at", args)
        self._cols = cur.description
        return [self._row(r) for r in cur.fetchall()]

    def get(self, version: str) -> Model:
        rows = self._query("id = ?", (version,))
        if not rows:
            raise RegistryError(f"no model version {version!r}")
        return rows[0]

    def all(self) -> list[Model]:
        return self._query()

    def champion(self) -> Model | None:
        rows = self._query("status = 'champion'")
        return rows[-1] if rows else None

    def shadows(self) -> list[Model]:
        return self._query("status = 'shadow'")

    def live_version(self) -> Model | None:
        """The version a human approved for live trading (most recent approval)."""
        cur = self.conn.execute(
            "SELECT id FROM models WHERE live_approved = 1 ORDER BY live_approved_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        return self.get(row[0]) if row else None

    def find(self, strategy: str, params: dict, filters: dict | None) -> Model | None:
        fp = fingerprint(strategy, params, filters)
        rows = self._query("fingerprint = ?", (fp,))
        return rows[-1] if rows else None

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT ts, version, event, detail FROM model_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"ts": r[0], "version": r[1], "event": r[2], "detail": r[3]} for r in rows]

    # ------------------------------------------------------------------ writing
    def _next_id(self) -> str:
        n = self.conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        return f"v{n + 1}"

    def _event(self, version: str, event: str, detail: str = "") -> None:
        self.conn.execute(
            "INSERT INTO model_events (ts, version, event, detail) VALUES (?,?,?,?)",
            (_now(), version, event, detail),
        )

    def add(
        self,
        strategy: str,
        params: dict[str, Any],
        filters: dict[str, Any] | None = None,
        status: str = "candidate",
        reason: str = "",
        metrics: dict[str, Any] | None = None,
        parent: str = "",
        source: str = "",
    ) -> Model:
        if status not in STATUSES:
            raise RegistryError(f"status must be one of {STATUSES}")
        existing = self.find(strategy, params, filters)
        if existing is not None:
            return existing  # the same model is never registered twice
        version = self._next_id()
        self.conn.execute(
            "INSERT INTO models (id, created_at, strategy, params, filters, status, reason,"
            " metrics, parent, source, fingerprint) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                version,
                _now(),
                strategy,
                json.dumps(params, sort_keys=True, default=str),
                json.dumps(filters or {}, sort_keys=True, default=str),
                status,
                reason,
                json.dumps(metrics or {}, default=str),
                parent,
                source,
                fingerprint(strategy, params, filters),
            ),
        )
        self._event(version, "registered", f"{status}: {reason}")
        self.conn.commit()
        return self.get(version)

    def set_status(
        self, version: str, status: str, reason: str = "", metrics: dict | None = None
    ) -> Model:
        if status not in STATUSES:
            raise RegistryError(f"status must be one of {STATUSES}")
        if status == "champion":
            raise RegistryError("use promote_to_paper or rollback to change the champion")
        sets, args = ["status = ?", "reason = ?"], [status, reason]
        if metrics is not None:
            sets.append("metrics = ?")
            args.append(json.dumps(metrics, default=str))
        if status == "shadow":
            sets.append("shadow_start = ?")
            args.append(_now())
        self.conn.execute(f"UPDATE models SET {', '.join(sets)} WHERE id = ?", (*args, version))
        self._event(version, status, reason)
        self.conn.commit()
        return self.get(version)

    def promote_to_paper(self, version: str, reason: str) -> Model:
        """Make `version` the paper champion. Paper only: live needs approve_live (a human)."""
        model = self.get(version)
        old = self.champion()
        if old is not None and old.id != version:
            self.conn.execute(
                "UPDATE models SET status = 'retired' WHERE id = ? AND status = 'champion'",
                (old.id,),
            )
            self._event(old.id, "retired", f"replaced by {version}")
        self.conn.execute(
            "UPDATE models SET status = 'champion', reason = ?, promoted_at = ? WHERE id = ?",
            (reason, _now(), model.id),
        )
        self._event(model.id, "champion", reason)
        self.conn.commit()
        return self.get(version)

    def rollback(self, version: str, who: str = "human") -> Model:
        """One-command rollback of the paper champion to any earlier version."""
        return self.promote_to_paper(version, f"rollback by {who}")

    def approve_live(self, version: str, who: str) -> Model:
        """HUMAN ONLY (called from the CLI with an explicit confirmation flag). The version must
        have been the paper champion first."""
        model = self.get(version)
        if not model.promoted_at:
            raise RegistryError(
                f"{version} has never been the paper champion; promote it on paper first"
            )
        self.conn.execute(
            "UPDATE models SET live_approved = 0 WHERE live_approved = 1 AND id != ?", (version,)
        )
        self.conn.execute(
            "UPDATE models SET live_approved = 1, live_approved_at = ?, live_approved_by = ?"
            " WHERE id = ?",
            (_now(), who, version),
        )
        self._event(version, "live_approved", f"by {who}")
        self.conn.commit()
        return self.get(version)

    def revoke_live(self, who: str) -> None:
        self.conn.execute("UPDATE models SET live_approved = 0 WHERE live_approved = 1")
        self._event("-", "live_revoked", f"by {who}")
        self.conn.commit()

    # ------------------------------------------------------------------ trials (deflation)
    def record_trials(self, run_id: str, n: int) -> None:
        self.conn.execute(
            "INSERT INTO learning_trials (ts, run_id, n_trials) VALUES (?,?,?)", (_now(), run_id, n)
        )
        self.conn.commit()

    def runs_since(self, since: datetime) -> int:
        """Learning runs (each one looks at the holdout once) since `since`."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM learning_trials WHERE ts >= ?", (since.isoformat(),)
        ).fetchone()
        return int(row[0])

    def trials_since(self, since: datetime) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(n_trials), 0) FROM learning_trials WHERE ts >= ?",
            (since.isoformat(),),
        ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------ the config baseline
    def sync_with_config(self, strategy: str, params: dict[str, Any]) -> Model:
        """config.yaml is the human's say. The first run registers it as the champion; a later
        human edit of strategy/params in config.yaml becomes the new champion too. While the
        config is unchanged, the registry's champion (possibly a learned one) stands."""
        fp = fingerprint(strategy, params, {})
        champ = self.champion()
        if champ is not None and self._get_state("config") == fp:
            return champ
        model = self.add(
            strategy, params, {}, status="candidate", reason="config.yaml", source="config"
        )
        if champ is None or champ.id != model.id:
            why = "initial config.yaml" if champ is None else "config.yaml changed by a human"
            model = self.promote_to_paper(model.id, why)
        self._set_state("config", fp)
        return model

    def _get_state(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM registry_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO registry_state (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()
