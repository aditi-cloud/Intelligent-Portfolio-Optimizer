"""Versioned JSON cache; no pickle or executable serialized data."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Callable

import pandas as pd

from ..contracts import PortfolioRequest
from ..errors import DataQualityError
from ..validation import validate_request
from .contracts import PriceDataset
from .preparation import validate_prices
from .provider import PriceProvider


class CachedPriceProvider:
    def __init__(self, provider: PriceProvider, directory: Path, *,
                 ttl: timedelta = timedelta(hours=6), clock: Callable | None = None):
        if ttl.total_seconds() <= 0 or not math.isfinite(ttl.total_seconds()):
            raise ValueError("Cache TTL must be positive and finite")
        self.provider = provider
        self.directory = Path(directory)
        self.ttl = ttl
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.name = provider.name
        self.version = provider.version
        self.adjustment_policy = provider.adjustment_policy

    def cache_key(self, request: PortfolioRequest) -> str:
        request = validate_request(request)
        inputs = dict(schema=1, tickers=request.tickers, start=request.start_date.isoformat(),
                      end=request.end_date.isoformat(), currency=request.currency,
                      provider=self.name, version=self.version, adjustment=self.adjustment_policy)
        return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    def fetch_prices(self, request: PortfolioRequest) -> PriceDataset:
        request = validate_request(request)
        now = self._clock()
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise ValueError("Cache clock must return a timezone-aware datetime")
        key = self.cache_key(request)
        path = self.directory / f"{key}.json"
        try:
            document = json.loads(path.read_text())
            if document["schema"] != 1 or document["key"] != key:
                raise ValueError("Cache schema/key mismatch")
            written = datetime.fromisoformat(document["cached_at"])
            if not timedelta(0) <= now - written < self.ttl:
                raise ValueError("Expired cache")
            dataset = PriceDataset(
                pd.DataFrame(document["values"], columns=document["tickers"],
                             index=pd.to_datetime(document["dates"])),
                document["currencies"], document["provider"],
                datetime.fromisoformat(document["retrieved_at"]), document["adjustment_policy"],
            )
            if dataset.provider != self.name or dataset.adjustment_policy != self.adjustment_policy:
                raise ValueError("Cache provenance mismatch")
            dataset.prices.index.name = document.get("index_name")
            return replace(dataset, prices=validate_prices(dataset, request))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, DataQualityError):
            pass  # Invalid/expired entries are misses; never return stale data.
        dataset = self.provider.fetch_prices(request)
        dataset = replace(dataset, prices=validate_prices(dataset, request))
        if dataset.provider != self.name or dataset.adjustment_policy != self.adjustment_policy:
            raise DataQualityError("Provider provenance does not match cache configuration")
        document = dict(
            schema=1, key=key, cached_at=now.isoformat(),
            retrieved_at=dataset.retrieved_at.isoformat(), provider=dataset.provider,
            adjustment_policy=dataset.adjustment_policy, currencies=dict(dataset.currencies),
            tickers=list(dataset.prices.columns),
            index_name=dataset.prices.index.name,
            dates=[day.date().isoformat() for day in dataset.prices.index],
            values=dataset.prices.astype(object).where(dataset.prices.notna(), None).values.tolist(),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        # Replace atomically so readers cannot observe a partially written file.
        with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(document, handle, allow_nan=False)
                handle.flush()
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return dataset
