"""Offline is the default contract: fail accidental HTTP data ingestion."""
import os
import pytest

@pytest.fixture(autouse=True)
def offline_http(monkeypatch):
    if os.environ.get('PITWALL_NETWORK_TESTS') != '1':
        import requests
        def blocked(*args, **kwargs):
            raise AssertionError('External HTTP is disabled in the offline suite')
        monkeypatch.setattr(requests.sessions.Session, 'request', blocked)
