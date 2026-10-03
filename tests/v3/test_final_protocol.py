import json
import sys
import pytest
from geopolicy.v3 import final_protocol as protocol


def test_source_hash_ignores_only_windows_line_endings(tmp_path):
    source = tmp_path/'source.py'
    source.write_bytes(b'x = 1\r\n')
    expected = protocol.source_hash(source)
    source.write_bytes(b'x = 1\n')
    assert protocol.source_hash(source) == expected
    source.write_bytes(b'x = 2\n')
    assert protocol.source_hash(source) != expected


def test_frozen_registry_rejects_unregistered_or_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    configs = tmp_path/'configs/v3'
    configs.mkdir(parents=True)
    plan,runtime = {},{}
    (configs/'plan.json').write_text(json.dumps(plan))
    (configs/'runtime_limits.json').write_text(json.dumps(runtime))
    source,checkpoint = tmp_path/'engine.py',tmp_path/'weights.pt'
    source.write_bytes(b'code\n');checkpoint.write_bytes(b'weights')
    record = dict(key='registered',checkpoint='weights.pt',checkpoint_sha256=protocol.sha(checkpoint))
    frozen = dict(registry=[record],sources={'engine.py':protocol.source_hash(source)},
                  python_version=list(sys.version_info[:3]),packages={},
                  plan_sha256=protocol.json_hash(plan),runtime_limits_sha256=protocol.json_hash(runtime))
    (configs/'final_protocol.json').write_text(json.dumps(frozen))
    assert protocol.registered('registered')[1] == record
    with pytest.raises(AssertionError):protocol.registered('new_model')
    checkpoint.write_bytes(b'changed')
    with pytest.raises(AssertionError):protocol.registered('registered')
    checkpoint.write_bytes(b'weights');source.write_bytes(b'changed code\n')
    with pytest.raises(AssertionError):protocol.registered('registered')
    source.write_bytes(b'code\n');(configs/'plan.json').write_text('{"changed":true}')
    with pytest.raises(AssertionError):protocol.registered('registered')
