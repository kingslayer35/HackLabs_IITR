"""A valid credential for another account must never create or upload a repo."""
import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from tools.upload_models import main, verify_account, verify_files


def test_wrong_environment_account_cannot_publish(monkeypatch):
    api=Mock();api.whoami.return_value={"name":"wrong-account"}
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(HfApi=lambda **kwargs:api))
    monkeypatch.setenv("HF_TOKEN","test-placeholder-not-a-real-token")
    monkeypatch.setattr(sys,"argv",["upload_models.py","--namespace","yobro4619"])
    with pytest.raises(RuntimeError,match="Account mismatch"):
        main()
    api.create_repo.assert_not_called()
    api.upload_folder.assert_not_called()


def test_exact_owner_is_required_for_personal_namespace():
    api=SimpleNamespace(whoami=lambda:{"name":"yobro4619"})
    assert verify_account(api,"yobro4619")=="yobro4619"
    with pytest.raises(RuntimeError,match="Account mismatch"):
        verify_account(api,"another-account")


def test_organization_requires_explicit_expected_token_owner():
    api=SimpleNamespace(whoami=lambda:{"name":"team-member"})
    assert verify_account(api,"team-organization",expected_user="team-member")=="team-member"


def test_remote_weight_and_text_bytes_are_verified(tmp_path):
    folder=tmp_path/"models"/"encoder"/"best"
    folder.mkdir(parents=True)
    weights=b"fixed model bytes"
    card=b"# Model\n"
    (folder/"model.safetensors").write_bytes(weights)
    (folder/"README.md").write_bytes(card)
    siblings=[
        SimpleNamespace(rfilename=".gitattributes"),
        SimpleNamespace(rfilename="model.safetensors",lfs={"sha256":hashlib.sha256(weights).hexdigest()}),
        SimpleNamespace(rfilename="README.md",lfs=None,
            blob_id=hashlib.sha1(f"blob {len(card)}\0".encode()+card).hexdigest()),
    ]
    api=SimpleNamespace(model_info=lambda *args,**kwargs:SimpleNamespace(private=True,sha="revision",siblings=siblings))
    result=verify_files(api,"owner/model",folder)
    assert result["revision"]=="revision"
    assert len(result["verified_files"])==2
    siblings[1].lfs["sha256"]="0"*64
    with pytest.raises(RuntimeError,match="checksum differs"):
        verify_files(api,"owner/model",folder)


def test_unexpected_remote_file_is_rejected(tmp_path):
    info=SimpleNamespace(private=True,siblings=[SimpleNamespace(rfilename="examples.parquet")])
    api=SimpleNamespace(model_info=lambda *args,**kwargs:info)
    with pytest.raises(RuntimeError,match="inventory differs"):
        verify_files(api,"owner/model",tmp_path)


def test_public_remote_repository_is_rejected(tmp_path):
    api=SimpleNamespace(model_info=lambda *args,**kwargs:SimpleNamespace(private=False))
    with pytest.raises(RuntimeError,match="must remain private"):
        verify_files(api,"owner/model",tmp_path)
