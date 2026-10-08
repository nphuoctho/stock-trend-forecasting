#!/usr/bin/env python3
"""Populate document indexes and fields in a generated DOCX via LibreOffice."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


def property_value(name: str, value: object) -> PropertyValue:
    prop = PropertyValue()
    prop.Name = name
    prop.Value = value
    return prop


def connect(pipe_name: str, attempts: int = 100):
    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    address = (
        f"uno:pipe,name={pipe_name};urp;StarOffice.ComponentContext"
    )
    for _ in range(attempts):
        try:
            return resolver.resolve(address)
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("Không thể kết nối tới LibreOffice để cập nhật trường.")


def update_fields(docx_path: Path) -> None:
    soffice = shutil.which("soffice")
    if soffice is None:
        raise RuntimeError("Không tìm thấy soffice trong PATH.")

    pipe_name = f"thesis_docx_{os.getpid()}"
    with tempfile.TemporaryDirectory(prefix="thesis-libreoffice-") as profile:
        profile_uri = Path(profile).resolve().as_uri()
        process = subprocess.Popen(
            [
                soffice,
                "--headless",
                "--nologo",
                "--nodefault",
                "--nofirststartwizard",
                "--norestore",
                f"-env:UserInstallation={profile_uri}",
                f"--accept=pipe,name={pipe_name};urp;StarOffice.ComponentContext",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        document = None
        try:
            context = connect(pipe_name)
            service_manager = context.ServiceManager
            desktop = service_manager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", context
            )
            document = desktop.loadComponentFromURL(
                uno.systemPathToFileUrl(str(docx_path.resolve())),
                "_blank",
                0,
                (
                    property_value("Hidden", True),
                    property_value("UpdateDocMode", 3),
                ),
            )
            if document is None:
                raise RuntimeError(f"Không thể mở {docx_path} bằng LibreOffice.")

            indexes = document.getDocumentIndexes()
            for index in range(indexes.getCount()):
                indexes.getByIndex(index).update()
            document.TextFields.refresh()
            document.store()
        finally:
            if document is not None:
                document.close(True)
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} FILE.docx")
    update_fields(Path(sys.argv[1]))
