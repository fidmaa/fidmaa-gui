# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src/fidmaa_gui/entrypoints.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src/fidmaa_gui/form.ui', '.'),
    ],
    hiddenimports=['fidmaa_gui'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='fidmaa_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='fidmaa_gui',
)
app = BUNDLE(
    coll,
    name='fidmaa_gui.app',
    icon=None,
    bundle_identifier=None,
)
