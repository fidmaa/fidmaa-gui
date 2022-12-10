#all:
#	@echo "Available targets: clean macos"

all: clean macos

clean:
	find . -name __pycache__ -type d -print0 | xargs -0 rm -rf
	find . -name \*~ -print0 | xargs -0 rm -f
	find . -name \*pyc -print0 | xargs -0 rm -f
	find . -name \*\\.log -print0 | xargs -0 rm -f
	rm -rf .tox build dist

macos: macos-build macos-copy-files

macos-build:
	pyinstaller --windowed  --add-data "src/fidmaa/form.ui:." src/application.py

VENV_PATH=`poetry env info --path`

macos-copy-files:
	cp -R $(VENV_PATH)/lib/python3*/site-packages/cv2/data/ ./dist/application.app/Contents/MacOS/cv2/data/
	mkdir -p dist/application.app/Contents/MacOS/pyheif/data/
	cp -R $(VENV_PATH)/lib/python3*/site-packages/pyheif/data/ ./dist/application.app/Contents/MacOS/pyheif/data/

zip-app:
	cd dist && zip -r application.app.zip application.app	
