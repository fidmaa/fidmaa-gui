#all:
#	@echo "Available targets: clean macos"

all: clean macos

clean:
	find . -name __pycache__ -type d -print0 | xargs -0 rm -rf
	find . -name \*~ -print0 | xargs -0 rm -f
	find . -name \*pyc -print0 | xargs -0 rm -f
	find . -name \*\\.log -print0 | xargs -0 rm -f
	rm -rf .tox build dist

macos:
	pyinstaller --windowed  --add-data "src/fidmaa/form.ui:." src/application.py
	cp -R ~/.virtualenvs/fidmaa/lib/python*/site-packages/cv2/data/ ~/Programowanie/fidmaa/dist/application.app/Contents/MacOS/cv2/data/
	mkdir -p ~/Programowanie/fidmaa/dist/application.app/Contents/MacOS/pyheif/data/
	cp -R ~/.virtualenvs/fidmaa/lib/python*/site-packages/pyheif/data/ ~/Programowanie/fidmaa/dist/application.app/Contents/MacOS/pyheif/data/
